from types import SimpleNamespace

import numpy as np
import pytest

from buzz.model_loader import ModelType, TranscriptionModel
from buzz.transcriber import whisper_npu
from buzz.transcriber.transcriber import (
    FileTranscriptionOptions,
    FileTranscriptionTask,
    Task,
    TranscriptionOptions,
)
from buzz.transcriber.whisper_npu import WhisperNpu

VOCAB = 51865
SOT, EOT = 50258, 50257
TS_FIRST = 50364  # <|0.00|>; 20 ms per step


class FakeDecoderSession:
    """Scripted decoder: one entry per run call.

    An int entry samples that token id; None is a pass-through call (forced
    prefix step, or the language-detection step whose logits the caller reads).
    After the script runs out, every call samples EOT. `decoy` adds a strong
    non-language logit to prove the restricted argmax in detection.
    """

    def __init__(self, calls, decoy=None, strict_inputs=True):
        self.calls = list(calls)
        self.decoy = decoy
        self.strict_inputs = strict_inputs
        self.cross_names = [f"cross_k_{i}" for i in range(6)] + [
            f"cross_v_{i}" for i in range(6)
        ]
        self.in_names = (
            ["input_ids", "position_ids", "attention_mask"]
            + [f"k_cache_self_{i}_in" for i in range(6)]
            + [f"v_cache_self_{i}_in" for i in range(6)]
            + self.cross_names
        )
        self.out_names = ["logits"] + [f"k_cache_self_{i}_out" for i in range(6)] + [
            f"v_cache_self_{i}_out" for i in range(6)
        ]

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self.in_names]

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in self.out_names]

    def run(self, out_names, feed):
        assert set(out_names) == set(self.out_names)
        # Name-fed contract: every declared input must be present.
        if self.strict_inputs:
            assert set(feed) >= set(self.in_names)

        logits = np.zeros(VOCAB, dtype=np.float32)
        if self.calls:
            call = self.calls.pop(0)
            if call is not None:
                logits[call] = 10.0
        else:
            logits[EOT] = 10.0
        if self.decoy is not None:
            decoy_id, decoy_value = self.decoy
            logits[decoy_id] = decoy_value

        outs = {"logits": logits.reshape(1, 1, VOCAB)}
        for i in range(6):
            outs[f"k_cache_self_{i}_out"] = np.zeros((8, 1, 64, 199), dtype=np.float16)
            outs[f"v_cache_self_{i}_out"] = np.zeros((8, 1, 199, 64), dtype=np.float16)
        return [outs[name] for name in out_names]


class FakeEncoderSession:
    def __init__(self):
        self.out_names = FakeDecoderSession([]).cross_names

    def get_inputs(self):
        return [SimpleNamespace(name="input_features")]

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in self.out_names]

    def run(self, out_names, feed):
        assert set(feed) == {"input_features"}
        return [
            np.zeros((8, 1, 64, 1500), dtype=np.float16) for _ in out_names
        ]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import WhisperTokenizer

    return WhisperTokenizer.from_pretrained("openai/whisper-base")


@pytest.fixture(scope="module")
def langs(tokenizer):
    return whisper_npu._whisper_language_ids(tokenizer)


def make_model_stub(tokenizer):
    return SimpleNamespace(
        tokenizer=tokenizer,
        sot=SOT,
        eot=EOT,
        no_timestamps=tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
        ts_first=TS_FIRST,
        ts_last=51864,
        task_ids={
            Task.TRANSCRIBE: tokenizer.convert_tokens_to_ids("<|transcribe|>"),
            Task.TRANSLATE: tokenizer.convert_tokens_to_ids("<|translate|>"),
        },
    )


def text_ids(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def ts_id(seconds):
    return TS_FIRST + int(seconds * 50)


def npu_model_with(tokenizer, dec_session):
    langs = whisper_npu._whisper_language_ids(tokenizer)
    """A _NpuWhisperModel without __init__ (fake sessions, real decode path)."""
    model = whisper_npu._NpuWhisperModel.__new__(whisper_npu._NpuWhisperModel)
    model.tokenizer = tokenizer
    model.sot = SOT
    model.eot = EOT
    model.no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    model.ts_first = TS_FIRST
    model.ts_last = 51864
    model.lang_ids = list(langs.values())
    model.dec_session = dec_session
    model.dec_out_names = [o.name for o in dec_session.get_outputs()]
    model.dec_in_names = [i.name for i in dec_session.get_inputs()]
    model.zero_k = None
    return model


class TestSegmentsFromTokenStream:
    def test_timestamped_segment(self, tokenizer):
        model = make_model_stub(tokenizer)
        ids = [ts_id(0.0)] + text_ids(tokenizer, "Hello world") + [ts_id(2.5)]
        segments = WhisperNpu._segments_from_token_stream(
            model, ids, offset_ms=30_000, window_ms=30_000)
        assert [(s.start, s.end, s.text) for s in segments] == [
            (30_000, 32_500, "Hello world")]

    def test_two_segments(self, tokenizer):
        model = make_model_stub(tokenizer)
        ids = ([ts_id(0.0)] + text_ids(tokenizer, "One")
               + [ts_id(1.0)] + text_ids(tokenizer, "Two") + [ts_id(2.0)])
        segments = WhisperNpu._segments_from_token_stream(
            model, ids, offset_ms=0, window_ms=30_000)
        assert [(s.start, s.end, s.text) for s in segments] == [
            (0, 1_000, "One"), (1_000, 2_000, "Two")]

    def test_text_before_first_timestamp_starts_at_window_start(self, tokenizer):
        model = make_model_stub(tokenizer)
        ids = text_ids(tokenizer, "Early") + [ts_id(1.0)]
        segments = WhisperNpu._segments_from_token_stream(
            model, ids, offset_ms=60_000, window_ms=30_000)
        assert [(s.start, s.end, s.text) for s in segments] == [
            (60_000, 61_000, "Early")]

    def test_trailing_text_ends_at_window_end(self, tokenizer):
        model = make_model_stub(tokenizer)
        ids = [ts_id(0.5)] + text_ids(tokenizer, "Tail")
        segments = WhisperNpu._segments_from_token_stream(
            model, ids, offset_ms=0, window_ms=30_000)
        assert [(s.start, s.end, s.text) for s in segments] == [
            (500, 30_000, "Tail")]

    def test_no_timestamps_is_one_window_segment(self, tokenizer):
        model = make_model_stub(tokenizer)
        ids = text_ids(tokenizer, "All of it")
        segments = WhisperNpu._segments_from_token_stream(
            model, ids, offset_ms=90_000, window_ms=30_000)
        assert [(s.start, s.end, s.text) for s in segments] == [
            (90_000, 120_000, "All of it")]

    def test_sot_skipped_eot_stops_and_ignores_later_tokens(self, tokenizer):
        model = make_model_stub(tokenizer)
        ids = [SOT, ts_id(0.0)] + text_ids(tokenizer, "Hi") + [EOT, ts_id(20.0)]
        segments = WhisperNpu._segments_from_token_stream(
            model, ids, offset_ms=0, window_ms=30_000)
        # "Hi" had no closing timestamp; it ends at the window end (the
        # documented fallback), and the post-EOT timestamp is ignored.
        assert [(s.start, s.end, s.text) for s in segments] == [(0, 30_000, "Hi")]

    def test_empty_text_between_timestamps_is_dropped(self, tokenizer):
        model = make_model_stub(tokenizer)
        ids = [ts_id(0.0), ts_id(1.0), ts_id(2.0)]
        assert WhisperNpu._segments_from_token_stream(
            model, ids, offset_ms=0, window_ms=30_000) == []


class TestBuildPrefix:
    def test_auto_transcribe_frees_the_decode(self, tokenizer):
        model = make_model_stub(tokenizer)
        prefix, suppress = WhisperNpu._build_prefix(
            model, None, Task.TRANSCRIBE, cross_attention={})
        assert prefix == [SOT]
        assert suppress == ()

    def test_explicit_language_forces_prefix_and_masks_notimestamps(self, tokenizer, langs):
        model = make_model_stub(tokenizer)
        model.lang_to_id = langs
        transcribe = tokenizer.convert_tokens_to_ids("<|transcribe|>")
        no_ts = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        prefix, suppress = WhisperNpu._build_prefix(
            model, "fr", Task.TRANSCRIBE, cross_attention={})
        assert prefix == [SOT, langs["fr"], transcribe]
        assert suppress == (no_ts,)

    def test_auto_translate_detects_then_forces(self, tokenizer, langs):
        model = make_model_stub(tokenizer)
        translate = tokenizer.convert_tokens_to_ids("<|translate|>")
        no_ts = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        model.detect_language = lambda cross: langs["fr"]
        prefix, suppress = WhisperNpu._build_prefix(
            model, None, Task.TRANSLATE, cross_attention={})
        assert prefix == [SOT, langs["fr"], translate]
        assert suppress == (no_ts,)

    def test_unknown_language_raises(self, tokenizer, langs):
        model = make_model_stub(tokenizer)
        model.lang_to_id = langs
        with pytest.raises(ValueError, match="Unknown language"):
            WhisperNpu._build_prefix(
                model, "klingon", Task.TRANSCRIBE, cross_attention={})


class TestDetectLanguage:
    def test_restricted_argmax_ignores_higher_non_language_logit(self, tokenizer, langs):
        fr = langs["fr"]
        # The decoy logit (id 1000, value 50) towers over the language token's
        # 10 — only a language-restricted argmax can still return fr.
        dec = FakeDecoderSession([fr], decoy=(1000, 50.0), strict_inputs=False)
        model = npu_model_with(tokenizer, dec)
        assert model.detect_language({}) == fr


def make_task(tmp_path, **options):
    return FileTranscriptionTask(
        transcription_options=TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_NPU),
            **options,
        ),
        file_transcription_options=FileTranscriptionOptions(),
        model_path=str(tmp_path / "asset"),
    )


def patch_engine(monkeypatch, audio, open_session):
    monkeypatch.setattr("buzz.whisper_audio.load_audio", lambda *a, **k: audio)
    monkeypatch.setattr(whisper_npu, "_open_qnn_session", open_session)
    monkeypatch.setattr(whisper_npu, "_pin_blas_threads", lambda: None)
    monkeypatch.setattr(whisper_npu, "_maybe_pin_affinity", lambda: None)


class TestTranscribeEndToEnd:
    def test_auto_transcribe_produces_segments(self, tokenizer, langs, tmp_path, monkeypatch):
        audio = np.zeros(16000 // 2, dtype=np.float32)  # 0.5 s -> one window
        gen = ([langs["en"],
                tokenizer.convert_tokens_to_ids("<|transcribe|>"),
                ts_id(0.0)]
               + text_ids(tokenizer, "Hi there") + [ts_id(1.0)])

        def open_session(path):
            if "encoder" in path:
                return FakeEncoderSession()
            return FakeDecoderSession(gen)

        patch_engine(monkeypatch, audio, open_session)
        segments = WhisperNpu.transcribe(make_task(tmp_path))
        assert [(s.start, s.end, s.text) for s in segments] == [
            (0, 1_000, "Hi there")]

    def test_auto_translate_detects_then_forces(self, tokenizer, langs, tmp_path, monkeypatch):
        audio = np.zeros(16000 // 2, dtype=np.float32)
        fr = langs["fr"]
        # Call order: detect (samples fr), then the forced prefix's two
        # pass-through steps, then the sampled timed text.
        calls = [fr, None, None] + text_ids(tokenizer, "Bonjour") + [ts_id(0.8)]

        def open_session(path):
            if "encoder" in path:
                return FakeEncoderSession()
            return FakeDecoderSession(calls)

        patch_engine(monkeypatch, audio, open_session)
        task = make_task(tmp_path, task=Task.TRANSLATE)
        segments = WhisperNpu.transcribe(task)
        assert [(s.start, s.end, s.text) for s in segments] == [
            (0, 800, "Bonjour")]

    def test_multi_window_offsets(self, tokenizer, tmp_path, monkeypatch):
        # 61 s of audio -> windows at 0 s / 30 s / 60 s; the middle window's
        # segment must carry the 30 s offset, the silent last window nothing.
        audio = np.zeros(16000 * 61, dtype=np.float32)
        # One decoder session serves all windows; each window's decode loop
        # stops at EOT, so the scripts concatenate with EOT separators.
        calls = ([ts_id(0.0)] + text_ids(tokenizer, "First") + [ts_id(2.0), EOT]
                 + [ts_id(0.0)] + text_ids(tokenizer, "Second") + [ts_id(3.0), EOT])
        dec = FakeDecoderSession(calls)

        def open_session(path):
            if "encoder" in path:
                return FakeEncoderSession()
            return dec

        patch_engine(monkeypatch, audio, open_session)
        segments = WhisperNpu.transcribe(make_task(tmp_path))
        assert [(s.start, s.end, s.text) for s in segments] == [
            (0, 2_000, "First"), (30_000, 33_000, "Second")]

    def test_explicit_language_forces_prefix(self, tokenizer, langs, tmp_path, monkeypatch):
        audio = np.zeros(16000 // 2, dtype=np.float32)
        # Forced [SOT, fr, transcribe]: two pass-through calls then text.
        calls = [None, None] + text_ids(tokenizer, "Bonjour") + [ts_id(0.5)]

        def open_session(path):
            if "encoder" in path:
                return FakeEncoderSession()
            return FakeDecoderSession(calls)

        patch_engine(monkeypatch, audio, open_session)
        task = make_task(tmp_path, language="fr")
        segments = WhisperNpu.transcribe(task)
        assert [(s.start, s.end, s.text) for s in segments] == [
            (0, 500, "Bonjour")]


class TestHallucinationLoopGuard:
    @staticmethod
    def _word_ids(tokenizer, words):
        ids = []
        for word in words:
            # Leading space inside the phrase = real BPE Ġ tokens, so the
            # decoded text splits into words the same way engine output does.
            ids.extend(text_ids(tokenizer, " " + word))
        return ids

    def test_repetitive_cap_hit_is_dropped(self, tokenizer, tmp_path, monkeypatch):
        # 199 sampled tokens of the same two words, never EOT -> cap hit with
        # repetitive text -> the window must be dropped entirely.
        loop_ids = self._word_ids(tokenizer, ["gentle", "music"]) * 100
        audio = np.zeros(16000 // 2, dtype=np.float32)

        def open_session(path):
            if "encoder" in path:
                return FakeEncoderSession()
            return FakeDecoderSession(loop_ids)

        patch_engine(monkeypatch, audio, open_session)
        segments = WhisperNpu.transcribe(make_task(tmp_path))
        assert segments == []

    def test_varied_cap_hit_is_kept(self, tokenizer, tmp_path, monkeypatch):
        # Full-length decode without EOT but varied words (ratio > 0.2) is
        # dense legit speech: kept, trailing flush at the window end.
        vocab_words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                       "golf", "hotel", "india", "juliett", "kilo", "lima",
                       "mike", "november", "oscar", "papa", "quebec", "romeo",
                       "sierra", "tango", "uniform", "victor", "whiskey",
                       "xray", "yankee", "zulu", "east", "west", "north",
                       "south", "river", "mountain", "forest", "desert",
                       "ocean", "valley", "storm", "breeze", "sunrise",
                       "sunset", "harbor", "meadow", "canyon", "island",
                       "prairie", "tundra", "jungle", "lagoon", "summit",
                       "ridge"]
        varied_ids = self._word_ids(tokenizer, vocab_words) * 5
        audio = np.zeros(16000 // 2, dtype=np.float32)

        def open_session(path):
            if "encoder" in path:
                return FakeEncoderSession()
            return FakeDecoderSession(varied_ids)

        patch_engine(monkeypatch, audio, open_session)
        segments = WhisperNpu.transcribe(make_task(tmp_path))
        assert len(segments) == 1
        assert (segments[0].start, segments[0].end) == (0, 30_000)
        assert len(segments[0].text.split()) > 50

    def test_is_repetitive_text_thresholds(self, tokenizer):
        model = make_model_stub(tokenizer)
        loop = self._word_ids(tokenizer, ["gentle", "music"]) * 50
        assert WhisperNpu._is_repetitive_text(model, loop) is True
        varied = self._word_ids(tokenizer, [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
            "hotel", "india", "juliett", "kilo", "lima", "mike", "november",
            "oscar", "papa", "quebec", "romeo", "sierra", "tango"]) * 3
        assert WhisperNpu._is_repetitive_text(model, varied) is False
        short = self._word_ids(tokenizer, ["la", "la"]) * 5
        assert WhisperNpu._is_repetitive_text(model, short) is False  # < 12 words


class TestWordLevelSegments:
    """CTC-aligner wiring: dict mapping + failure fallback to segment level."""

    @staticmethod
    def make_segments():
        from buzz.transcriber.transcriber import Segment

        return [Segment(start=0, end=2000, text="Hello world"),
                Segment(start=2000, end=4000, text="Goodbye")]

    def test_word_dicts_mapped_to_segments(self, monkeypatch):
        monkeypatch.setattr(
            "buzz.transcriber.whisper_npu.WhisperNpu._align_words",
            lambda audio, segments, language: [
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.6, "end": 1.9},
                {"text": "Goodbye", "start": 2.1, "end": 3.8},
            ])
        out = WhisperNpu._word_level_segments(
            np.zeros(16000, dtype=np.float32), self.make_segments(),
            SimpleNamespace(language=None))
        assert [(s.start, s.end, s.text) for s in out] == [
            (0, 500, "Hello"), (600, 1900, "world"), (2100, 3800, "Goodbye")]

    def test_aligner_failure_falls_back_to_segments(self, monkeypatch):
        def boom(audio, segments, language):
            raise RuntimeError("model download failed")

        monkeypatch.setattr(
            "buzz.transcriber.whisper_npu.WhisperNpu._align_words", boom)
        segments = self.make_segments()
        out = WhisperNpu._word_level_segments(
            np.zeros(16000, dtype=np.float32), segments,
            SimpleNamespace(language="en"))
        assert out is segments  # segment-level output preserved

    def test_aligner_raises_falls_back_to_segments(self, monkeypatch):
        from buzz.transcriber.speaker_identifier import AlignmentFailedError

        def mismatch(audio, segments, language):
            raise AlignmentFailedError("transcript does not match audio")

        monkeypatch.setattr(
            "buzz.transcriber.whisper_npu.WhisperNpu._align_words", mismatch)
        segments = self.make_segments()
        out = WhisperNpu._word_level_segments(
            np.zeros(16000, dtype=np.float32), segments,
            SimpleNamespace(language="en"))
        assert out is segments

    def test_empty_segments_short_circuit(self, monkeypatch):
        monkeypatch.setattr(
            "buzz.transcriber.whisper_npu.WhisperNpu._align_words",
            lambda *a: (_ for _ in ()).throw(AssertionError("must not be called")))
        assert WhisperNpu._word_level_segments(
            np.zeros(1, dtype=np.float32), [], SimpleNamespace(language=None)) == []


class TestNpuCtcEmissions:
    """The w8a16 wav2vec2 emissions backend (unit: no device needed)."""

    def test_discover_none_without_env(self, monkeypatch):
        monkeypatch.delenv("BUZZ_WAV2VEC2_ALIGNER_DIR", raising=False)
        assert whisper_npu._NpuCtcEmissions.discover() is None

    def test_discover_none_when_asset_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BUZZ_WAV2VEC2_ALIGNER_DIR", str(tmp_path))
        assert whisper_npu._NpuCtcEmissions.discover() is None

    def test_preprocess_no_stars(self):
        fn = whisper_npu._preprocess_text_no_stars
        tokens, texts = fn("Hello world", romanize=True, language="eng")
        assert "<star>" not in tokens
        assert "<star>" not in texts
        assert "h e l l o" in tokens and "w o r l d" in tokens

    def test_emissions_stitching_and_dequantization(self, monkeypatch, tmp_path):
        import json as _json

        asset = tmp_path / "asset"
        asset.mkdir()
        metadata = {
            "model_files": {"wav2vec2_conformer.onnx": {
                "inputs": {"input_values": {"quantization_parameters": {
                    "scale": 0.001, "zero_point": 1000.0}}},
                "outputs": {"logits": {"quantization_parameters": {
                    "scale": 0.01, "zero_point": 500.0}}}}}}
        (asset / "metadata.json").write_text(_json.dumps(metadata))
        (asset / "wav2vec2_conformer.onnx").write_bytes(b"x")

        emissions_cls = whisper_npu._NpuCtcEmissions
        model = emissions_cls.__new__(emissions_cls)
        model.asset_dir = str(asset)
        model.in_scale, model.in_zp = 0.001, 1000.0
        model.out_scale, model.out_zp = 0.01, 500.0

        class FakeSession:
            def __init__(self):
                self.feeds = []

            def get_inputs(self):
                return [SimpleNamespace(name="input_values")]

            def run(self, _, feed):
                self.feeds.append(feed["input_values"])
                # quantized value 1500 -> (1500-500)*0.01 = 10.0
                return [np.full((1, 499, 32), 1500, dtype=np.uint16)]

        fake = FakeSession()
        model.session = fake
        model.input_name = "input_values"

        audio = np.zeros(320000, dtype=np.float32)  # 20 s -> 2 windows
        out = model.emissions(audio)

        assert out.shape == (2 * 499, 32)
        assert np.allclose(out, 10.0)
        # Two windows advanced by the model's own frame grid
        assert len(fake.feeds) == 2
        assert fake.feeds[0].shape == (1, 160000)


class TestNpuFallback:
    """File-path fallback to whisper.cpp when the NPU engine fails."""

    @staticmethod
    def make_task():
        return FileTranscriptionTask(
            transcription_options=TranscriptionOptions(
                model=TranscriptionModel(model_type=ModelType.WHISPER_NPU)),
            file_transcription_options=FileTranscriptionOptions(),
            model_path="fake-asset-dir",
        )

    def test_falls_back_to_whisper_cpp_with_replaced_model_path(self, monkeypatch):
        from buzz.transcriber import whisper_file_transcriber as wft
        from buzz.transcriber.transcriber import Segment

        def failing_npu(task):
            raise RuntimeError("QNN context load failed")

        seen = {}

        def fake_whisper_cpp(task):
            seen["model_path"] = task.model_path
            return [Segment(start=0, end=100, text="via whisper.cpp")]

        monkeypatch.setattr(
            "buzz.transcriber.whisper_npu.WhisperNpu.transcribe",
            staticmethod(failing_npu))
        monkeypatch.setattr(wft.WhisperCpp, "transcribe", staticmethod(fake_whisper_cpp))
        monkeypatch.setattr(
            wft.WhisperFileTranscriber, "_whisper_cpp_fallback_model_path",
            staticmethod(lambda: "/models/ggml-base.bin"))

        out = wft.WhisperFileTranscriber.transcribe_whisper_npu(self.make_task())

        assert [s.text for s in out] == ["via whisper.cpp"]
        assert seen["model_path"] == "/models/ggml-base.bin"
        # every other field of the task is preserved
        assert seen["model_path"] != "fake-asset-dir"

    def test_no_fallback_model_raises_combined_error(self, monkeypatch):
        import pytest

        from buzz.transcriber import whisper_file_transcriber as wft

        def failing_npu(task):
            raise RuntimeError("QNN context load failed")

        monkeypatch.setattr(
            "buzz.transcriber.whisper_npu.WhisperNpu.transcribe",
            staticmethod(failing_npu))
        monkeypatch.setattr(
            wft.WhisperFileTranscriber, "_whisper_cpp_fallback_model_path",
            staticmethod(lambda: None))

        with pytest.raises(Exception, match="QNN context load failed.*Whisper.cpp"):
            wft.WhisperFileTranscriber.transcribe_whisper_npu(self.make_task())

    def test_success_does_not_touch_fallback(self, monkeypatch):
        from buzz.transcriber import whisper_file_transcriber as wft
        from buzz.transcriber.transcriber import Segment

        monkeypatch.setattr(
            "buzz.transcriber.whisper_npu.WhisperNpu.transcribe",
            staticmethod(lambda task: [Segment(start=0, end=1, text="npu")]))
        monkeypatch.setattr(
            wft.WhisperFileTranscriber, "_whisper_cpp_fallback_model_path",
            staticmethod(lambda: (_ for _ in ()).throw(AssertionError("not needed"))))
        out = wft.WhisperFileTranscriber.transcribe_whisper_npu(self.make_task())
        assert [s.text for s in out] == ["npu"]
