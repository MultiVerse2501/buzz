"""Whisper NPU engine — Hexagon-accelerated transcription via the ORT-QNN stack.

P2 milestone M2 (porting/plans/P2/M2-file-transcription.md). The decode loop is
the validated prototype (porting/scripts/npu/whsample_bench_utterances.py +
npu_forcing_test.py; numerics proven in P2-npu-benchmark-report.md §2/§7):

- ORT 1.27 + onnxruntime-qnn 2.6.0 sessions fed strictly BY NAME (asset I/O
  order varies per export);
- right-aligned -100 attention mask over a 200-token decode window, KV-cache
  rotation per step;
- auto language = free-run from <|startoftranscript|>; explicit language (or
  translate) = forced [SOT, lang, task] prefix with <|notimestamps|> masked out
  so timestamp tokens keep flowing (verified recipe, report §7.6);
- whisper timestamp tokens (20 ms steps) close Segments; a window with no
  timestamp tokens degrades to one window-length Segment.

The engine runs inside the WhisperFileTranscriber worker subprocess; progress
is written to stderr as "NN%" lines and cancellation is the process kill.
"""
import ctypes
import logging
import os
import re
import sys
from typing import List, Optional

from buzz.transcriber.transcriber import FileTranscriptionTask, Segment, Task

# The whisper-base assets decode on the HTP while the host side does small
# tensor work; competing BLAS threads cost measurable time (report §4). These
# only help if set before the libraries load, so also pin the already-loaded
# OpenBLAS below.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = logging.getLogger(__name__)

SOT = 50258
EOT = 50257
DECODE_LEN = 200
MASK_NEG = -100.0
N_LAYERS, N_HEADS, HEAD_DIM = 6, 8, 64
# Timestamp tokens: <|0.00|> .. <|30.00|>, 20 ms per step (openai/whisper-base).
TIMESTAMP_TOKEN = "<|notimestamps|>"
WINDOW_SECONDS = 30
WHISPER_SAMPLE_RATE = 16000

# Vendor-default QNN EP options (qai_hub_models aihub_defaults), the measured
# configuration — do not add or change options here without re-benchmarking.
QNN_PROVIDER_OPTIONS = {
    "enable_htp_fp16_precision": "1",
    "htp_performance_mode": "burst",
    "htp_graph_finalization_optimization_mode": "3",
    "offload_graph_io_quantization": "1",
}


def _pin_blas_threads() -> None:
    """Best-effort: pin already-loaded OpenBLAS to one thread (report §4).

    numpy is imported before this module in the worker process, so the env
    vars above cannot reach its BLAS anymore. Never fatal.
    """
    try:
        import psutil

        for mapping in psutil.Process().memory_maps():
            path = mapping.path
            if "openblas" not in os.path.basename(path).lower():
                continue
            lib = ctypes.CDLL(path)
            for name in ("openblas_set_num_threads", "openblas_set_num_threads64_"):
                setter = getattr(lib, name, None)
                if setter is not None:
                    setter(1)
                    logger.debug("Pinned %s to 1 thread via %s", path, name)
                    return
    except Exception:
        logger.debug("OpenBLAS thread pin failed", exc_info=True)


def _open_qnn_session(model_path: str):
    import onnxruntime as ort

    from buzz.npu_capability import get_qnn_ep_devices

    devices = get_qnn_ep_devices()
    if not devices:
        raise RuntimeError(
            "QNN execution provider has no devices; the NPU engine cannot run"
        )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.add_provider_for_devices(devices, QNN_PROVIDER_OPTIONS)
    return ort.InferenceSession(model_path, sess_options=session_options)


def _report_progress(percent: int) -> None:
    """Emit a progress line the transcription parent parses off stderr.

    In the worker, sys.stderr is buzz's ConnWriter (write-only; every write is
    one message down the multiprocessing pipe) -- never call flush() here.
    """
    sys.stderr.write(f"{percent}%{chr(10)}")


def _pad(audio, target_len: int, noise_floor: bool = False):
    """Pad a short chunk to a full 30-s window, deterministically.

    noise_floor=True pads with a tiny noise (~-60 dBFS) instead of exact
    zeros: whisper-base's translate mode hallucinates on long exact-silence
    tails ("I'm going to the beach" loops), and low-level noise dampens that.
    Transcribe decodes measure measurably better with plain zero padding
    (test-clean: 4.85% vs 6.41% WER), so callers pick per task. Seeded by the
    audio length so the same input always produces the same pad.
    """
    import numpy as np

    pad_n = target_len - len(audio)
    if pad_n <= 0:
        return audio
    if not noise_floor:
        return np.concatenate([audio, np.zeros(pad_n, dtype=np.float32)])
    noise = np.random.default_rng(len(audio)).standard_normal(pad_n).astype(np.float32)
    return np.concatenate([audio, noise * 1e-3])


def _maybe_pin_affinity() -> None:
    """Apply BUZZ_NPU_AFFINITY_MASK (hex) to the process when set.

    Unpinned is near-best on the ORT stack (report §4), so this is opt-in only:
    pinning the whole app process would also drag the GUI onto the chosen cores.
    """
    mask = os.getenv("BUZZ_NPU_AFFINITY_MASK")
    if not mask:
        return
    if sys.platform != "win32":
        return
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.kernel32.SetProcessAffinityMask(handle, int(mask, 16))
        logger.debug("BUZZ_NPU_AFFINITY_MASK %s: set=%s", mask, bool(ok))
    except Exception:
        logger.debug("Failed to apply BUZZ_NPU_AFFINITY_MASK", exc_info=True)


def warm_up() -> None:
    """Process-level prep before the first NPU decode in this process.

    Called by both the file-transcription worker (via transcribe()) and the
    in-process recording path: the env vars set at module import cannot reach
    an already-loaded OpenBLAS, so the loaded library is pinned directly.
    """
    _maybe_pin_affinity()
    _pin_blas_threads()


_LANGUAGE_TOKEN_RE = re.compile(r"^<\|([a-z]{2,3})\|>$")


def _whisper_language_ids(tokenizer) -> dict:
    """language code -> token id, built from the tokenizer's added tokens.

    transformers 5 removed WhisperTokenizer.lang_to_id; the added-vocab scan
    is version-proof (whisper-base: 99 languages, en=50259 .. su=50357).
    """
    languages = {}
    for token, token_id in tokenizer.get_added_vocab().items():
        match = _LANGUAGE_TOKEN_RE.match(token)
        if match:
            languages[match.group(1)] = token_id
    return languages


class _NpuWhisperModel:
    """Loaded NPU asset: encoder/decoder sessions + tokenizer + feature extractor."""

    def __init__(self, model_dir: str):
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-base"
        )

        self.sot = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.eot = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.no_timestamps = self.tokenizer.convert_tokens_to_ids(TIMESTAMP_TOKEN)
        self.ts_first = self.tokenizer.convert_tokens_to_ids("<|0.00|>")
        self.ts_last = self.tokenizer.convert_tokens_to_ids("<|30.00|>")
        self.task_ids = {
            Task.TRANSCRIBE: self.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
            Task.TRANSLATE: self.tokenizer.convert_tokens_to_ids("<|translate|>"),
        }
        # Language-token ids for the restricted-argmax detection step and for
        # explicit-language prefixes.
        self.lang_to_id = _whisper_language_ids(self.tokenizer)
        self.lang_ids = list(self.lang_to_id.values())

        self.enc_session = _open_qnn_session(os.path.join(model_dir, "encoder.onnx"))
        self.dec_session = _open_qnn_session(os.path.join(model_dir, "decoder.onnx"))
        self.enc_out_names = [o.name for o in self.enc_session.get_outputs()]
        self.dec_in_names = [i.name for i in self.dec_session.get_inputs()]
        self.dec_out_names = [o.name for o in self.dec_session.get_outputs()]

        self.zero_k = None  # built lazily (device dtype)

    def encode(self, features) -> dict:
        """Run the encoder, returning cross-attention tensors keyed by name."""
        outputs = self.enc_session.run(
            self.enc_out_names, {"input_features": features}
        )
        return dict(zip(self.enc_out_names, outputs))

    def detect_language(self, cross_attention: dict) -> int:
        """One-step language detection: argmax over language ids after SOT.

        The same restricted-argmax shape OpenAI's whisper uses for its
        detect_language pass.
        """
        logits = self._decode_step(
            prefix_ids=[self.sot], cross_attention=cross_attention, max_steps=1
        )[0]
        return max(self.lang_ids, key=lambda tid: logits[tid])

    def decode(self, prefix_ids: List[int], cross_attention: dict,
               suppress: tuple = ()) -> List[int]:
        """Greedy-generate after `prefix_ids` (force-fed, KV built) up to EOT.

        `suppress` holds token ids masked to -inf before every argmax (used to
        keep <|notimestamps|> out of forced-prefix runs). Returns the forced
        prefix followed by the generated ids (excluding EOT).
        """
        return self._decode_step(
            prefix_ids=prefix_ids,
            cross_attention=cross_attention,
            max_steps=DECODE_LEN - 1,
            suppress=suppress,
        )[1]

    def _decode_step(self, prefix_ids: List[int], cross_attention: dict,
                     max_steps: int, suppress: tuple = ()):
        """Run the decoder; returns (first-step logits, generated token ids).

        One step at a time with rotated KV caches and the right-aligned -100
        attention mask from the validated prototype. The prefix (SOT,
        language, task) is fed in order, one token per position — matching
        Qualcomm's reference decode (ai-hub-models hf_whisper app) — and the
        sampled tokens continue from the position after the last prefix
        token. An earlier revision seeded the stream with the whole prefix
        and fed its tail first, so the cache conditioned on
        [task, lang, task] with SOT never entered; on weak-language content
        that drove non-terminating loops (recorded 2026-11-10 in the porting
        plans).
        """
        import numpy as np

        if self.zero_k is None:
            self.zero_k = np.zeros(
                (N_HEADS, 1, HEAD_DIM, DECODE_LEN - 1), dtype=np.float16)
            self.zero_v = np.zeros(
                (N_HEADS, 1, DECODE_LEN - 1, HEAD_DIM), dtype=np.float16)

        sampled: List[int] = []
        pos = np.array([0], dtype=np.int32)
        attn = np.full((1, 1, 1, DECODE_LEN), MASK_NEG, dtype=np.float16)
        state = {}
        for layer in range(N_LAYERS):
            state[f"k_cache_self_{layer}_in"] = self.zero_k
            state[f"v_cache_self_{layer}_in"] = self.zero_v

        first_logits = None
        for step in range(max_steps):
            attn[:, :, :, DECODE_LEN - step - 1] = 0.0
            feed = dict(state)
            feed.update(cross_attention)
            fed_id = prefix_ids[step] if step < len(prefix_ids) else sampled[-1]
            feed["input_ids"] = np.array([[fed_id]], dtype=np.int32)
            feed["position_ids"] = pos
            feed["attention_mask"] = attn
            outputs = self.dec_session.run(self.dec_out_names, feed)
            by_name = dict(zip(self.dec_out_names, outputs))

            logits = by_name["logits"].reshape(-1)
            if first_logits is None:
                first_logits = logits
            if step >= len(prefix_ids) - 1:
                if suppress:
                    logits = logits.copy()
                    for token_id in suppress:
                        logits[token_id] = -np.inf
                token_id = int(logits.argmax())
                if token_id == self.eot:
                    break
                sampled.append(token_id)

            for layer in range(N_LAYERS):
                state[f"k_cache_self_{layer}_in"] = by_name[f"k_cache_self_{layer}_out"]
                state[f"v_cache_self_{layer}_in"] = by_name[f"v_cache_self_{layer}_out"]
            pos = pos + 1

        return first_logits, list(prefix_ids) + sampled


_WAV2VEC2_ALIGNER_DIR_ENV = "BUZZ_WAV2VEC2_ALIGNER_DIR"


def _preprocess_text_no_stars(text, romanize, language):
    """``preprocess_text`` without ``<star>`` wildcard tokens.

    The w8a16 wav2vec2 asset's 32-symbol vocab has no room for the wildcard id
    the aligner appends, and single-chunk alignment (this engine's per-chunk
    flow) does not need inter-segment wildcards.
    """
    from ctc_forced_aligner.ctc_forced_aligner import preprocess_text

    tokens, texts = preprocess_text(text, romanize=romanize, language=language)
    return (
        [t for t in tokens if t != "<star>"],
        [t for t in texts if t != "<star>"],
    )


class _NpuCtcEmissions:
    """Aligner CTC emissions computed on the HTP via the w8a16 wav2vec2 asset.

    Qualcomm ships the conformer-large-960h export on the same public release
    bucket as the whisper assets. Fixed 10-s windows, 20 ms/frame; IO is
    affine-quantized uint16 (scale/zero_point from the asset's metadata.json).
    Windows advance by the model's own frame grid (499 frames x 320-sample
    stride) so the stitched frame timeline has no cumulative drift.
    """

    WINDOW_SAMPLES = 160000
    WINDOW_STRIDE_SAMPLES = 159680  # 499 frames x 320-sample model stride
    FRAMES_PER_WINDOW = 499
    STRIDE_MS = 20.0

    def __init__(self, asset_dir: str):
        import json

        import onnxruntime as ort

        from buzz.npu_capability import get_qnn_ep_devices

        self.asset_dir = asset_dir
        with open(os.path.join(asset_dir, "metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)
        io_spec = metadata["model_files"]["wav2vec2_conformer.onnx"]
        quant = io_spec["inputs"]["input_values"]["quantization_parameters"]
        self.in_scale, self.in_zp = quant["scale"], quant["zero_point"]
        quant = io_spec["outputs"]["logits"]["quantization_parameters"]
        self.out_scale, self.out_zp = quant["scale"], quant["zero_point"]

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.add_provider_for_devices(
            get_qnn_ep_devices(), QNN_PROVIDER_OPTIONS
        )
        self.session = ort.InferenceSession(
            os.path.join(asset_dir, "wav2vec2_conformer.onnx"),
            sess_options=session_options,
        )
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def discover(cls) -> Optional["_NpuCtcEmissions"]:
        """The configured aligner asset, or None (caller falls back to CPU).

        Point ``BUZZ_WAV2VEC2_ALIGNER_DIR`` at an extracted
        wav2vec2_conformer-precompiled_qnn_onnx-w8a16-* directory. Download /
        bundling of this asset is not productized yet — without it the aligner
        runs its acoustic model on CPU (MMS-300m) exactly as before.
        """
        asset_dir = os.getenv(_WAV2VEC2_ALIGNER_DIR_ENV)
        if not asset_dir or not os.path.isfile(
            os.path.join(asset_dir, "wav2vec2_conformer.onnx")
        ):
            return None
        try:
            return cls(asset_dir)
        except Exception:
            logger.warning("NPU aligner asset failed to load", exc_info=True)
            return None

    def emissions(self, audio):
        """Stitched (frames, 32) float32 logits for 16 kHz float32 mono audio."""
        import numpy as np

        windows = []
        for start in range(0, len(audio), self.WINDOW_STRIDE_SAMPLES):
            window = audio[start: start + self.WINDOW_SAMPLES]
            if len(window) < 1600:  # <0.1 s tail: nothing alignable
                break
            window = (window - window.mean()) / np.sqrt(window.var() + 1e-7)
            padded = np.zeros(self.WINDOW_SAMPLES, dtype=np.float32)
            padded[: len(window)] = window
            q = np.round(
                padded.reshape(1, -1) / self.in_scale + self.in_zp
            ).clip(0, 65535).astype(np.uint16)
            logits = self.session.run(None, {self.input_name: q})[0][0]
            windows.append(
                (logits.astype(np.float32) - self.out_zp) * self.out_scale
            )
            if len(window) < self.WINDOW_SAMPLES:
                break
        return np.concatenate(windows, axis=0) if windows else np.zeros((0, 32), np.float32)


_SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
_SILERO_VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
VAD_SAMPLES_PER_CHUNK = 512  # 32 ms at 16 kHz
VAD_CONTEXT_SAMPLES = 64  # prepended to every chunk (v5 feeding contract)
VAD_SPEECH_ENTER = 0.5
VAD_SPEECH_EXIT = 0.35
VAD_MAX_MERGE_GAP_S = 0.9  # tuned on librispeech_long vs reference (see plans)
VAD_MIN_SPEECH_S = 0.2
VAD_PAD_S = 0.15
MAX_UTTERANCE_S = 28  # one decode window per utterance
# Short inputs are already single utterances: plain windows equal
# per-utterance there, without VAD's span-edge clipping risk (measured on
# single-utterance files: windowed 4.85% vs VAD 9.25% WER).
VAD_MIN_FILE_S = 60.0


def _ensure_vad_onnx() -> Optional[str]:
    """Path to the Silero VAD model, downloading it on first use.

    Returns None when the model is unavailable (the caller decodes plain
    30-s windows instead): VAD is a quality enhancement, never a failure.
    """
    import hashlib

    try:
        from buzz.model_loader import model_root_dir

        vad_dir = os.path.join(model_root_dir, "silero-vad")
        vad_path = os.path.join(vad_dir, "silero_vad.onnx")
        if os.path.isfile(vad_path):
            return vad_path
        os.makedirs(vad_dir, exist_ok=True)
        import requests

        response = requests.get(_SILERO_VAD_URL, timeout=60)
        response.raise_for_status()
        data = response.content
        if hashlib.sha256(data).hexdigest() != _SILERO_VAD_SHA256:
            raise RuntimeError("Silero VAD download failed the sha256 check")
        with open(vad_path, "wb") as f:
            f.write(data)
        return vad_path
    except Exception:
        logger.warning("Silero VAD unavailable; using plain 30-s windows", exc_info=True)
        return None


class _SileroVad:
    """Speech probability per 32-ms chunk of 16 kHz float32 mono audio.

    Silero v5 feeding contract: every chunk is fed as [1, 64 + 512] -- a
    64-sample context prepended to the new samples -- together with the
    recurrent state; the output is the speech probability for the chunk.
    """

    def __init__(self, onnx_path: str):
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._np = np
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, VAD_CONTEXT_SAMPLES), dtype=np.float32)

    def speech_probs(self, audio) -> List[float]:
        np = self._np
        sr = np.array(WHISPER_SAMPLE_RATE, dtype=np.int64)
        probs = []
        for start in range(0, len(audio) - VAD_SAMPLES_PER_CHUNK + 1,
                           VAD_SAMPLES_PER_CHUNK):
            chunk = audio[start: start + VAD_SAMPLES_PER_CHUNK].reshape(1, -1)
            feed_input = np.concatenate([self._context, chunk], axis=1)
            output, self._state = self.session.run(None, {
                "input": feed_input, "state": self._state, "sr": sr,
            })
            probs.append(float(output[0][0]))
            self._context = chunk[:, -VAD_CONTEXT_SAMPLES:]
        return probs


def _speech_spans(probs: List[float]) -> List[tuple]:
    """(start_s, end_s) speech spans from per-32-ms-chunk probabilities.

    Hysteresis (enter >= 0.5, hold until < 0.35), short gaps merged
    (mid-sentence pauses), blips dropped, boundaries padded, and long runs
    hard-split so every utterance fits one decode window.
    """
    seconds_per_chunk = VAD_SAMPLES_PER_CHUNK / WHISPER_SAMPLE_RATE
    spans = []
    start = None
    for index, prob in enumerate(probs):
        time_s = index * seconds_per_chunk
        if start is None:
            if prob >= VAD_SPEECH_ENTER:
                start = time_s
        elif prob < VAD_SPEECH_EXIT:
            spans.append((start, time_s))
            start = None
    if start is not None:
        spans.append((start, len(probs) * seconds_per_chunk))

    merged = []
    for span in spans:
        if merged and span[0] - merged[-1][1] <= VAD_MAX_MERGE_GAP_S:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    merged = [s for s in merged if s[1] - s[0] >= VAD_MIN_SPEECH_S]

    result = []
    for span_start, span_end in merged:
        span_start = max(0.0, span_start - VAD_PAD_S)
        span_end += VAD_PAD_S
        parts = int(-(-(span_end - span_start) // MAX_UTTERANCE_S))
        for part in range(parts):
            result.append((
                span_start + part * MAX_UTTERANCE_S,
                min(span_end, span_start + (part + 1) * MAX_UTTERANCE_S),
            ))
    return result


class WhisperNpu:
    @staticmethod
    def transcribe(task: FileTranscriptionTask) -> List[Segment]:
        warm_up()

        model = _NpuWhisperModel(task.model_path)

        from buzz import whisper_audio

        audio = whisper_audio.load_audio(task.file_path)
        segments = WhisperNpu.transcribe_audio(
            model, audio, task.transcription_options, utterance_split=True
        )
        if task.transcription_options.word_level_timings:
            return WhisperNpu._word_level_segments(audio, segments, task.transcription_options)
        return segments

    @classmethod
    def transcribe_audio(cls, model: _NpuWhisperModel, audio, options,
                         utterance_split: bool = False) -> List[Segment]:
        """Decode 16 kHz float32 mono audio with a loaded model.

        utterance_split=True decodes VAD-detected speech spans as independent
        utterances (file transcription: utterance boundaries restore the
        per-utterance accuracy the plain windows lose); the default plain
        30-s windows serve the recording path and any caller that already
        holds stream chunks. VAD unavailability silently falls back to the
        plain windows.
        """
        task_kind = options.task
        language = options.language or None

        spans = None
        if utterance_split:
            vad = WhisperNpu._load_vad()
            if vad is not None:
                spans = _speech_spans(vad.speech_probs(audio))
                spans = WhisperNpu._keep_vad_spans_only_when_useful(audio, spans)

        segments: List[Segment] = []
        if spans:
            total_s = max(1.0, len(audio) / WHISPER_SAMPLE_RATE)
            for start_s, end_s in spans:
                start_sample = int(start_s * WHISPER_SAMPLE_RATE)
                window = audio[start_sample: int(end_s * WHISPER_SAMPLE_RATE)]
                segments.extend(cls._decode_window(
                    model, window, language, task_kind,
                    offset_ms=int(start_s * 1000),
                    label=f"utterance at {start_s:.1f}s",
                ))
                _report_progress(int(100 * min(1.0, end_s / total_s)))
            return segments

        total_windows = max(1, -(-len(audio) // (WHISPER_SAMPLE_RATE * WINDOW_SECONDS)))
        for window_index in range(total_windows):
            window = audio[
                window_index * WHISPER_SAMPLE_RATE * WINDOW_SECONDS:
                (window_index + 1) * WHISPER_SAMPLE_RATE * WINDOW_SECONDS
            ]
            if len(window) == 0:
                continue
            segments.extend(cls._decode_window(
                model, window, language, task_kind,
                offset_ms=window_index * WINDOW_SECONDS * 1000,
                label=f"window {window_index}",
            ))
            # The worker's stderr is regex-parsed by the parent for progress.
            _report_progress(int(100 * (window_index + 1) / total_windows))

        return segments

    @staticmethod
    def _keep_vad_spans_only_when_useful(audio, spans: List[tuple]) -> Optional[List[tuple]]:
        """The VAD spans when they add boundary value, None for plain windows.

        Short inputs and inputs that are one continuous speech run are already
        single utterances — decoding them windowed is per-utterance behavior
        with none of VAD's span-edge clipping risk.
        """
        if not spans:
            return None
        duration_s = len(audio) / WHISPER_SAMPLE_RATE
        if duration_s < VAD_MIN_FILE_S:
            return None
        if len(spans) == 1 and spans[0][1] - spans[0][0] > 0.9 * duration_s:
            return None
        return spans

    @staticmethod
    def _load_vad():
        onnx_path = _ensure_vad_onnx()
        if onnx_path is None:
            return None
        try:
            return _SileroVad(onnx_path)
        except Exception:
            logger.warning("Silero VAD failed to load; using plain 30-s windows",
                           exc_info=True)
            return None

    @classmethod
    def _decode_window(cls, model: _NpuWhisperModel, window, language: Optional[str],
                       task_kind, offset_ms: int, label: str) -> List[Segment]:
        """Decode one padded 30-s window; Segments carry `offset_ms`.

        Padded with a noise floor only for translate (the hallucination-prone
        mode); transcribe decodes measure better with zero padding.
        """
        if len(window) == 0:
            return []
        if len(window) < WHISPER_SAMPLE_RATE * WINDOW_SECONDS:
            window = _pad(window, WHISPER_SAMPLE_RATE * WINDOW_SECONDS,
                          noise_floor=task_kind == Task.TRANSLATE)

        features = model.feature_extractor(
            window, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="np"
        )["input_features"].astype("float16")
        cross_attention = model.encode(features)

        prefix, suppress = WhisperNpu._build_prefix(
            model, language, task_kind, cross_attention
        )
        token_ids = model.decode(prefix, cross_attention, suppress)

        # Hallucination-loop guard: a full-length decode never sampled EOT
        # (EOT always breaks before appending), and repetitive text on top
        # of that is the observed loop signature (e.g. forced language on
        # a music intro → "(gentle music)" to the cap). Drop the window;
        # the cap already bounds the damage. Cap-hit with varied text is
        # dense legit speech and stays (trailing flush at the window end).
        if len(token_ids) >= DECODE_LEN and WhisperNpu._is_repetitive_text(model, token_ids):
            logger.warning(
                "NPU decode hit the token cap with repetitive text (%s); dropping it",
                label)
            return []

        return WhisperNpu._segments_from_token_stream(
            model=model,
            token_ids=token_ids,
            offset_ms=offset_ms,
            window_ms=WINDOW_SECONDS * 1000,
        )

    @staticmethod
    def _build_prefix(
        model: _NpuWhisperModel,
        language: Optional[str],
        task_kind: Task,
        cross_attention: dict,
    ):
        """Decoder prefix + argmax suppressions for the language/task request."""
        task_id = model.task_ids[task_kind]
        if language is None:
            if task_kind == Task.TRANSCRIBE:
                # Free-run: the model picks the language itself and keeps
                # timestamps on (verified: auto-detect + ts tokens in stream).
                return [model.sot], ()
            # Auto-translate: detect the language first, then force the task.
            lang_id = model.detect_language(cross_attention)
            return [model.sot, lang_id, task_id], (model.no_timestamps,)

        lang_id = model.lang_to_id.get(language)
        if lang_id is None:
            raise ValueError(f"Unknown language for the NPU engine: {language}")
        return [model.sot, lang_id, task_id], (model.no_timestamps,)

    @staticmethod
    def _word_level_segments(audio, segments: List[Segment], options) -> List[Segment]:
        """Word-level Segments via the vendored CTC forced aligner (plan §1).

        The NPU engine's timed Segments seed the chunked aligner
        (TranscriptAligner, reused from speaker identification) so each word
        lands on the real audio instead of an interpolation. Any aligner
        failure (import, model download, transcript mismatch) degrades to the
        segment-level output with a warning — word timings are an enhancement,
        never a failure of the transcription.
        """
        if not segments:
            return segments
        try:
            words = WhisperNpu._align_words(
                audio, segments, options.language or "en")
        except Exception:
            logger.warning(
                "CTC word alignment failed; keeping segment-level timestamps",
                exc_info=True,
            )
            return segments

        if not words:
            logger.warning("CTC word alignment produced no words; keeping segment-level timestamps")
            return segments
        return [
            Segment(
                start=int(word["start"] * 1000),
                end=int(word["end"] * 1000),
                text=word["text"].strip(),
            )
            for word in words
            if word["text"].strip()
        ]

    @staticmethod
    def _align_words(audio, segments: List[Segment], language: str) -> List[dict]:
        """Run the vendored CTC aligner over the NPU segments; raises on failure.

        The acoustic-model pass is language-gated: English uses the w8a16
        wav2vec2 asset on the HTP when it is available (configured via
        ``BUZZ_WAV2VEC2_ALIGNER_DIR``); everything else runs the multilingual
        MMS-300m model on CPU, exactly as the speaker-identification flow does.
        The Viterbi (forced_align C++) and the transcript handling stay shared.
        """
        import torch
        from ctc_forced_aligner.ctc_forced_aligner import (
            generate_emissions,
            get_alignments,
            get_spans,
            load_alignment_model,
            postprocess_results,
            preprocess_text,
        )
        from whisper_diarization.helpers import langs_to_iso

        from buzz.transcriber.speaker_identifier import TranscriptAligner

        npu_ctc = _NpuCtcEmissions.discover() if language == "en" else None
        if npu_ctc is not None:
            from transformers import AutoTokenizer

            emissions = torch.from_numpy(npu_ctc.emissions(audio))
            stride = npu_ctc.STRIDE_MS
            if emissions.size(0) < 2:
                raise RuntimeError("Audio too short to align on the NPU.")
            tokenizer = AutoTokenizer.from_pretrained(
                "facebook/wav2vec2-conformer-rel-pos-large-960h-ft",
                word_delimiter_token=None,
            )
            preprocess_fn = _preprocess_text_no_stars
        else:
            model, tokenizer = load_alignment_model(device="cpu", dtype=torch.float32)
            emissions, stride = generate_emissions(model, torch.from_numpy(audio))
            preprocess_fn = preprocess_text

        aligner = TranscriptAligner(
            preprocess_fn,
            get_alignments,
            get_spans,
            postprocess_results,
            langs_to_iso,
        )
        transcript = " ".join(segment.text for segment in segments)
        return aligner.align(
            transcript, language, emissions, stride, tokenizer, segments=segments
        )

    @staticmethod
    def _is_repetitive_text(model: _NpuWhisperModel, token_ids: List[int]) -> bool:
        """True when the decoded text is mostly the same few words repeated.

        Repetition ratio over ≥ 12 words keeps short legitimate phrases out.
        """
        text = model.tokenizer.decode(token_ids, skip_special_tokens=True)
        words = text.split()
        if len(words) < 12:
            return False
        return len(set(words)) / len(words) < 0.2

    @staticmethod
    def _segments_from_token_stream(
        model: _NpuWhisperModel,
        token_ids: List[int],
        offset_ms: int,
        window_ms: int,
    ) -> List[Segment]:
        """Build buzz Segments from one window's tokens (incl. forced prefix).

        A timestamp token closes the segment since the previous one (standard
        whisper semantics); times are 20 ms steps added to the window offset.
        Text before the first timestamp belongs to a segment starting at the
        window start; trailing text without a closing timestamp ends at the
        window end; a window with no timestamp tokens at all becomes one
        window-length Segment (the forced-translate fallback).
        """
        segments: List[Segment] = []
        text_ids: List[int] = []
        seg_start_ms = 0

        def flush(end_ms: int):
            text = model.tokenizer.decode(text_ids, skip_special_tokens=True).strip()
            text_ids.clear()
            if not text:
                return
            segments.append(Segment(
                start=offset_ms + seg_start_ms,
                end=offset_ms + end_ms,
                text=text,
            ))

        for token_id in token_ids:
            if token_id == model.sot:
                continue
            if model.ts_first <= token_id <= model.ts_last:
                ts_ms = (token_id - model.ts_first) * 20
                flush(end_ms=ts_ms)
                seg_start_ms = ts_ms
                continue
            if token_id == model.eot:
                break
            text_ids.append(token_id)

        flush(end_ms=window_ms)
        return segments
