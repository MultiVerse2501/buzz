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


def _pad(audio, target_len: int):
    import numpy as np

    return np.pad(audio, (0, target_len - len(audio)))


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
        attention mask from the validated prototype.
        """
        import numpy as np

        if self.zero_k is None:
            self.zero_k = np.zeros(
                (N_HEADS, 1, HEAD_DIM, DECODE_LEN - 1), dtype=np.float16)
            self.zero_v = np.zeros(
                (N_HEADS, 1, DECODE_LEN - 1, HEAD_DIM), dtype=np.float16)

        out_ids = list(prefix_ids)
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
            feed["input_ids"] = np.array([[out_ids[-1]]], dtype=np.int32)
            feed["position_ids"] = pos
            feed["attention_mask"] = attn
            outputs = self.dec_session.run(self.dec_out_names, feed)
            by_name = dict(zip(self.dec_out_names, outputs))

            logits = by_name["logits"].reshape(-1)
            if first_logits is None:
                first_logits = logits
            if step < len(prefix_ids) - 1:
                out_ids.append(prefix_ids[step + 1])  # forced token
            else:
                if suppress:
                    logits = logits.copy()
                    for token_id in suppress:
                        logits[token_id] = -np.inf
                token_id = int(logits.argmax())
                if token_id == self.eot:
                    break
                out_ids.append(token_id)

            for layer in range(N_LAYERS):
                state[f"k_cache_self_{layer}_in"] = by_name[f"k_cache_self_{layer}_out"]
                state[f"v_cache_self_{layer}_in"] = by_name[f"v_cache_self_{layer}_out"]
            pos = pos + 1

        return first_logits, out_ids


class WhisperNpu:
    @staticmethod
    def transcribe(task: FileTranscriptionTask) -> List[Segment]:
        _maybe_pin_affinity()
        _pin_blas_threads()

        model = _NpuWhisperModel(task.model_path)

        from buzz import whisper_audio

        audio = whisper_audio.load_audio(task.file_path)
        if task.transcription_options.word_level_timings:
            logger.info(
                "Word-level timings are not supported by the NPU engine; "
                "producing segment-level timestamps"
            )
        return WhisperNpu.transcribe_audio(model, audio, task.transcription_options)

    @classmethod
    def transcribe_audio(cls, model: _NpuWhisperModel, audio, options) -> List[Segment]:
        """Windowed decode of 16 kHz float32 mono audio with a loaded model.

        Split from transcribe() so callers that already hold samples (the
        recording path in M4, batch verification) reuse one loaded model.
        """
        task_kind = options.task
        language = options.language or None

        total_windows = max(1, -(-len(audio) // (16000 * WINDOW_SECONDS)))
        segments: List[Segment] = []
        for window_index in range(total_windows):
            window = audio[
                window_index * 16000 * WINDOW_SECONDS:
                (window_index + 1) * 16000 * WINDOW_SECONDS
            ]
            if len(window) == 0:
                continue
            if len(window) < 16000 * WINDOW_SECONDS:
                window = _pad(window, 16000 * WINDOW_SECONDS)

            features = model.feature_extractor(
                window, sampling_rate=16000, return_tensors="np"
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
                    "NPU decode hit the token cap with repetitive text "
                    "(window %d); dropping the window", window_index)
                sys.stderr.write(f"{int(100 * (window_index + 1) / total_windows)}%\n")
                sys.stderr.flush()
                continue

            offset_ms = window_index * WINDOW_SECONDS * 1000
            segments.extend(
                WhisperNpu._segments_from_token_stream(
                    model=model,
                    token_ids=token_ids,
                    offset_ms=offset_ms,
                    window_ms=WINDOW_SECONDS * 1000,
                )
            )
            # The worker's stderr is regex-parsed by the parent for progress.
            sys.stderr.write(f"{int(100 * (window_index + 1) / total_windows)}%\n")
            sys.stderr.flush()

        return segments

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
