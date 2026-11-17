"""NPU speaker diarization (Track O/O2) — the ARM64 backend for the speaker
identification worker's diarization stage.

Replaces the NeMo MSDD path on Snapdragon machines with three QNN contexts
running through the app's pinned ORT-QNN runtime (onnxruntime-qnn wheel,
QAIRT 2.50.40):

  1. pyannote segmentation (Qualcomm AI Hub prebuilt, release v0.66.0)
     — speech/slot activity, waveform-in [32,1,80000] -> [32,293,7]
  2. titanet_large speaker embedding (compiled from the same weights the
     NeMo MSDD path uses) — [32,80,150] mel + length -> [32,192]
  3. MSDD decoder, reconstructed from the bare NGC checkpoint (no NeMo)
     — [1,50,5,192] + [1,5,192,2] -> per-frame 2-speaker probabilities

Orchestration (NeMo-free, plan P3-speaker-id-arm64.md §17): activity →
multiscale titanet embeddings (5 scales, batch-32) → base-scale clustering
(natural k, capped, dense-remapped) → per-scale per-cluster profiles →
MSDD per speaker pair → max-over-pairs merge → per-speaker turns.

Assets (compiled contexts + the prebuilt segmentation context) live in
`<model_root>/speaker-id-npu/` (or `BUZZ_SPEAKER_ID_NPU_DIR`):

  segmentation.bin                  pyannote context (raw QNN blob)
  segmentation.metadata.json        its I/O contract (wrapper building)
  titanet.onnx / titanet.bin        EPContext pair (batch-32)
  msdd.onnx / msdd.bin              EPContext pair

Download-at-first-use rides the P3 GitHub-Releases hosting decision; until
then the directory is provisioned out-of-band. All NPU sessions are opened
lazily on first use and cached for the process lifetime. Set
BUZZ_DIARIZER_NPU=0 to force the legacy (NeMo) diarizer path.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

SR = 16000
SEG_FRAMES = 80000          # 5 s window (samples)
SEG_STEP_S = 1.0            # window step (s)
SEG_BATCH = 32
OUT_FRAMES = 293
FRAME_SPAN = SEG_FRAMES / OUT_FRAMES      # ~17.06 ms per output frame
MAX_SLOTS = 3
POWERSET_TABLE = ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
FEAT_T = 150                # titanet mel frames (1.5 s pad-to-longest)
TITANET_BATCH = 32
N_MELS = 80
BASE_WIN = 0.5              # MSDD finest scale (s)
BASE_SHIFT = 0.25           # base-frame duration (s)
SEQ_T = 50                  # MSDD window (base frames)
SCALES = (1.5, 1.25, 1.0, 0.75, 0.5)
BIN_THRESHOLD = 0.5
SPEECH_THRESHOLD = 0.3
MERGE_GAP_S = 0.3
MIN_TURN_S = 0.2
CLUSTER_THRESHOLD = 0.45
MAX_SPEAKERS = 8
ASSET_SUBDIR = "speaker-id-npu"

_ASSETS_CHECKED = False
_SESSIONS = None            # {name: InferenceSession} (cached)


# --- assets ---------------------------------------------------------------------

def _model_root() -> Path:
    from buzz.model_loader import model_root_dir
    return Path(model_root_dir)


def assets_dir() -> Path:
    """Resolved asset directory (env override wins)."""
    override = os.getenv("BUZZ_SPEAKER_ID_NPU_DIR")
    if override:
        return Path(override)
    return _model_root() / ASSET_SUBDIR


def is_available() -> bool:
    """All three model artifacts present (and not disabled by env)."""
    if os.getenv("BUZZ_DIARIZER_NPU", "1") == "0":
        return False
    d = assets_dir()
    needed = ["segmentation.bin", "segmentation.metadata.json",
              "titanet.onnx", "titanet.bin", "msdd.onnx", "msdd.bin"]
    return all((d / f).is_file() for f in needed)


def _sessions() -> dict:
    """Open the three QNN sessions lazily (cached for the process)."""
    global _SESSIONS, _ASSETS_CHECKED
    if _SESSIONS is not None:
        return _SESSIONS
    import onnxruntime as ort

    from buzz.npu_capability import get_qnn_ep_devices

    devices = get_qnn_ep_devices()
    if not devices:
        raise RuntimeError(
            "NPU speaker diarization: no QNN devices available.")
    d = assets_dir()
    if not is_available():
        raise RuntimeError(
            "NPU speaker diarization assets are missing from "
            f"{d} (expected segmentation.bin, titanet.onnx/.bin, "
            "msdd.onnx/.bin).")

    def sess(path: Path) -> "ort.InferenceSession":
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.add_provider_for_devices(devices, {})
        return ort.InferenceSession(str(path), sess_options=opts)

    _SESSIONS = {
        "segmentation": sess(_segmentation_wrapper(d)),
        "titanet": sess(d / "titanet.onnx"),
        "msdd": sess(d / "msdd.onnx"),
    }
    _ASSETS_CHECKED = True
    return _SESSIONS


def _segmentation_wrapper(d: Path) -> Path:
    """The pyannote context ships as a raw QNN blob; build (once) the thin
    EPContext wrapper ONNX that ORT-QNN loads, from the metadata contract."""
    wrapper = d / "segmentation_wrapper.onnx"
    if wrapper.is_file():
        return wrapper
    import onnx
    from onnx import TensorProto, helper

    meta = json.loads((d / "segmentation.metadata.json").read_text(encoding="utf-8"))
    spec = meta["model_files"]["segmentation.bin"]
    et_map = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16,
              "int32": TensorProto.INT32}

    def vi(name, entry):
        return helper.make_tensor_value_info(
            name, et_map[entry["dtype"]], entry["shape"])

    node = helper.make_node(
        "EPContext", list(spec["inputs"]), list(spec["outputs"]), "QNNContext",
        domain="com.microsoft",
        ep_cache_context="segmentation.bin", embed_mode=0, source="QNN",
    )
    m = helper.make_model(
        helper.make_graph([node], "diarization",
                          [vi(n, spec["inputs"][n]) for n in spec["inputs"]],
                          [vi(n, spec["outputs"][n]) for n in spec["outputs"]]),
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="Qualcomm AI Hub Workbench",
    )
    m.ir_version = 13
    onnx.save(m, str(wrapper))
    logging.debug("Speaker diarization NPU: built segmentation wrapper %s", wrapper)
    return wrapper


# --- frontends --------------------------------------------------------------------

def hz2mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)


def _mel_banks_kaldi(n_mels: int, n_fft: int, sr: int,
                     low_f: float, high_f: float) -> np.ndarray:
    """Kaldi MelBanks (vtln off) — verbatim port of the harness bank
    (diarization_common._mel_banks_kaldi) the titanet features were validated
    with."""
    if high_f <= 0.0:
        high_f = sr / 2.0 - low_f if low_f > 0 else sr / 2.0
    mel_low, mel_high = hz2mel(np.array([low_f, high_f]))
    mels = np.linspace(mel_low, mel_high, n_mels + 2)
    freqs = 700.0 * (np.exp(mels / 1127.0) - 1.0)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for b in range(n_mels):
        l, c, r = bins[b], bins[b + 1], bins[b + 2]
        up = (freqs[b + 1] - freqs[b]) / max(c - l, 1)
        dn = (freqs[b + 1] - freqs[b]) / max(r - c, 1)
        if c > l:
            fb[b, l:c] = up * (np.arange(l, c) - l)
        if r > c:
            fb[b, c:r] = dn * (r - np.arange(c, r))
        fb[b, c] = 1.0
    return fb


def _kaldi_fbank(wave1d: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """Kaldi-style 80-mel log-fbank (25/10 ms, povey, snip_edges, low_freq 20)
    in numpy — verbatim port of the harness dc.kaldi_fbank the titanet features
    were validated with (the torchaudio.compliance.kaldi.fbank call the weights
    were exported for)."""
    x = np.asarray(wave1d, dtype=np.float64)
    flen, fshift, nfft = 400, 160, 512
    n = 1 + max(0, (len(x) - flen) // fshift)
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, flen), strides=(x.strides[0] * fshift, x.strides[0])).copy()
    frames -= frames.mean(axis=1, keepdims=True)
    pre = np.empty_like(frames)
    pre[:, 0] = frames[:, 0] * (1.0 - 0.97)
    pre[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    i = np.arange(flen, dtype=np.float64)
    win = (0.54 - 0.46 * np.cos(2 * np.pi * i / (flen - 1))) ** 0.85
    frames = pre * win[None, :]
    spec = np.abs(np.fft.rfft(frames, n=nfft, axis=1)) ** 2 / nfft
    fb = _mel_banks_kaldi(n_mels, nfft, SR, 20.0, 0.0)
    return np.log(np.maximum(spec @ fb.T, 1e-10)).astype(np.float32)


def _titanet_features(wav1d: np.ndarray) -> tuple[np.ndarray, int]:
    """(80,150) features + valid-frame count (nemo_norm + pad-to-longest)."""
    fb = _kaldi_fbank(wav1d)
    n_valid = fb.shape[0]
    fb = (fb - fb.mean(axis=0, keepdims=True)) / (fb.std(axis=0, keepdims=True) + 1e-5)
    if fb.shape[0] < FEAT_T:
        fb = np.concatenate([fb, np.repeat(fb[-1:], FEAT_T - fb.shape[0], axis=0)], axis=0)
    fb = fb[:FEAT_T]                      # defensive crop (windows ≤ 1.5 s)
    return fb.T.astype(np.float32), min(n_valid, FEAT_T)


# --- windows / powerset / fusion ---------------------------------------------------

def _sliding_chunks(audio: np.ndarray) -> list[np.ndarray]:
    out, start = [], 0
    while start + SEG_FRAMES <= len(audio):
        out.append(audio[start:start + SEG_FRAMES])
        start += int(SEG_STEP_S * SR)
    if start < len(audio):
        tail = audio[start:]
        out.append(np.concatenate([tail, np.zeros(SEG_FRAMES - len(tail), np.float32)]))
    return out


def _powerset_to_slots(probs: np.ndarray) -> np.ndarray:
    out = np.zeros(probs.shape[:-1] + (MAX_SLOTS,), dtype=np.float32)
    for j, subset in enumerate(POWERSET_TABLE):
        for slot in subset:
            out[..., slot] = np.maximum(out[..., slot], probs[..., j])
    return out


def _window_frame_offset(w: int) -> int:
    return int(round(w * SEG_STEP_S * SR / FRAME_SPAN))


def _fuse_windows(win_slots: np.ndarray, n_windows: int) -> np.ndarray:
    total = _window_frame_offset(n_windows - 1) + OUT_FRAMES + 1
    acc = np.zeros((total, MAX_SLOTS), np.float64)
    cnt = np.zeros((total, 1), np.float64)
    for w, p in enumerate(win_slots):
        i0 = _window_frame_offset(w)
        acc[i0:i0 + OUT_FRAMES] += p
        cnt[i0:i0 + OUT_FRAMES] += 1.0
    return (acc / np.maximum(cnt, 1.0)).astype(np.float32)


def _cluster_embeddings(vectors: np.ndarray, threshold: float) -> np.ndarray:
    """Prototype clustering (fixed first-embedding prototypes + 2 refinement
    passes over true means) on cosine distance — numpy-only, deterministic."""
    vectors = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    protos: list[np.ndarray] = []
    labels = np.zeros(len(vectors), dtype=int)
    for i, v in enumerate(vectors):
        if not protos:
            protos.append(v.astype(np.float64))
            continue
        P = np.stack(protos)
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        dist = 1.0 - P @ v.astype(np.float64)
        best = int(np.argmin(dist))
        if dist[best] <= threshold:
            labels[i] = best
        else:
            protos.append(v.astype(np.float64))
            labels[i] = len(protos) - 1
    k = len(protos)
    if k == 0:
        return labels
    means = np.zeros((k, vectors.shape[1]), np.float64)
    counts = np.zeros(k, np.int64)
    for i, lab in enumerate(labels):
        means[lab] += vectors[i]
        counts[lab] += 1
    means /= np.maximum(counts, 1)[:, None]
    for _ in range(2):
        M = means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-9)
        labels = np.argmax(vectors.astype(np.float64) @ M.T, axis=1)
        for c in range(k):
            m = labels == c
            if m.any():
                means[c] = vectors[m].mean(axis=0)
    return labels


# --- turns ---------------------------------------------------------------------

def _turns_from(activity: np.ndarray, frame_s: float) -> list:
    """Binary frame activity → [(start_s, end_s)] with gap-merge + min turn."""
    on = activity > 0.5
    spans, t0 = [], None
    for i, v in enumerate(on):
        if v and t0 is None:
            t0 = i
        elif not v and t0 is not None:
            spans.append([t0 * frame_s, i * frame_s])
            t0 = None
    if t0 is not None:
        spans.append([t0 * frame_s, len(on) * frame_s])
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < MERGE_GAP_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= MIN_TURN_S]


# --- diarization entry point ------------------------------------------------------

def diarize_speaker_turns(audio: np.ndarray,
                          num_speakers: int | None = None) -> list:
    """Full pipeline → speaker turns as (start_ms, end_ms, spk_index) — the
    contract the speaker-identification worker's mapping stage consumes.

    num_speakers: exact count from the UI combo (clamps the detected k), or
    None for automatic detection (k capped at MAX_SPEAKERS)."""
    sessions = _sessions()
    audio = np.asarray(audio, dtype=np.float32)

    # 1. speech + slot activity (pyannote segmentation, batch-32)
    chunks = _sliding_chunks(audio)
    n_win = len(chunks)
    win_slots = np.empty((n_win, OUT_FRAMES, MAX_SLOTS), np.float32)
    for i in range(0, n_win, SEG_BATCH):
        b = np.stack(chunks[i:i + SEG_BATCH])
        if len(b) < SEG_BATCH:
            b = np.pad(b, ((0, SEG_BATCH - len(b)), (0, 0)))
        out = sessions["segmentation"].run(None, {"waveforms": b[:, None, :]})[0]
        for w in range(len(chunks[i:i + SEG_BATCH])):
            e = np.exp(out[w] - out[w].max(-1, keepdims=True))
            e /= e.sum(-1, keepdims=True)
            win_slots[i + w] = _powerset_to_slots(e[None])[0]
    fused = _fuse_windows(win_slots, n_win)
    speech = fused.max(axis=1) > SPEECH_THRESHOLD
    regions = _turns_from(speech.astype(np.float32), FRAME_SPAN / SR)

    # 2. multiscale titanet embeddings (batch-32, speech-overlapping windows)
    dur = len(audio) / SR
    def speech_coverage(t0: float, s: float) -> float:
        total = sum(max(0.0, min(t0 + s, re) - max(t0, rs)) for rs, re in regions)
        return total / s

    feats_list, keys, wins_by_scale = [], [], {}
    for si, s in enumerate(SCALES):
        shift = s / 2
        wins = []
        t = 0.0
        while t + s <= dur + 1e-9:
            # only windows mostly inside speech: boundary/gap-dominated
            # windows chain the clustering (single-linkage effect)
            if speech_coverage(t, s) >= 0.6:
                wins.append(t)
            t += shift
        wins_by_scale[s] = wins
        for wi, t0 in enumerate(wins):
            wav = audio[int(t0 * SR): int((t0 + s) * SR)]
            if len(wav) < int(s * SR):
                wav = np.pad(wav, (0, int(s * SR) - len(wav)))
            f, nv = _titanet_features(wav)
            feats_list.append((f, nv))
            keys.append((si, wi))
    if not feats_list:
        return []
    valid_list = [nv for (_, nv) in feats_list]
    feats_only = [f for (f, _) in feats_list]
    embeddings = np.empty((len(feats_only), 192), np.float32)
    for i in range(0, len(feats_only), TITANET_BATCH):
        b = list(feats_only[i:i + TITANET_BATCH])
        keep = len(b)
        while len(b) < TITANET_BATCH:
            b.append(np.zeros_like(b[0]))
        lengths = np.array(valid_list[i:i + TITANET_BATCH]
                           + [148] * (TITANET_BATCH - keep), dtype=np.int32)
        embeddings[i:i + keep] = sessions["titanet"].run(
            ["output_1"], {"audio_signal": np.stack(b), "length": lengths})[0][:keep]
    ms_embs = {s: {} for s in SCALES}
    for (si, wi), e in zip(keys, embeddings):
        ms_embs[SCALES[si]][wi] = e

    # 3. base-scale clustering → k speakers (natural, capped, dense-remapped)
    wins_base = wins_by_scale[BASE_WIN]
    base_emb = np.stack([ms_embs[BASE_WIN][wi] for wi in range(len(wins_base))
                         if wi in ms_embs[BASE_WIN]])
    base_idx = [wi for wi in range(len(wins_base)) if wi in ms_embs[BASE_WIN]]
    if len(base_emb) == 0:
        return []
    labels = _cluster_embeddings(base_emb, CLUSTER_THRESHOLD)
    clusters = sorted(set(labels.tolist()),
                      key=lambda c: -int((labels == c).sum()))[:MAX_SPEAKERS]
    cent = {c: base_emb[labels == c].mean(axis=0) for c in clusters}
    for i, l in enumerate(labels):
        if l not in clusters:
            labels[i] = min(clusters, key=lambda c: float(
                np.linalg.norm(base_emb[i] - cent[c])))
    # dense remap PRESERVING the size order (dense 0 = largest cluster)
    remap = {c: i for i, c in enumerate(clusters)}
    labels = np.array([remap[int(l)] for l in labels], dtype=int)
    k = len(clusters)
    if num_speakers and 1 <= num_speakers <= k:
        # exact count from the UI combo: keep the num_speakers largest
        # clusters (dense ids 0..num_speakers-1), reassign the rest to the
        # nearest kept centroid
        keep = list(range(num_speakers))
        cent_k = {c: base_emb[labels == c].mean(axis=0) for c in keep}
        for i, l in enumerate(labels):
            if l not in keep:
                labels[i] = min(keep, key=lambda c: float(
                    np.linalg.norm(base_emb[i] - cent_k[c])))
        k = num_speakers

    # 4. per-scale per-cluster profiles (decisive majority ≥60%)
    frame_labels = {}
    for emb_i, wi in zip(labels, base_idx):
        f0 = _window_frame_offset(wi)
        for f in range(f0, f0 + OUT_FRAMES):
            frame_labels[f] = emb_i
    base_cent = {c: base_emb[labels == c].mean(axis=0) for c in range(k)
                 if (labels == c).any()}
    prof = np.zeros((len(SCALES), k, 192), np.float32)
    for si, s in enumerate(SCALES):
        wins = wins_by_scale[s]
        for c_idx in range(k):
            rows = []
            for wi, e in ms_embs[s].items():
                f0 = int(round(wins[wi] / BASE_SHIFT))
                f1 = int(round((wins[wi] + s) / BASE_SHIFT))
                votes = [frame_labels[f] for f in range(f0, f1) if f in frame_labels]
                if votes:
                    top = int(np.argmax(np.bincount(votes)))
                    if top == c_idx and max(np.bincount(votes)) / len(votes) >= 0.6:
                        rows.append(e)
            if rows:
                prof[si, c_idx] = np.mean(rows, axis=0)
            elif c_idx in base_cent:
                # fallback: the cluster's base-scale centroid (some small
                # clusters have no decisive-majority windows at coarse scales)
                prof[si, c_idx] = base_cent[c_idx]

    # 5. MSDD decoder per speaker pair; speaker-s prob = max over pairs
    n_base = int(np.floor((dur - BASE_WIN) / BASE_SHIFT)) + 1
    win_of_frame = {}
    for si, s in enumerate(SCALES):
        n_w = len(wins_by_scale[s])
        arr = np.zeros(max(n_base, 1), np.int64)
        for f in range(max(n_base, 1)):
            center = f * BASE_SHIFT + BASE_WIN / 2
            arr[f] = int(np.clip(round((center - s / 2) / (s / 2)), 0, max(n_w - 1, 0)))
        win_of_frame[si] = arr
    speaker_probs = np.zeros((max(n_base, 1), k), np.float32)
    for pi in range(k):
        for pj in range(pi + 1, k):
            ms_avg = np.zeros((1, len(SCALES), 192, 2), np.float32)
            ms_avg[0, :, :, 0] = prof[:, pi, :]
            ms_avg[0, :, :, 1] = prof[:, pj, :]
            for w0 in range(0, n_base, SEQ_T):
                seq = np.zeros((SEQ_T, len(SCALES), 192), np.float32)
                any_valid = False
                for f_local in range(SEQ_T):
                    f = w0 + f_local
                    if f >= n_base:
                        break
                    for si, s in enumerate(SCALES):
                        wi = int(win_of_frame[si][f])
                        if wi in ms_embs[s]:
                            seq[f_local, si] = ms_embs[s][wi]
                            any_valid = True
                if not any_valid:
                    continue
                preds = sessions["msdd"].run(
                    ["output_0"], {"ms_emb_seq": seq[None],
                                   "ms_avg_embs": ms_avg})[0][0]
                lo, hi = w0, min(w0 + SEQ_T, n_base)
                speaker_probs[lo:hi, pi] = np.maximum(
                    speaker_probs[lo:hi, pi], preds[:hi - lo, 0])
                speaker_probs[lo:hi, pj] = np.maximum(
                    speaker_probs[lo:hi, pj], preds[:hi - lo, 1])

    if os.getenv("BUZZ_DIARIZER_NPU_DEBUG"):
        _u = base_emb / np.maximum(np.linalg.norm(base_emb, axis=1, keepdims=True), 1e-9)
        _iu = np.triu_indices(len(_u), 1)
        logging.debug("SDN debug: base_emb n=%d norm_mean=%.3f pairwise_cos_mean=%.3f",
                      len(base_emb), float(np.mean(np.linalg.norm(base_emb, axis=1))),
                      float((_u @ _u.T)[_iu].mean()) if len(_u) > 1 else -1)
        for _th in (0.45, 0.3, 0.2):
            _labs = _cluster_embeddings(base_emb.copy(), _th)
            logging.debug("SDN debug: sweep th=%.2f -> k=%d sizes=%s", _th,
                          len(set(_labs.tolist())),
                          sorted([int((_labs == c).sum()) for c in set(_labs)], reverse=True)[:6])
    if os.getenv("BUZZ_DIARIZER_NPU_DEBUG"):
        logging.debug("SDN debug: regions=%d k=%d prof_norms=%s probs_mean=%.3f "
                      "probs_p90=%.3f probs_max=%.3f",
                      len(regions), k,
                      [round(float(np.linalg.norm(prof[0, c])), 1) for c in range(k)],
                      float(speaker_probs.mean()),
                      float(np.percentile(speaker_probs, 90)),
                      float(speaker_probs.max()))
    # 6. turns per speaker, masked to detected speech (base grid = 0.25 s)
    speech_frames = np.zeros(max(n_base, 1), np.float32)
    for r0, r1 in regions:
        f0 = int(np.floor(r0 / BASE_SHIFT))
        f1 = int(np.ceil(r1 / BASE_SHIFT))
        speech_frames[f0:min(f1, n_base)] = 1.0
    speaker_ts = []
    for c in range(k):
        act = ((speaker_probs[:, c] > BIN_THRESHOLD)
               & (speech_frames > 0.5)).astype(np.float32)
        for a, b in _turns_from(act, BASE_SHIFT):
            speaker_ts.append((int(round(a * 1000)), int(round(b * 1000)), int(c)))
    speaker_ts.sort()
    logging.debug("Speaker diarization NPU: %d turns across %d speakers",
                  len(speaker_ts), k)
    return speaker_ts
