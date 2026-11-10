"""Whisper-internal DTW token timestamps for the NPU engine (path C X2b).

Pure-numpy port of the whisper token-timestamp chain used by openai/whisper's
``whisper/timing.py`` and whisper.cpp's DTW mode (PR #1485, faithful port of
timing.py). Input: the softmax cross-attention weights of the alignment
heads, captured per decode step by the attention-exporting decoder asset
(`whisper_npu._NpuWhisperModel` capture rows). Output: one timestamp per text
token, grouped into word Segments by the caller.

Chain (whisper.cpp src/whisper.cpp:8915-8964 == timing.py find_alignment):
normalize per audio position across the token axis (biased variance, eps
1e-9) -> median filter width 7 over the audio axis (reflect-padded, sorted)
-> mean over heads -> negate -> trim to real audio frames -> DTW + backtrace
-> place per-token frames (20 ms each).

Provenance / licensing: openai/whisper timing.py (MIT); whisper.cpp PR #1485
(MIT). The DP is computed row-wise via a prefix-scan identity
(``d[i] = C[i] + min_{k<=i}(base[k] - C[k])``, C = sequential prefix sum via
``np.add.accumulate``) for speed; ``_dtw_naive`` is the cell-exact reference
implementation and tests assert path equivalence. Ties in the DP follow
whisper.cpp's strict-comparison order (diag < up < left, ties to left).

Boundary contract: pure functions, numpy is the only import, no engine/QNN/
tokenizer/IO dependencies. The engine (whisper_npu) imports this module
lazily inside the word-timestamp dispatch.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Languages without spaces between words (whisper_cpp.py semantics): timings
# are grouped per character instead of per space-delimited word.
NON_SPACE_LANGUAGES = frozenset({"zh", "ja", "th", "lo", "km", "my"})

MEDFILT_WIDTH = 7
NORM_EPS = 1e-9
# whisper audio tokens are 20 ms apart (1500 tokens per 30 s window).
MS_PER_FRAME = 20


def select_text_token_indices(
    fed_ids: Sequence[int],
    eot: int,
    ts_first: int | None = None,
    ts_last: int | None = None,
) -> List[int]:
    """Indices of capture rows that belong to TEXT tokens (DTW matrix rows).

    whisper's DTW consumes text tokens only: whisper.cpp re-decodes
    ``t.id < eot`` (src/whisper.cpp:8887-8896) and openai teacher-forces the
    text sequence. Text tokens have ``id < eot``; SOT/language/task/timestamp
    tokens sit above ``eot`` (the ts band check is kept as an explicit guard).
    """
    return [
        i for i, tok in enumerate(fed_ids)
        if tok < eot and not (ts_first is not None and ts_first <= tok <= ts_last)
    ]


def _median_filter_audio(x: np.ndarray, width: int = MEDFILT_WIDTH) -> np.ndarray:
    """Median filter over the AUDIO axis for every (token, head) column.

    whisper.cpp src/whisper.cpp:8786-8813: reflect padding
    (``idx < 0 -> -idx; idx >= n -> 2*(n-1) - idx``), median = sorted window's
    middle element, odd width. Equivalent to np.pad(mode="reflect") +
    sliding-window sorted median over axis 1 of ``[T, A, H]``.
    """
    assert width % 2 == 1, "filter width must be odd"
    if x.shape[1] <= 1:
        return x
    pad = width // 2
    if x.shape[1] <= pad:  # shorter than half the window: whisper.cpp asserts
        # against width < ne[2]; degrade to no-op instead of crashing.
        return x
    padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=1)
    return np.median(windows, axis=-1).astype(x.dtype)


def _dtw_naive(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-exact whisper.cpp DTW (:8698-8762) — reference implementation.

    cost [N, M]; dp [N+1, M+1] with INF borders and dp[0,0]=0;
    dp[i,j] = cost[i-1,j-1] + min(diag, up, left) with whisper.cpp's
    strict-comparison tie order (ties -> left); border traces overridden
    (trace[:,0]=1 up, trace[0,:]=2 left); backtrace from (N, M) to (0, 0),
    returned in forward order as (text_indices, time_indices).
    """
    n, m = cost.shape
    inf = np.float32(np.inf)
    dp = np.full((n + 1, m + 1), inf, dtype=np.float32)
    trace = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c0 = dp[i - 1, j - 1]
            c1 = dp[i - 1, j]
            c2 = dp[i, j - 1]
            if c0 < c1 and c0 < c2:
                best, t = c0, 0
            elif c1 < c0 and c1 < c2:
                best, t = c1, 1
            else:
                best, t = c2, 2
            dp[i, j] = np.float32(cost[i - 1, j - 1]) + best
            trace[i, j] = t
    trace[:, 0] = 1
    trace[0, :] = 2
    return _backtrace(dp, trace)


def _backtrace(dp: np.ndarray, trace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """whisper.cpp backtrace (:8736-8756): walk (N, M) -> (0, 0), forward out."""
    n, m = dp.shape[0] - 1, dp.shape[1] - 1
    text_idx: List[int] = []
    time_idx: List[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        text_idx.append(i - 1)
        time_idx.append(j - 1)
        t = trace[i, j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i = i - 1
        else:
            j = j - 1
    return (
        np.asarray(text_idx[::-1], dtype=np.int64),
        np.asarray(time_idx[::-1], dtype=np.int64),
    )


def _dtw(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Production DTW: whisper.cpp DP semantics, row-wise prefix-scan.

    Identical recurrence and border handling to ``_dtw_naive``; each row is
    unrolled through the identity
    ``d[i] = C[i] + min_{k<=i}(base[k] - C[k])`` (C = sequential prefix sum,
    ``np.add.accumulate``) so the O(N*M) DP becomes O(N) vectorized rows.
    Accumulation order differs from the cell-sequential naive form (~1e-5
    relative in float32) — path equivalence on real matrices is asserted in
    tests. Trace is reconstructed from the stored rows with whisper.cpp's
    tie order.
    """
    n, m = cost.shape
    inf = np.float32(np.inf)
    dp = np.full((n + 1, m + 1), inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        prev = dp[i - 1]
        a = cost[i - 1]  # local cost added along this row
        # base[j] (j=1..M) = min(dp[i-1, j-1], dp[i-1, j]); dp[i, 0] stays INF
        base = np.minimum(prev[:-1], prev[1:])
        cum_incl = np.add.accumulate(a)  # C[idx] = sum(a[0..idx])
        # candidate k contributes base[k] + sum(a[k+1..idx]) = base[k] + C[idx] - C[k-1]
        cum_excl = np.concatenate(
            ([np.float32(0.0)], cum_incl[:-1])
        ).astype(np.float32)  # C[idx-1], empty sum for idx=0
        g = base - cum_excl
        dp[i, 1:] = cum_incl + np.minimum.accumulate(g)
    trace = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        c0 = dp[i - 1, :-1]  # diag
        c1 = dp[i - 1, 1:]  # up
        c2 = dp[i, :-1]  # left (current row's previous cell; INF at j=0)
        trace[i, 1:] = np.where(
            (c0 < c1) & (c0 < c2), 0, np.where((c1 < c0) & (c1 < c2), 1, 2)
        )
    trace[:, 0] = 1
    trace[0, :] = 2
    return _backtrace(dp, trace)


def _place_timestamps(
    path_text: np.ndarray, path_time: np.ndarray, n_tokens: int
) -> np.ndarray:
    """whisper.cpp placement (:8945-8968) over text-only rows.

    Walking the forward path, every time the path advances INTO a new token
    row (``v != last_v``), that frame index is assigned to the next token.
    whisper.cpp leaves the final token unset (its CLI never consumes it); we
    fill unplaced tokens with the path's last time index because our word
    grouping consumes every token time.
    """
    frames = np.full(n_tokens, -1, dtype=np.int64)
    last_v = 0
    tok = 0
    for col in range(path_text.shape[0]):
        v = int(path_text[col])
        if v != last_v:
            if tok < n_tokens:
                frames[tok] = int(path_time[col])
            tok += 1
            last_v = v
    if tok < n_tokens:
        frames[tok:] = int(path_time[-1]) if path_time.size else 0
    return frames


def dtw_token_frames(
    attn: np.ndarray, real_frames: int | None = None
) -> np.ndarray:
    """Full chain: attention weights -> per-text-token frame index.

    ``attn`` rows are the captured alignment weights of TEXT tokens in stream
    order, in the CAPTURE-NATIVE layout [T, heads, frames] (each capture row
    is [8, 1500]) — or an already head-collapsed [T, frames]. Internally the
    chain works on [T, frames, heads] (whisper.cpp's layout: normalize over
    tokens per (frame, head), median filter over frames, mean over heads).
    ``real_frames`` is the window's real audio-token count (20 ms units) —
    attention beyond it attends to zero-padding and is trimmed before the
    chain (whisper.cpp :7730 n_frames / timing.py num_frames//2).
    """
    if attn.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    # float32 throughout (whisper.cpp computes in f32; fp16 capture rows NaN
    # out in the normalization/median chain)
    attn = np.asarray(attn, dtype=np.float32)
    if attn.ndim == 3 and attn.shape[1] < attn.shape[2]:
        attn = attn.transpose(0, 2, 1)  # [T, heads, frames] -> [T, frames, heads]
    elif attn.ndim == 2:
        attn = attn[:, :, None]
    a_tokens = attn.shape[1]
    if real_frames is not None and real_frames < a_tokens:
        attn = attn[:, :real_frames, :]
    # normalize per audio position across tokens (biased variance + eps)
    mean = attn.mean(axis=0, keepdims=True)
    var = attn.var(axis=0, keepdims=True)  # ddof=0 == timing.py unbiased=False
    w = (attn - mean) / np.sqrt(var + NORM_EPS)
    w = _median_filter_audio(w)
    w = w.mean(axis=2) if w.shape[2] > 1 else w[:, :, 0]
    cost = (-w).astype(np.float32)
    path_text, path_time = _dtw(np.ascontiguousarray(cost))
    return _place_timestamps(path_text, path_time, attn.shape[0])


def token_times_ms(
    attn: np.ndarray,
    real_seconds: float,
    offset_ms: int = 0,
) -> np.ndarray:
    """Per-text-token timestamps in ms, window-offset applied.

    ``attn`` is capture-native [T, heads, frames] (see dtw_token_frames).
    """
    real_frames = max(0, int(round(real_seconds * 1000 / MS_PER_FRAME)))
    frames = dtw_token_frames(attn, real_frames)
    return (frames * MS_PER_FRAME + offset_ms).astype(np.int64)


def group_word_segments(
    texts: Sequence[bytes],
    times_ms: Sequence[int],
    end_ms: int,
    non_space_language: bool = False,
) -> List[Tuple[int, int, str]]:
    """Group per-token times into (start_ms, end_ms, text) words.

    Mirrors whisper_cpp.py ``_parse_word_level_timings`` semantics: spaced
    languages split on a leading space (and the ``", "`` token case);
    non-space languages (zh/ja/th/lo/km/my) group per complete UTF-8
    character. ``texts`` are per-token UTF-8 byte strings (the engine maps
    BPE pieces through the tokenizer's byte_decoder, mirroring whisper_cpp.py
    's latin-1 round-trip, so byte-split characters across tokens reassemble).

    Time convention differs from whisper_cpp.py by necessity: DTW gives ONE
    timestamp per token (its start), not a from/to pair — so every unit ends
    at the NEXT unit's start (openai/whisper's word-boundary convention) and
    the last unit ends at ``end_ms`` (the window/segment end).
    """
    # pass 1: split into units (start_ms, text) with whisper_cpp's rules
    units: List[Tuple[int, str]] = []
    if non_space_language:
        buffer = bytearray()
        char_start = 0
        for piece, t in zip(texts, times_ms):
            if not buffer:
                char_start = t
            buffer += piece
            pos = 0
            consumed = 0
            while pos < len(buffer):
                for length in range(1, min(4, len(buffer) - pos) + 1):
                    try:
                        ch = buffer[pos:pos + length].decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    if ch.strip():
                        units.append((char_start, ch))
                    pos += length
                    consumed = pos
                    break
                else:
                    break
            if consumed:
                del buffer[:consumed]
                char_start = t
        # trailing incomplete UTF-8 bytes are dropped (whisper_cpp.py keeps
        # them pending until the next token; end-of-stream drops them too)
    else:
        word: List[bytes] = []
        word_start = 0
        for piece, t in zip(texts, times_ms):
            if piece.startswith(b" ") and word:
                text = b"".join(word).decode("utf-8", errors="replace").strip()
                if text:
                    units.append((word_start, text))
                word = [piece]
                word_start = t
            elif piece.startswith(b", "):
                if word:
                    word.append(b",")
                    text = b"".join(word).decode("utf-8", errors="replace").strip()
                    if text:
                        units.append((word_start, text))
                word = [piece.lstrip(b",")]
                word_start = t
            else:
                if not word:
                    word_start = t
                word.append(piece)
        if word:
            text = b"".join(word).decode("utf-8", errors="replace").strip()
            if text:
                units.append((word_start, text))

    # pass 2: each unit ends at the next unit's start; the last at end_ms
    return [
        (units[i][0], units[i + 1][0] if i + 1 < len(units) else end_ms, units[i][1])
        for i in range(len(units))
    ]
