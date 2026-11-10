"""Path C X2b unit tests — npu_dtw pure functions (V1 gate).

Contract tests on synthetic inputs (no engine/QNN): text-row selection,
naive-vs-scan DTW equivalence (whisper.cpp DP semantics), monotonic token
times, trim behavior, placement fallback, word grouping (spaced + ", " +
non-space CJK), and degenerate inputs.
"""
import numpy as np
import pytest

from buzz.transcriber import npu_dtw


def test_select_text_token_indices():
    fed = [50258, 50259, 50359, 50364, 634, 19737, 50369, 400, 50257]
    # eot=50257: SOT/lang/task/ts tokens are >= eot; text tokens < eot
    idx = npu_dtw.select_text_token_indices(fed, eot=50257, ts_first=50364, ts_last=50413)
    assert idx == [4, 5, 7]
    assert [fed[i] for i in idx] == [634, 19737, 400]


def test_dtw_naive_vs_scan_random():
    rng = np.random.default_rng(7)
    for n, m in [(5, 12), (20, 40), (37, 9), (1, 6), (12, 1)]:
        cost = rng.uniform(0, 5, size=(n, m)).astype(np.float32)
        t_naive, s_naive = npu_dtw._dtw_naive(cost)
        t_scan, s_scan = npu_dtw._dtw(cost)
        assert t_naive.tolist() == t_scan.tolist(), (n, m)
        assert s_naive.tolist() == s_scan.tolist(), (n, m)
        # monotonic non-decreasing in both axes, spans the full grid
        assert t_scan[0] == 0 and t_scan[-1] == n - 1
        assert s_scan[0] == 0 and s_scan[-1] == m - 1
        assert (np.diff(t_scan) >= 0).all() and (np.diff(s_scan) >= 0).all()


def test_dtw_path_is_monotonic_and_complete():
    rng = np.random.default_rng(3)
    cost = rng.uniform(0, 1, size=(50, 300)).astype(np.float32)
    t_idx, s_idx = npu_dtw._dtw(cost)
    assert t_idx.tolist() == sorted(t_idx.tolist())
    assert len(set(t_idx.tolist())) == 50  # every token row visited


def test_token_frames_monotonic_within_real_audio():
    rng = np.random.default_rng(11)
    attn = rng.uniform(0, 1, size=(30, 8, 1500)).astype(np.float32)
    # one token per ~10 frames -> strongly diagonal cost favors monotonicity
    frames = npu_dtw.dtw_token_frames(attn, real_frames=1500)
    assert (np.diff(frames) >= 0).all()
    assert frames[0] >= 0 and frames[-1] <= 1500 - 1


def test_trim_to_real_frames():
    rng = np.random.default_rng(5)
    # peak attention far inside the padding zone must not attract tokens
    attn = rng.uniform(0.0, 0.01, size=(10, 1, 1500)).astype(np.float32)
    attn[:, :, 1490:1500] = 10.0  # garbage in the padded region
    frames = npu_dtw.dtw_token_frames(attn, real_frames=500)
    assert frames.max() <= 499
    # without trim the path must still traverse the padding before its end
    # row, distorting late tokens' entering frames far past the real audio
    frames_untrimmed = npu_dtw.dtw_token_frames(attn, real_frames=None)
    assert frames_untrimmed.max() > frames.max()


def test_head_collapse_equivalent():
    rng = np.random.default_rng(9)
    base = rng.uniform(0, 1, size=(20, 1, 200)).astype(np.float32)
    heads = np.repeat(base, 8, axis=1)
    f1 = npu_dtw.dtw_token_frames(base, real_frames=200)
    f8 = npu_dtw.dtw_token_frames(heads, real_frames=200)
    assert f1.tolist() == f8.tolist()


def test_placement_fallback_last_token():
    # constant cost: the path is forced diagonal; whisper.cpp leaves the last
    # token unplaced, the fallback fills it with the path's last time index
    cost = np.zeros((4, 10), dtype=np.float32)
    path_t, path_s = npu_dtw._dtw(cost)
    frames = npu_dtw._place_timestamps(path_t, path_s, 4)
    assert (frames >= 0).all() and len(frames) == 4


def test_token_times_ms_offset():
    attn = np.zeros((2, 1, 100), dtype=np.float32)
    times = npu_dtw.token_times_ms(attn, real_seconds=2.0, offset_ms=1500)
    assert times[0] >= 1500
    assert times.dtype == np.int64


def test_place_entering_semantics():
    # path [v=(0,0,1,1,2), t=(0,5,9,12,20)]: entering frames [0, 9, 20]
    pt = np.array([0, 0, 1, 1, 2])
    ps = np.array([0, 5, 9, 12, 20])
    frames = npu_dtw._place_timestamps(pt, ps, 3)
    assert frames.tolist() == [0, 9, 20]
    # relation to whisper.cpp t_dtw: entering[k] == t_dtw_cpp[k-1]

def test_refine_valley_between_bumps():
    # two attention bumps: token 0 centered at frame 20, token 1 at frame 80;
    # the valley (quietest point between peaks) becomes token 1's boundary
    T, A = 2, 100
    prof = np.zeros((T, A), dtype=np.float32)
    prof[0, 15:25] = 1.0
    prof[1, 75:85] = 1.0
    frames = npu_dtw._refine_valley(prof, np.array([10, 50]))
    # argmin-quietest semantics (the sweep-validated criterion): token 1's
    # profile is zero from peak(0)=15 until its bump at 75, so the quietest
    # frame in [peak(0), peak(1)] is 15
    assert frames.tolist() == [0, 15]
    # monotonicity under a degenerate flat profile
    flat = np.ones((5, 40), dtype=np.float32)
    frames = npu_dtw._refine_valley(flat, np.array([0, 10, 20, 30, 39]))
    assert (np.diff(frames) >= 0).all() and len(frames) == 5

def test_degenerate_inputs():
    assert npu_dtw.dtw_token_frames(np.zeros((0, 8, 100), dtype=np.float32)).size == 0
    # fewer frames than tokens: DTW still assigns every token a frame
    attn = np.zeros((6, 1, 2), dtype=np.float32)
    frames = npu_dtw.dtw_token_frames(attn, real_frames=2)
    assert len(frames) == 6 and (frames >= 0).all()


def test_group_words_spaced():
    texts = [b"He", b" hoped", b" there", b" would", b" be"]
    times = [100, 120, 200, 300, 340]
    words = npu_dtw.group_word_segments(texts, times, end_ms=500)
    assert words == [
        (100, 120, "He"),
        (120, 200, "hoped"),
        (200, 300, "there"),
        (300, 340, "would"),
        (340, 500, "be"),
    ]


def test_group_words_comma_token():
    texts = [b"ten", b", fresh", b" Nelly"]
    times = [100, 200, 300]
    words = npu_dtw.group_word_segments(texts, times, end_ms=400)
    assert words == [(100, 200, "ten,"), (200, 300, "fresh"), (300, 400, "Nelly")]


def test_group_words_non_space_cjk():
    # one token carrying two complete chars + one byte-split char pair
    texts = ["你".encode("utf-8"), "你好".encode("utf-8")]
    times = [100, 200]
    words = npu_dtw.group_word_segments(texts, times, end_ms=300, non_space_language=True)
    chars = [w[2] for w in words]
    assert chars == ["你", "你", "好"]
    assert all(w[1] == 200 for w in words[:2])  # token span shared


def test_group_words_byte_split_reassembles():
    # a UTF-8 char split across two tokens must reassemble to one char
    full = "好".encode("utf-8")
    texts = ["你".encode("utf-8"), full[:1], full[1:]]
    times = [100, 150, 200]
    words = npu_dtw.group_word_segments(texts, times, end_ms=300, non_space_language=True)
    assert [w[2] for w in words] == ["你", "好"]
