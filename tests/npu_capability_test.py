import platform
import sys

import pytest

from buzz.npu_capability import is_npu_transcription_available


def test_probe_returns_bool_and_never_raises():
    result = is_npu_transcription_available()
    assert isinstance(result, bool)


def test_probe_false_off_win32_arm64(monkeypatch):
    # The fast-exit path: anything that is not win32 + ARM64 has no Hexagon NPU
    # engine, regardless of what onnxruntime reports.
    if sys.platform == "win32" and platform.machine() == "ARM64":
        pytest.skip("host is win32-ARM64; off-platform path not testable here")
    is_npu_transcription_available.cache_clear()
    monkeypatch.setattr(sys, "platform", "linux")
    try:
        assert is_npu_transcription_available() is False
    finally:
        is_npu_transcription_available.cache_clear()


def test_probe_cached():
    is_npu_transcription_available.cache_clear()
    is_npu_transcription_available()
    is_npu_transcription_available()
    assert is_npu_transcription_available.cache_info().hits >= 1
