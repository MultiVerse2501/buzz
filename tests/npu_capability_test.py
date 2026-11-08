import platform
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from buzz.npu_capability import detected_chipset, is_npu_transcription_available


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


def _fake_winreg(processor_name):
    """A winreg stand-in whose ProcessorNameString is `processor_name`."""

    @contextmanager
    def open_key(*args, **kwargs):
        yield SimpleNamespace()

    return SimpleNamespace(
        HKEY_LOCAL_MACHINE="HKLM",
        OpenKey=open_key,
        QueryValueEx=lambda key, value: (processor_name, "REG_SZ"),
    )


def _patch_windows_arm64(monkeypatch, processor_name):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(processor_name))


@pytest.mark.parametrize(
    "processor_name,expected",
    [
        # This machine's actual registry string
        ("Snapdragon(R) X2 Elite Extreme - X2E94100 - Qualcomm Oryon(TM) CPU", "x2-elite"),
        ("Snapdragon(R) X2 Elite - X2E80100 - Qualcomm Oryon(TM) CPU", "x2-elite"),
        ("Snapdragon(R) X Elite - X1E80100 - Qualcomm Oryon(TM) CPU", "x-elite"),
        ("Snapdragon(R) X1 Plus - X1P42100 - Qualcomm Oryon(TM) CPU", "x-elite"),
        ("AMD Ryzen 9 7950X", None),
        ("", None),
    ],
)
def test_detected_chipset_from_registry(monkeypatch, processor_name, expected):
    _patch_windows_arm64(monkeypatch, processor_name)
    monkeypatch.setattr(
        "buzz.npu_capability._adreno_gpu_descriptions", lambda: ())
    detected_chipset.cache_clear()
    try:
        assert detected_chipset() == expected
    finally:
        detected_chipset.cache_clear()


def test_detected_chipset_falls_back_to_adreno_gpu(monkeypatch):
    _patch_windows_arm64(monkeypatch, "")  # registry name unavailable
    monkeypatch.setattr(
        "buzz.npu_capability._adreno_gpu_descriptions",
        lambda: ("Qualcomm(R) Adreno(TM) X1-85 GPU",))
    detected_chipset.cache_clear()
    try:
        assert detected_chipset() == "x-elite"
    finally:
        detected_chipset.cache_clear()


def test_detected_chipset_none_off_windows_arm64(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    detected_chipset.cache_clear()
    try:
        assert detected_chipset() is None
    finally:
        detected_chipset.cache_clear()


def test_detected_chipset_cached(monkeypatch):
    _patch_windows_arm64(monkeypatch, "Snapdragon(R) X2 Elite - X2E80100")
    monkeypatch.setattr(
        "buzz.npu_capability._adreno_gpu_descriptions", lambda: ())
    detected_chipset.cache_clear()
    try:
        detected_chipset()
        detected_chipset()
        assert detected_chipset.cache_info().hits >= 1
    finally:
        detected_chipset.cache_clear()
