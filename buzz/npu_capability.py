"""NPU (Hexagon) capability probe and chipset detection for the ORT-QNN engine.

P2 milestones M0/M1 (see porting/plans/P2/): the NPU engine is offered only when
the QNN execution provider imports cleanly AND enumerates a device, and the
downloaded model asset is picked by the machine's detected Snapdragon chipset
(x2-elite = HTP v81, x-elite = first-gen X Elite/X Plus = HTP v73; v73 contexts
also run on v81, v81 contexts are rejected on v73). Everything here is
defensive — these functions must never raise, so callers (model_loader, the
engine module, the fallback path) can treat a False/None return as "hide the
NPU engine, keep whisper.cpp".
"""
import functools
import logging
import platform
import re
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)

# ProcessorNameString, e.g. "Snapdragon(R) X2 Elite Extreme - X2E94100 - ..." or
# "Snapdragon(R) X Elite - X1E80100 - ...". X2 must be tested first: the X2
# strings also contain "X Elite".
_CHIPSET_X2_RE = re.compile(r"Snapdragon.*X2", re.IGNORECASE)
_CHIPSET_X1_RE = re.compile(r"Snapdragon.*(X1[EP]|\bX Elite)", re.IGNORECASE)
# Fallback signal when the registry name is unavailable: the Adreno GPU that
# ORT enumerates next to the NPU (X2-90/X2-85 vs X1-85 etc.).
_ADRENO_X2_RE = re.compile(r"Adreno.*X2-", re.IGNORECASE)
_ADRENO_X1_RE = re.compile(r"Adreno.*X1-", re.IGNORECASE)


@functools.cache
def _qnn_devices() -> List[object]:
    """Register the QNN EP plugin (best-effort) and return its devices.

    Never raises; an empty list covers every failure (wheels missing, backend
    unloadable, no Hexagon device). Requires ORT 1.27-style plugin registration:
    without it, get_ep_devices() only lists the CPU device.
    """
    try:
        import onnxruntime as ort
        import onnxruntime_qnn as qnn
    except Exception:  # missing wheels on odd installs, broken DLL loads
        logger.debug("ORT-QNN stack unavailable (import failed)", exc_info=True)
        return []

    try:
        ort.register_execution_provider_library(qnn.EP_NAME, qnn.get_library_path())
    except Exception:
        # Raises "EP library is already registered" when called twice in one
        # process — harmless, enumeration below still sees the QNN devices.
        logger.debug("QNN EP registration failed (or already registered)", exc_info=True)

    try:
        return [d for d in ort.get_ep_devices() if d.ep_name == qnn.EP_NAME]
    except Exception:  # no Hexagon device / backend load failure
        logger.debug("QNN device enumeration failed", exc_info=True)
        return []


@functools.cache
def is_npu_transcription_available() -> bool:
    """True when the QNN EP plugin loads and enumerates a (Hexagon NPU) device.

    Result is cached for the process lifetime: QNN device presence does not
    change while the app runs.
    """
    if sys.platform != "win32" or platform.machine() != "ARM64":
        return False
    return bool(_qnn_devices())


def get_qnn_ep_devices() -> List[object]:
    """The QNN EP devices ORT enumerated (empty when the stack is unavailable).

    Callers pass these to SessionOptions.add_provider_for_devices. Registers the
    plugin EP if that has not happened yet in this process.
    """
    return _qnn_devices()


@functools.cache
def _adreno_gpu_descriptions() -> tuple:
    """Descriptions of GPU devices the QNN EP enumerated (for chipset fallback)."""
    descriptions = []
    for device in _qnn_devices():
        try:
            description = dict(device.device.metadata).get("Description", "")
            if "Adreno" in description:
                descriptions.append(description)
        except Exception:
            logger.debug("Could not read QNN GPU device metadata", exc_info=True)
    return tuple(descriptions)


@functools.cache
def detected_chipset() -> Optional[str]:
    """The Snapdragon chipset generation of this machine, as an asset tag.

    Returns "x2-elite" (HTP v81), "x-elite" (first-gen X Elite / X Plus,
    HTP v73) or None when the machine is not a supported Snapdragon X.
    Cached for the process lifetime; never raises.
    """
    if sys.platform != "win32" or platform.machine() != "ARM64":
        return None

    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            processor_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    except OSError:
        logger.debug("Could not read processor name from the registry", exc_info=True)
        processor_name = ""

    if _CHIPSET_X2_RE.search(processor_name):
        return "x2-elite"
    if _CHIPSET_X1_RE.search(processor_name):
        return "x-elite"

    for description in _adreno_gpu_descriptions():
        if _ADRENO_X2_RE.search(description):
            return "x2-elite"
        if _ADRENO_X1_RE.search(description):
            return "x-elite"

    logger.debug("No supported Snapdragon chipset detected (cpu=%r)", processor_name)
    return None
