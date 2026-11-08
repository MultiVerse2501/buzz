"""NPU (Hexagon) capability probe for the ORT-QNN whisper engine.

P2 milestone M0 (see porting/plans/P2/M0-deps-capability-gate.md): the NPU engine
is offered only when the QNN execution provider imports cleanly AND enumerates a
device. Everything here is defensive — the probe must never raise, so callers
(model_loader entries in M1, the engine module in M2, the fallback path in M5)
can treat a False return as "hide the NPU engine, keep whisper.cpp".
"""
import functools
import logging
import platform
import sys

logger = logging.getLogger(__name__)


@functools.cache
def is_npu_transcription_available() -> bool:
    """True when the QNN EP plugin loads and enumerates a (Hexagon NPU) device.

    Result is cached for the process lifetime: QNN device presence does not
    change while the app runs.
    """
    if sys.platform != "win32" or platform.machine() != "ARM64":
        return False

    try:
        import onnxruntime as ort
        import onnxruntime_qnn as qnn
    except Exception:  # missing wheels on odd installs, broken DLL loads
        logger.debug("ORT-QNN stack unavailable (import failed)", exc_info=True)
        return False

    try:
        # The QNN EP ships as a plugin library: without this registration,
        # get_ep_devices() only lists the CPU device. Raises "EP library is
        # already registered" when called twice in one process — harmless, the
        # enumeration below still sees the QNN devices, so this is best-effort.
        ort.register_execution_provider_library(qnn.EP_NAME, qnn.get_library_path())
    except Exception:
        logger.debug("QNN EP registration failed (or already registered)", exc_info=True)
    try:
        return any(d.ep_name == qnn.EP_NAME for d in ort.get_ep_devices())
    except Exception:  # no Hexagon device / backend load failure
        logger.debug("QNN device enumeration failed", exc_info=True)
        return False
