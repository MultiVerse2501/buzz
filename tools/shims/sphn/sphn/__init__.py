"""Pure-Python stand-in for the sphn audio extension (Windows ARM64).

Implements the ``read``/``write`` API that demucs uses, backed by
``soundfile``, for platforms where the real sphn Rust extension has no
wheels (Windows on ARM64). Distributed under the real ``sphn`` name so
demucs's dependency resolves; the ``+arm64.shim`` local version suffix
keeps installs identifiable in logs.
"""

__version__ = "0.2.1+arm64.shim"

import numpy as np
import soundfile as sf


def read(filename, sample_rate=-1, duration=None):
    """Read an audio file, returning ``(data, sample_rate)``.

    ``data`` is a ``(channels, samples)`` float32 array scaled to [-1, 1],
    matching the real sphn's contract (demucs feeds it straight into
    ``torch.from_numpy`` + ``convert_audio``).

    ``sample_rate`` optionally resamples on load (matching real sphn;
    uses julius/torch, imported lazily). ``duration`` optionally truncates
    to that many seconds.
    """
    data, sr = sf.read(filename, dtype="float32", always_2d=True)
    data = np.ascontiguousarray(data.T)  # (samples, channels) -> (channels, samples)

    if sample_rate is not None and int(sample_rate) > 0 and int(sample_rate) != int(sr):
        import torch

        import julius

        wav = julius.resample_frac(torch.from_numpy(data), int(sr), int(sample_rate))
        data = np.ascontiguousarray(wav.numpy())
        sr = int(sample_rate)

    if duration is not None and float(duration) > 0:
        data = data[:, : int(float(duration) * sr)]

    return data, int(sr)


def write(filename, data, sample_rate):
    """Write ``(channels, samples)`` float audio to ``filename``.

    Mono 1-D input is also accepted. Real sphn writes via its Rust encoder;
    here libsndfile does the work, which covers WAV/FLAC/OGG/MP3.
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        # sphn's convention is (channels, samples); soundfile wants (frames, channels).
        arr = arr.T
    # keep float input lossless (soundfile's default subtype is PCM_16)
    subtype = "FLOAT" if arr.dtype.kind == "f" else None
    if subtype:
        sf.write(filename, arr, int(sample_rate), subtype=subtype)
    else:
        sf.write(filename, arr, int(sample_rate))
