"""Pure-Python stand-in for torchaudio (Windows ARM64).

Mirrors the small API surface used by this project (stable-ts and buzz):
``save``/``load``/``info`` via soundfile — matching torchaudio's own
soundfile backend, including its dtype-to-subtype table, so WAV output is
byte-identical — with a PyAV fallback for formats libsndfile cannot encode.
DSP lives in :mod:`torchaudio.functional` (julius-backed; different filter
family than the real kernels, documented there).

The ``+arm64.shim`` local version suffix keeps installs identifiable in
logs; on every other platform the real torchaudio resolves instead.
"""

from collections import namedtuple

import numpy as np
import soundfile
import torch

from . import functional

__version__ = "2.8.0+arm64.shim"

AudioMetaData = namedtuple(
    "AudioMetaData",
    ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
)

_SUBTYPE_TO_BITS_PER_SAMPLE = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
    "ULAW": 8,
    "ALAW": 8,
    "VORBIS": 0,
    "MPEG_LAYER_III": 0,
}


def _get_subtype_for_wav(dtype, encoding, bits_per_sample):
    if not encoding:
        if not bits_per_sample:
            subtype = {
                torch.uint8: "PCM_U8",
                torch.int16: "PCM_16",
                torch.int32: "PCM_32",
                torch.float32: "FLOAT",
                torch.float64: "DOUBLE",
            }.get(dtype)
            if not subtype:
                raise ValueError(f"Unsupported dtype for wav: {dtype}")
            return subtype
        if bits_per_sample == 8:
            return "PCM_U8"
        return f"PCM_{bits_per_sample}"
    if encoding == "PCM_S":
        if not bits_per_sample:
            return "PCM_32"
        if bits_per_sample == 8:
            raise ValueError("wav does not support 8-bit signed PCM encoding.")
        return f"PCM_{bits_per_sample}"
    if encoding == "PCM_U":
        if bits_per_sample in (None, 8):
            return "PCM_U8"
        raise ValueError("wav only supports 8-bit unsigned PCM encoding.")
    if encoding == "PCM_F":
        if bits_per_sample in (None, 32):
            return "FLOAT"
        if bits_per_sample == 64:
            return "DOUBLE"
        raise ValueError("wav only supports 32/64-bit float PCM encoding.")
    if encoding == "ULAW":
        if bits_per_sample in (None, 8):
            return "ULAW"
        raise ValueError("wav only supports 8-bit mu-law encoding.")
    if encoding == "ALAW":
        if bits_per_sample in (None, 8):
            return "ALAW"
        raise ValueError("wav only supports 8-bit a-law encoding.")
    raise ValueError(f"Unexpected encoding: {encoding}")


def _get_subtype_for_sphere(encoding, bits_per_sample):
    if encoding in (None, "PCM_S"):
        return f"PCM_{bits_per_sample}" if bits_per_sample else "PCM_32"
    if encoding in ("PCM_U", "PCM_F"):
        raise ValueError(f"sph does not support {encoding} encoding.")
    if encoding == "ULAW":
        if bits_per_sample in (None, 8):
            return "ULAW"
        raise ValueError("sph only supports 8-bit for mu-law encoding.")
    if encoding == "ALAW":
        return "ALAW"
    raise ValueError(f"sph does not support {encoding}.")


def _get_subtype(dtype, format, encoding, bits_per_sample):
    if format == "wav":
        return _get_subtype_for_wav(dtype, encoding, bits_per_sample)
    if format == "flac":
        if encoding:
            raise ValueError("flac does not support encoding.")
        if not bits_per_sample:
            return "PCM_16"
        if bits_per_sample > 24:
            raise ValueError("flac does not support bits_per_sample > 24.")
        return "PCM_S8" if bits_per_sample == 8 else f"PCM_{bits_per_sample}"
    if format in ("ogg", "vorbis"):
        if bits_per_sample:
            raise ValueError("ogg/vorbis does not support bits_per_sample.")
        if encoding is None or encoding == "vorbis":
            return "VORBIS"
        if encoding == "opus":
            return "OPUS"
        raise ValueError(f"Unexpected encoding: {encoding}")
    if format == "mp3":
        return "MPEG_LAYER_III"
    if format == "sph":
        return _get_subtype_for_sphere(encoding, bits_per_sample)
    if format in ("nis", "nist"):
        return "PCM_16"
    raise ValueError(f"Unsupported format: {format}")


def save(
    filepath,
    src,
    sample_rate,
    channels_first=True,
    compression=None,
    format=None,
    encoding=None,
    bits_per_sample=None,
):
    """Save audio data to a path or file-like object, torchaudio-compatible.

    Mirrors torchaudio 2.8.0's soundfile backend: 2-D tensor input,
    extension-inferred format for paths, ``format`` required (RuntimeError)
    for file objects, dtype-derived subtype, NIST override for
    sph/nist/nis. Formats libsndfile cannot encode fall back to PyAV.
    """
    if src.ndim != 2:
        raise ValueError(f"Expected 2D Tensor, got {src.ndim}D.")
    if compression is not None:
        import warnings

        warnings.warn(
            '`save` function of the arm64 shim does not support the "compression" '
            "parameter. The argument is silently ignored."
        )
    if hasattr(filepath, "write"):
        if format is None:
            raise RuntimeError("`format` is required when saving to file object.")
        ext = format.lower()
    else:
        ext = str(filepath).split(".")[-1].lower()

    if bits_per_sample not in (None, 8, 16, 24, 32, 64):
        raise ValueError("Invalid bits_per_sample.")
    if bits_per_sample == 24:
        import warnings

        warnings.warn(
            "Saving audio with 24 bits per sample might warp samples near -1. "
            "Using 16 bits per sample might be able to avoid this."
        )
    subtype = _get_subtype(src.dtype, ext, encoding, bits_per_sample)

    # sph is a TED-LIUM extension that soundfile does not recognize as NIST;
    # extend the format manually, exactly like the real backend.
    if ext in ("nis", "nist", "sph") and format is None:
        format = "NIST"

    data = src.t() if channels_first else src

    try:
        soundfile.write(
            file=filepath, data=data, samplerate=sample_rate, subtype=subtype, format=format
        )
    except (soundfile.LibsndfileError, ValueError):
        _save_with_av(filepath, data, sample_rate, ext)


def _save_with_av(filepath, data, sample_rate, ext):
    """PyAV fallback for formats libsndfile cannot encode (mono/stereo)."""
    import av

    spec = {
        "wav": ("wav", "pcm_s16le", "s16"),
        "flac": ("flac", "flac", "s32"),
        "mp3": ("mp3", "mp3", "fltp"),
        "ogg": ("ogg", "libvorbis", "fltp"),
        "opus": ("ogg", "libopus", "fltp"),
    }.get(ext)
    if spec is None:
        raise ValueError(f"Unsupported format for the PyAV fallback: {ext}")
    container_fmt, codec, av_format = spec
    channels = int(data.shape[1])
    if channels > 2:
        raise ValueError(f"PyAV fallback supports mono/stereo, got {channels} channels.")
    layout = "mono" if channels == 1 else "stereo"
    arr = np.ascontiguousarray(data.numpy())  # (frames, channels), float32
    with av.open(filepath, mode="w", format=container_fmt) as container:
        stream = container.add_stream(codec, rate=sample_rate)
        stream.layout = layout
        if av_format == "fltp":
            payload = np.ascontiguousarray(arr.T)  # (channels, frames)
        elif av_format == "s16":
            payload = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).reshape(1, -1)
        else:  # s32
            payload = (np.clip(arr, -1.0, 1.0) * 2147483647.0).astype(np.int32).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(payload, format=av_format, layout=layout)
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
    """Load audio data, torchaudio-compatible.

    Mirrors the soundfile backend: float32 (channel, time) output by
    default; integer WAV keeps its dtype when ``normalize=False``.
    """
    with soundfile.SoundFile(filepath, "r") as file_:
        if file_.format != "WAV" or normalize:
            dtype = "float32"
        else:
            # Same table as the real soundfile backend's _SUBTYPE2DTYPE.
            dtype = {
                "PCM_S8": "int8",
                "PCM_U8": "uint8",
                "PCM_16": "int16",
                "PCM_32": "int32",
                "FLOAT": "float32",
                "DOUBLE": "float64",
            }.get(file_.subtype)
            if dtype is None:
                raise ValueError(f"Unsupported subtype: {file_.subtype}")

        if frame_offset > 0:
            file_.seek(frame_offset)
        frames = num_frames if num_frames >= 0 else -1
        waveform = file_.read(frames, dtype, always_2d=True)
        sample_rate = file_.samplerate

    waveform = torch.from_numpy(waveform)
    if channels_first:
        waveform = waveform.t()
    return waveform, sample_rate


def info(filepath):
    """Return AudioMetaData for a file (simplified soundfile-backend equivalent)."""
    sf_info = soundfile.info(filepath)
    return AudioMetaData(
        sample_rate=sf_info.samplerate,
        num_frames=sf_info.frames,
        num_channels=sf_info.channels,
        bits_per_sample=_SUBTYPE_TO_BITS_PER_SAMPLE.get(sf_info.subtype, 0),
        encoding=sf_info.format,
    )


def list_audio_backends():
    return ["soundfile"]


def get_audio_backend():
    return "soundfile"
