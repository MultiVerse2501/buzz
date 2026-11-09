"""torchaudio.functional surface, julius-backed (Windows ARM64 shim).

Deviations from the real kernels, same purpose, documented on purpose:

- ``resample`` uses julius's windowed-sinc resampler instead of
  torchaudio's ``sinc_interpolation``/``kaiser_window`` kernels. The
  ``lowpass_filter_width``/``rolloff``/``resampling_method``/``beta``
  arguments are accepted for signature compatibility; only ``rolloff`` is
  forwarded.
- ``lowpass_biquad``/``highpass_biquad`` use julius's FFT-based filters
  instead of RBJ biquads; the ``Q`` argument is accepted and ignored.
  Different filter family, equivalent band-limiting for ASR preprocessing.

Shapes: 1-D is treated as mono ``(1, time)`` and squeezed back; 2-D/3-D
pass through to julius as ``(channel, time)``/``(batch, channel, time)``.
"""

import torch

__all__ = ["resample", "lowpass_biquad", "highpass_biquad"]


def resample(
    waveform,
    orig_freq,
    new_freq,
    lowpass_filter_width=6,
    rolloff=0.99,
    resampling_method="sinc_interpolation",
    beta=None,
):
    """Resample along the last dimension via julius (windowed-sinc)."""
    if orig_freq == new_freq:
        return waveform
    import julius

    kwargs = {}
    if rolloff is not None:
        kwargs["rolloff"] = float(rolloff)
    squeeze = waveform.ndim == 1
    out = julius.resample_frac(
        waveform.unsqueeze(0) if squeeze else waveform, int(orig_freq), int(new_freq), **kwargs
    )
    return out.squeeze(0) if squeeze else out


def lowpass_biquad(waveform, sample_rate, cutoff_freq, Q=0.707):
    """Low-pass via julius's FFT filter (replaces the RBJ biquad)."""
    return _julius_filter("LowPassFilter", waveform, cutoff_freq, sample_rate)


def highpass_biquad(waveform, sample_rate, cutoff_freq, Q=0.707):
    """High-pass via julius's FFT filter (replaces the RBJ biquad)."""
    return _julius_filter("HighPassFilter", waveform, cutoff_freq, sample_rate)


def _julius_filter(class_name, waveform, cutoff_freq, sample_rate):
    import julius.filters

    # julius filters take the cutoff as a fraction of the sample rate
    # (Nyquist = 0.5), not Hz.
    cutoff = float(cutoff_freq) / float(sample_rate)
    if not 0 < cutoff < 0.5:
        raise ValueError(
            f"cutoff_freq {cutoff_freq} must be between 0 and {float(sample_rate) / 2} Hz"
        )
    filter_class = getattr(julius.filters, class_name)
    squeeze = waveform.ndim == 1
    out = filter_class(cutoff)(waveform.unsqueeze(0) if squeeze else waveform)
    return out.squeeze(0) if squeeze else out
