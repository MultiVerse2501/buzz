# torchaudio (Windows ARM64 stand-in)

The real `torchaudio` publishes no Windows ARM64 wheels, which blocks
`stable-ts` (hard metadata dependency) and with it the transcript-resizer
plugin and the openai-whisper engine's word-level timings. This package is a
pure-Python re-implementation of the API surface this project actually uses,
distributed under the real `torchaudio` name so stable-ts's dependency
resolves. It is built as a `py3-none-win_arm64` wheel and only ever
installed on Windows ARM64; every other platform keeps resolving the real
torchaudio. The `+arm64.shim` local version suffix makes installs
identifiable in logs.

Coverage and fidelity, verified against torchaudio 2.8.0's soundfile backend
source:

- `save` / `load` / `info` mirror the soundfile backend, including the
  dtype-to-subtype table, the file-object `format` requirement (RuntimeError),
  extension inference and the NIST extension override — so WAV output is
  byte-identical to real torchaudio. Formats libsndfile cannot encode fall
  back to PyAV (mono/stereo).
- `functional.resample` uses julius (windowed-sinc) instead of torchaudio's
  sinc_interpolation/kaiser kernels; `functional.lowpass_biquad` /
  `highpass_biquad` use julius's FFT filters instead of RBJ biquads. Same
  purpose, different filter family — acceptable for ASR preprocessing, and
  documented here rather than hidden.

Not implemented: the rest of torchaudio (sox_io backend, kaldi utilities,
models, datasets, compliance). The verified call sites in this project are
stable-ts 2.19.1 (`save`, `functional.resample`,
`functional.lowpass_biquad`/`highpass_biquad`) and one conditional resample
in `buzz/transformers_whisper.py`.
