# sphn (Windows ARM64 stand-in)

The real `sphn` is a Rust audio extension with no Windows ARM64 wheels, which
blocks `demucs` (its `demucs/api.py` imports `sphn` at module level). This
package is a small pure-Python re-implementation of the API surface demucs
actually uses — `read()` and `write()` — backed by `soundfile`, distributed
under the real `sphn` name so demucs's dependency resolves.

It is built as a `py3-none-win_arm64` wheel and only ever installed on
Windows ARM64; every other platform keeps resolving the real `sphn` from its
own index. The `+arm64.shim` local version suffix makes installs
identifiable in logs. demucs additionally carries its own ffmpeg fallback for
audio loading, so this shim only needs to behave sanely, not bit-exactly.

Not part of the shim: the rest of the real sphn API (`read_webm`, and the
`duration` resampling path beyond simple truncation). Call sites in this
project (demucs 4.1.x) use none of those.
