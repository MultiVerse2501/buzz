import io
import platform

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows" or platform.machine() != "ARM64",
    reason="The pure-Python torchaudio/sphn stand-ins only install on Windows ARM64",
)

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")
sphn = pytest.importorskip("sphn")
soundfile = pytest.importorskip("soundfile")


class TestTorchaudioShim:
    def test_version_identifies_the_shim(self):
        assert torchaudio.__version__ == "2.8.0+arm64.shim"
        assert sphn.__version__ == "0.2.1+arm64.shim"

    def test_save_wav_matches_real_torchaudio_subtype(self, tmp_path):
        # torchaudio 2.8's soundfile backend writes float32 tensors as
        # subtype FLOAT; the shim must match byte for byte.
        src = torch.from_numpy(
            np.sin(np.linspace(0, 3.14, 16000, dtype=np.float32)).reshape(1, -1)
        )
        path = str(tmp_path / "float.wav")
        torchaudio.save(path, src, 16000)
        info = soundfile.info(path)
        assert (info.format, info.subtype, info.frames) == ("WAV", "FLOAT", 16000)

    def test_save_int16_uses_pcm16(self, tmp_path):
        src = (torch.randn(1, 8000) * 1000).to(torch.int16)
        path = str(tmp_path / "int.wav")
        torchaudio.save(path, src, 16000)
        assert soundfile.info(path).subtype == "PCM_16"

    def test_save_to_file_object_requires_format(self):
        with pytest.raises(RuntimeError):
            torchaudio.save(io.BytesIO(), torch.zeros(1, 100), 16000)

    def test_save_unknown_extension_raises_value_error(self, tmp_path):
        # stable-ts catches ValueError around save; keep that contract.
        with pytest.raises(ValueError):
            torchaudio.save(str(tmp_path / "clip.xyz"), torch.zeros(1, 100), 16000)

    def test_save_to_file_object_roundtrips(self, tmp_path):
        src = torch.from_numpy(
            np.sin(np.linspace(0, 3.14, 16000, dtype=np.float32)).reshape(1, -1)
        )
        buffer = io.BytesIO()
        torchaudio.save(buffer, src, 16000, format="wav")
        buffer.seek(0)
        loaded, sr = torchaudio.load(buffer)
        assert sr == 16000
        assert torch.equal(loaded, src)

    def test_load_roundtrip_is_lossless(self, tmp_path):
        src = torch.from_numpy(
            np.sin(np.linspace(0, 3.14, 16000, dtype=np.float32)).reshape(1, -1)
        )
        path = str(tmp_path / "a.wav")
        torchaudio.save(path, src, 16000)
        loaded, sr = torchaudio.load(path)
        assert (sr, tuple(loaded.shape)) == (16000, (1, 16000))
        assert torch.equal(loaded, src)

    def test_load_normalize_false_keeps_int16(self, tmp_path):
        src = (torch.randn(1, 8000) * 1000).to(torch.int16)
        path = str(tmp_path / "int.wav")
        torchaudio.save(path, src, 16000)
        raw, _ = torchaudio.load(path, normalize=False)
        assert raw.dtype == torch.int16
        normalized, _ = torchaudio.load(path)
        assert normalized.dtype == torch.float32

    def test_resample_identity_and_ratio(self):
        x = torch.sin(torch.linspace(0, 1000, 48000))
        assert torch.equal(torchaudio.functional.resample(x, 48000, 48000), x)
        down = torchaudio.functional.resample(x, 48000, 16000)
        assert tuple(down.shape) == (16000,)
        assert bool(torch.isfinite(down).all())

    def test_biquads_band_limit(self):
        x = torch.sin(torch.linspace(0, 1000, 48000))
        low = torchaudio.functional.lowpass_biquad(x, 48000, 5000)
        high = torchaudio.functional.highpass_biquad(x, 48000, 200)
        band = torchaudio.functional.highpass_biquad(low, 48000, 200)
        assert bool(torch.isfinite(low).all()) and bool(torch.isfinite(high).all())
        # the band-limited signal must retain only a small fraction of the
        # energy of the full sweep
        assert float((band**2).sum() / (x**2).sum()) < 0.1

    def test_biquad_rejects_cutoff_above_nyquist(self):
        x = torch.zeros(4800)
        with pytest.raises(ValueError):
            torchaudio.functional.lowpass_biquad(x, 48000, 40000)


class TestSphnShim:
    def test_read_is_channels_first_float32(self, tmp_path):
        frames = np.sin(np.linspace(0, 3.14, 8000, dtype=np.float32))
        path = str(tmp_path / "mono.wav")
        soundfile.write(path, frames, 8000)
        data, sr = sphn.read(path)
        assert data.shape == (1, 8000)
        assert data.dtype == np.float32
        assert sr == 8000

    def test_write_read_roundtrip(self, tmp_path):
        data = np.sin(np.linspace(0, 3.14, 8000, dtype=np.float32)).reshape(1, -1)
        path = str(tmp_path / "rt.wav")
        sphn.write(path, data, 8000)
        back, sr = sphn.read(path)
        assert sr == 8000
        assert np.abs(back - data).max() == 0.0


class TestRestoredFeatures:
    def test_stable_whisper_imports_without_openai_whisper(self):
        # tiktoken (openai-whisper) stays gated on ARM64; stable-ts's
        # compatibility layer must degrade to whisper-less mode.
        import stable_whisper

        assert stable_whisper.whisper_compatibility.whisper is None

    def test_demucs_api_imports(self):
        from demucs import api as demucs_api

        assert demucs_api.Separator is not None

    def test_buzz_file_transcriber_sees_stable_whisper(self):
        from buzz.transcriber import whisper_file_transcriber

        assert whisper_file_transcriber.stable_whisper is not None

    def test_transformers_sees_torchaudio(self):
        import transformers

        assert transformers.utils.is_torchaudio_available() is True
