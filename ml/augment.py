"""Random training-time augmentation for 16 kHz clips.

Uses real room impulse responses and noises from OpenSLR 28 (data/aug/RIRS_NOISES)
when present; otherwise falls back to the synthetic versions in audio.py.
Each call picks a random chain, so the model sees clean audio as well as
reverberant, noisy, band-limited, low-bitrate and slightly sped-up audio.
"""
import io
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from audio import add_noise, add_reverb
from config import DATA, SAMPLE_RATE

AUG_ROOT = DATA / "aug" / "RIRS_NOISES"


def _list(patterns):
    files = []
    for p in patterns:
        files += sorted(AUG_ROOT.glob(p))
    return files


def _rms(x):
    return float(np.sqrt((x ** 2).mean() + 1e-12))


class Augmenter:
    def __init__(self, strength=1.0):
        self.rirs = _list(["simulated_rirs/*/*/*.wav", "real_rirs_isotropic_noises/*rir*.wav"])
        self.noises = _list(["pointsource_noises/*.wav", "real_rirs_isotropic_noises/*noise*.wav"])
        self.s = strength
        self._cache = {}

    def describe(self):
        return f"{len(self.rirs)} RIRs, {len(self.noises)} noise files" + (
            "" if self.rirs else " (synthetic fallback)")

    def _read(self, path):
        # Small LRU-ish cache: workers keep a few hundred short files in memory.
        if path not in self._cache:
            if len(self._cache) > 400:
                self._cache.pop(next(iter(self._cache)))
            x, sr = sf.read(str(path), dtype="float32", always_2d=True)
            x = x[:, 0]
            if sr != SAMPLE_RATE:
                x = signal.resample_poly(x, SAMPLE_RATE, sr).astype(np.float32)
            self._cache[path] = x
        return self._cache[path]

    # -- individual effects ------------------------------------------------
    def speed(self, x, rng):
        f = rng.choice([0.95, 1.05])
        up, down = (20, 19) if f < 1 else (19, 20)      # resample = speed + pitch change
        y = signal.resample_poly(x, up, down).astype(np.float32)
        return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))

    def reverb(self, x, rng):
        if not self.rirs:
            return add_reverb(x, rng, rt60=rng.uniform(0.2, 1.2))
        rir = self._read(self.rirs[rng.integers(len(self.rirs))])
        rir = rir[np.argmax(np.abs(rir)):][: SAMPLE_RATE * 2]     # from direct path, max 2 s
        rir = rir / (np.sqrt((rir ** 2).sum()) + 1e-9)
        y = signal.fftconvolve(x, rir)[: len(x)].astype(np.float32)
        return y * (_rms(x) / _rms(y))

    def noise(self, x, rng):
        snr = rng.uniform(0, 20)
        if not self.noises or rng.random() < 0.3:
            return add_noise(x, rng, snr_db=snr)
        n = self._read(self.noises[rng.integers(len(self.noises))])
        if len(n) < len(x):
            n = np.tile(n, len(x) // len(n) + 1)
        start = rng.integers(len(n) - len(x) + 1)
        n = n[start:start + len(x)]
        n = n * (_rms(x) / (_rms(n) + 1e-9) / (10 ** (snr / 20)))
        return (x + n).astype(np.float32)

    def band(self, x, rng):
        low = rng.uniform(80, 400)
        high = rng.uniform(3000, 7500)
        sos = signal.butter(4, [low, high], btype="band", fs=SAMPLE_RATE, output="sos")
        return signal.sosfilt(sos, x).astype(np.float32)

    def codec(self, x, rng):
        buf = io.BytesIO()
        peak = np.abs(x).max() + 1e-9
        sf.write(buf, x / peak * 0.9, SAMPLE_RATE, format="MP3", subtype="MPEG_LAYER_III",
                 compression_level=float(rng.uniform(0.5, 1.0)), bitrate_mode="CONSTANT")
        buf.seek(0)
        y = sf.read(buf, dtype="float32")[0] * peak / 0.9
        return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))

    # -- random chain ------------------------------------------------------
    def __call__(self, x, rng):
        s = self.s
        if rng.random() < 0.2 * s:
            x = self.speed(x, rng)
        if rng.random() < 0.4 * s:
            x = self.reverb(x, rng)
        if rng.random() < 0.5 * s:
            x = self.noise(x, rng)
        if rng.random() < 0.3 * s:
            x = self.band(x, rng)
        if rng.random() < 0.15 * s:
            try:
                x = self.codec(x, rng)
            except Exception:
                pass
        x = x * (10 ** (rng.uniform(-6, 6) / 20))        # random gain
        return np.clip(x, -1, 1).astype(np.float32)
