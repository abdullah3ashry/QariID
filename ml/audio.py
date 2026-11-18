"""Audio loading, silence removal, clipping and degradation helpers."""
import io

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy import signal

from config import AYAH_COUNT, CACHE, RAW, SAMPLE_RATE


def _decode(path):
    """Read an audio file as mono float32 at 16 kHz."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = AF.resample(torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    return wav


def remove_silence(wav, frame_ms=20, rel_db=-35.0):
    """Drop frames quieter than rel_db below the loud level of the recording."""
    n = int(SAMPLE_RATE * frame_ms / 1000)
    usable = len(wav) // n * n
    if usable == 0:
        return wav
    frames = wav[:usable].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    loud = np.percentile(rms, 95)
    keep = rms > loud * 10 ** (rel_db / 20)
    return frames[keep].reshape(-1)


def load_surah(folder, surah):
    """All ayahs of one surah by one reciter, concatenated, silence removed.

    Returns None if no files exist. Result is cached as FLAC.
    """
    cached = CACHE / folder / f"{surah:03d}.flac"
    if cached.exists():
        return sf.read(str(cached), dtype="float32")[0]
    parts = []
    for a in range(1, AYAH_COUNT[surah - 1] + 1):
        f = RAW / folder / f"{surah:03d}{a:03d}.mp3"
        if not f.exists():
            continue
        try:
            parts.append(_decode(f))
        except Exception as e:  # corrupt download
            print(f"  could not decode {f}: {e}")
    if not parts:
        return None
    wav = remove_silence(np.concatenate(parts))
    cached.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(cached), wav, SAMPLE_RATE)
    return wav


def load_surahs(folder, surahs):
    parts = [w for w in (load_surah(folder, s) for s in surahs) if w is not None]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def split_clips(wav, seconds, max_clips=None, seed=0):
    """Non-overlapping clips of a fixed length; a random subset if capped."""
    n = int(seconds * SAMPLE_RATE)
    count = len(wav) // n
    clips = wav[: count * n].reshape(count, n)
    if max_clips is not None and count > max_clips:
        idx = np.sort(np.random.default_rng(seed).choice(count, max_clips, replace=False))
        clips = clips[idx]
    return clips


# ---------------------------------------------------------------------------
# Degradations: rough simulations of real listening conditions.
# Each takes and returns a 1-D float32 clip at 16 kHz.
# ---------------------------------------------------------------------------

def _rms(x):
    return float(np.sqrt((x ** 2).mean() + 1e-12))


def add_noise(clip, rng, snr_db=10.0):
    """Pink-ish background noise at the given signal-to-noise ratio."""
    white = rng.standard_normal(len(clip)).astype(np.float32)
    b, a = signal.butter(1, 0.05)           # tilt the spectrum towards lows
    noise = signal.lfilter(b, a, white).astype(np.float32)
    noise *= _rms(clip) / _rms(noise) / (10 ** (snr_db / 20))
    return clip + noise


def add_reverb(clip, rng, rt60=0.9):
    """Synthetic room: exponentially decaying noise as the impulse response."""
    n = int(rt60 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir *= 0.5 / np.sqrt((rir ** 2).sum())  # reverberant energy vs direct sound
    rir[0] = 1.0
    out = signal.fftconvolve(clip, rir)[: len(clip)].astype(np.float32)
    return out * (_rms(clip) / _rms(out))


def band_limit(clip, rng=None, low=300.0, high=3400.0):
    """AM-radio / telephone style narrow band."""
    sos = signal.butter(4, [low, high], btype="band", fs=SAMPLE_RATE, output="sos")
    return signal.sosfilt(sos, clip).astype(np.float32)


def mp3_low(clip, rng=None):
    """Round trip through a very low bitrate MP3 encoding (poor stream)."""
    buf = io.BytesIO()
    sf.write(buf, clip, SAMPLE_RATE, format="MP3", subtype="MPEG_LAYER_III",
             compression_level=0.9, bitrate_mode="CONSTANT")
    buf.seek(0)
    out = sf.read(buf, dtype="float32")[0]
    if len(out) < len(clip):
        out = np.pad(out, (0, len(clip) - len(out)))
    return out[: len(clip)]


def phone_far(clip, rng):
    """A phone hearing a loudspeaker across a room: reverb + noise + thin band."""
    return band_limit(add_noise(add_reverb(clip, rng), rng, snr_db=10.0),
                      low=200.0, high=4000.0)


CONDITIONS = {
    "clean": lambda clip, rng: clip,
    "noise_10dB": add_noise,
    "reverb": add_reverb,
    "narrow_band": band_limit,
    "mp3_low": mp3_low,
    "phone_far": phone_far,
}


def degrade(clips, condition, seed=0):
    rng = np.random.default_rng(seed)
    fn = CONDITIONS[condition]
    return np.stack([fn(c, rng) for c in clips]).astype(np.float32)
