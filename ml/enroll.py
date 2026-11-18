"""Build one voice vector per known reciter from the enrollment surahs.

Usage:  python ml\\enroll.py
Writes data\\reciters.npz (ids, names, vectors).
"""
import numpy as np

from audio import load_surahs, split_clips
from config import ENROLL_SURAHS, ENROLLMENT_FILE, KNOWN, SAMPLE_RATE
from embedder import Embedder

SEGMENT_SECONDS = 10


def main():
    embedder = Embedder()
    print(f"Model on {embedder.device}")
    ids, names, vectors = [], [], []
    for rid, (name, folder) in KNOWN.items():
        wav = load_surahs(folder, ENROLL_SURAHS)
        clips = split_clips(wav, SEGMENT_SECONDS)
        if len(clips) < 6:
            print(f"SKIP {rid}: only {len(wav) / SAMPLE_RATE:.0f} s of audio")
            continue
        centroid = embedder.embed(clips).mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        ids.append(rid)
        names.append(name)
        vectors.append(centroid)
        print(f"{rid:12s} {len(wav) / SAMPLE_RATE / 60:5.1f} min  {len(clips)} segments")
    np.savez(ENROLLMENT_FILE, ids=np.array(ids), names=np.array(names),
             vectors=np.stack(vectors).astype(np.float32))
    print(f"Enrolled {len(ids)} reciters -> {ENROLLMENT_FILE}")


if __name__ == "__main__":
    main()
