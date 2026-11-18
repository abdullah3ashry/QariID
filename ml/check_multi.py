"""Does enrolling a reciter from two recordings help on an unseen third one?

Usage:  python ml\\check_multi.py
For three reciters with several recordings on everyayah.com:
  A = murattal recording (the one used in Phase 1 enrollment)
  B = mujawwad recording
  C = a third recording, never used for enrollment (the test)
Compares three ways of enrolling:
  single     vector from A only                     (Phase 1 baseline)
  multi_avg  one vector averaged over A and B
  multi_max  separate vectors for A and B, best match wins
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from audio import degrade, load_surahs, split_clips
from config import (AYAH_COUNT, CROSS, ENROLL_SURAHS, ENROLLMENT_FILE, KNOWN,
                    RAW, RESULTS, TEST_SURAHS)
from download import fetch
from embedder import Embedder

THIRD = {
    "abdul_basit": "warsh/warsh_Abdul_Basit_128kbps",
    "husary": "Husary_Muallim_128kbps",
    "minshawy": "Menshawi_32kbps",
}
THIRD_SURAHS = [50, 18]
CONDITIONS = ("clean", "phone_far")
SECONDS = 10
MAX_CLIPS = 60


def download():
    jobs = []
    for folder in CROSS.values():
        jobs += [(folder, f"{s:03d}{a:03d}.mp3") for s in ENROLL_SURAHS
                 for a in range(1, AYAH_COUNT[s - 1] + 1)]
    for folder in THIRD.values():
        jobs += [(folder, f"{s:03d}{a:03d}.mp3") for s in THIRD_SURAHS
                 for a in range(1, AYAH_COUNT[s - 1] + 1)]
    counts = {}
    with ThreadPoolExecutor(6) as pool:
        for status, (folder, _) in pool.map(fetch, jobs):
            counts[(folder, status)] = counts.get((folder, status), 0) + 1
    for (folder, status), n in sorted(counts.items()):
        print(f"  {folder}: {n} {status}")


def same_recording(folder_a, folder_b, surah=50, ayahs=12):
    """Largest per-ayah duration difference (s); ~0 means the same recording."""
    diffs = []
    for a in range(1, ayahs + 1):
        fa = RAW / folder_a / f"{surah:03d}{a:03d}.mp3"
        fb = RAW / folder_b / f"{surah:03d}{a:03d}.mp3"
        if fa.exists() and fb.exists():
            diffs.append(abs(sf.info(str(fa)).duration - sf.info(str(fb)).duration))
    return (max(diffs), float(np.median(diffs))) if diffs else (None, None)


def centroid(embedder, folder, surahs):
    clips = split_clips(load_surahs(folder, surahs), SECONDS)
    c = embedder.embed(clips).mean(axis=0)
    return c / np.linalg.norm(c), len(clips)


def main():
    print("Downloading extra recordings...")
    download()

    print("\nIs the third recording really different from recording A?")
    for r, folder in THIRD.items():
        mx, med = same_recording(KNOWN[r][1], folder)
        note = "n/a" if mx is None else f"max {mx:.1f} s, median {med:.1f} s per-ayah duration difference"
        print(f"  {r}: {note}")

    enrolled = np.load(ENROLLMENT_FILE)
    ids = list(enrolled["ids"])
    single = enrolled["vectors"]
    embedder = Embedder()

    print("\nEnrolling from recording B (mujawwad)...")
    b_vec = {}
    for r, folder in CROSS.items():
        b_vec[r], n = centroid(embedder, folder, ENROLL_SURAHS)
        print(f"  {r}: {n} segments")

    multi_avg = single.copy()
    for r, v in b_vec.items():
        m = single[ids.index(r)] + v
        multi_avg[ids.index(r)] = m / np.linalg.norm(m)
    extra = np.stack([b_vec[r] for r in b_vec])
    extra_owner = np.array([ids.index(r) for r in b_vec])

    def score(emb, scheme):
        if scheme == "single":
            return emb @ single.T
        if scheme == "multi_avg":
            return emb @ multi_avg.T
        sims = emb @ single.T
        ex = emb @ extra.T
        for j, owner in enumerate(extra_owner):
            sims[:, owner] = np.maximum(sims[:, owner], ex[:, j])
        return sims

    def gather(wavs):
        clips, truth = [], []
        for r, wav in wavs.items():
            c = split_clips(wav, SECONDS, MAX_CLIPS)
            clips.append(c)
            truth += [ids.index(r)] * len(c)
        return np.concatenate(clips), np.array(truth)

    third_clips, third_truth = gather({r: load_surahs(f, THIRD_SURAHS) for r, f in THIRD.items()})
    known_clips, known_truth = gather({r: load_surahs(KNOWN[r][1], TEST_SURAHS) for r in ids})

    results = []
    for cond in CONDITIONS:
        t_emb = embedder.embed(degrade(third_clips, cond))
        k_emb = embedder.embed(degrade(known_clips, cond))
        for scheme in ("single", "multi_avg", "multi_max"):
            t_pred = score(t_emb, scheme).argmax(axis=1)
            k_pred = score(k_emb, scheme).argmax(axis=1)
            row = dict(condition=cond, scheme=scheme,
                       all_20_same_recording=float((k_pred == known_truth).mean()))
            for r in THIRD:
                mask = third_truth == ids.index(r)
                row[r] = float((t_pred[mask] == ids.index(r)).mean()) if mask.any() else None
                row[r + "_clips"] = int(mask.sum())
                wrong = t_pred[mask][t_pred[mask] != ids.index(r)]
                if len(wrong):
                    vals, cnt = np.unique(wrong, return_counts=True)
                    row[r + "_mistaken_for"] = ids[vals[cnt.argmax()]]
            results.append(row)

    def pct(v):
        return "  n/a" if v is None else f"{100 * v:4.0f}%"

    print(f"\nAccuracy on the unseen third recording ({SECONDS} s clips)")
    print(f"{'condition':<10} {'enrollment':<10} " + " ".join(f"{r:>12}" for r in THIRD)
          + f" {'all 20 (same rec.)':>20}")
    for row in results:
        print(f"{row['condition']:<10} {row['scheme']:<10} "
              + " ".join(f"{pct(row[r]):>12}" for r in THIRD)
              + f" {100 * row['all_20_same_recording']:>19.1f}%")
    print("clips per reciter: " + ", ".join(f"{r} {results[0][r + '_clips']}" for r in THIRD))
    for row in results:
        if row["scheme"] == "multi_max":
            mis = {r: row.get(r + "_mistaken_for") for r in THIRD if row.get(r + "_mistaken_for")}
            print(f"[{row['condition']}/multi_max] most often mistaken for: {mis}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"multi_enroll_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
