"""Download and prepare the Phase 2 audio listed in data/train/catalog.json.

Usage:  python ml\\fetch_train.py [--workers 4] [--only REC_ID ...] [--limit N]

For each recording:
  * role "train": whole surahs (never the evaluation or long surahs), in a
    per-recording random order, until the minute budget for its style is met.
    Written to data/train/<rec_id>/<SSS>.flac.
  * eval_surahs (held-out, unknown and enrollment audio): written to
    data/eval/<rec_id>/<SSS>.flac.
Everything is 16 kHz mono with silence removed. MP3s are deleted after decoding.
Safe to re-run: finished surahs and recordings are skipped.
Finally writes data/train/manifest.csv listing every training file.
"""
import argparse
import csv
import json
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

from audio import remove_silence
from config import (BUDGET_MIN, CATALOG_FILE, EVAL, EVAL_SURAHS, LONG_SURAHS,
                    MANIFEST_FILE, SAMPLE_RATE, SPLIT_SEED, TRAIN)

HEADERS = {"User-Agent": "QariID-research-prototype"}
TMP = TRAIN / "_tmp"
_local = __import__("threading").local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HEADERS)
    return _local.s


def download(url, dest):
    """Stream url to dest. Returns 'ok', 'missing' or 'failed'."""
    for attempt in range(4):
        try:
            with session().get(url, stream=True, timeout=(15, 120)) as r:
                if r.status_code == 404:
                    return "missing"
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return "ok"
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return "failed"


def decode(path, block_s=60):
    """Decode in blocks to 16 kHz mono float32 (keeps memory low for long files)."""
    sr = sf.info(str(path)).samplerate
    parts = []
    for block in sf.blocks(str(path), blocksize=sr * block_s, dtype="float32", always_2d=True):
        mono = block.mean(axis=1)
        if sr != SAMPLE_RATE:
            mono = AF.resample(torch.from_numpy(mono), sr, SAMPLE_RATE).numpy()
        parts.append(mono)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def fetch_surah(rec, surah, dest, max_seconds=None):
    """Download, decode, clean and save one surah. Returns seconds saved, or a status string."""
    if dest.exists():
        return sf.info(str(dest)).duration
    TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=TMP, suffix=".mp3", delete=False) as t:
        tmp = Path(t.name)
    try:
        status = download(f"{rec['url']}{surah:03d}.mp3", tmp)
        if status != "ok":
            return status
        try:
            wav = remove_silence(decode(tmp))
        except Exception as e:  # corrupt file
            print(f"  could not decode {rec['rec_id']}/{surah:03d}: {e}")
            return "corrupt"
        if max_seconds is not None:
            wav = wav[: int(max_seconds * SAMPLE_RATE)]
        if len(wav) < SAMPLE_RATE:      # under a second of speech: not useful
            return "empty"
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".part.flac")
        sf.write(str(part), wav, SAMPLE_RATE)
        part.replace(dest)
        return len(wav) / SAMPLE_RATE
    finally:
        tmp.unlink(missing_ok=True)


def surah_order(rec):
    excluded = set(EVAL_SURAHS) | set(LONG_SURAHS)
    surahs = [s for s in rec["surahs"] if s not in excluded]
    random.Random(f"{SPLIT_SEED}:{rec['rec_id']}").shuffle(surahs)
    return surahs


def process(rec):
    problems = []
    # Evaluation audio (fixed surahs, no budget).
    for s in rec["eval_surahs"]:
        if s not in rec["surahs"]:
            continue
        got = fetch_surah(rec, s, EVAL / rec["rec_id"] / f"{s:03d}.flac")
        if isinstance(got, str):
            problems.append(f"eval {s}: {got}")

    if rec["role"] != "train":
        return rec["rec_id"], 0.0, problems
    folder = TRAIN / rec["rec_id"]
    done = folder / "done.json"
    if done.exists():
        return rec["rec_id"], json.loads(done.read_text())["seconds"], problems

    budget = BUDGET_MIN[rec["style"]] * 60
    total = 0.0
    for s in surah_order(rec):
        if budget - total < 30:     # close enough; a tiny tail would be rejected as empty
            break
        got = fetch_surah(rec, s, folder / f"{s:03d}.flac", max_seconds=budget - total)
        if isinstance(got, str):
            problems.append(f"train {s}: {got}")
            if got == "failed":            # network trouble: do not mark done
                return rec["rec_id"], total, problems
        else:
            total += got
    folder.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({"seconds": total, "problems": problems}))
    return rec["rec_id"], total, problems


def write_manifest(cat):
    names = {k: v["name"] for k, v in cat["reciters"].items()}
    rows = []
    for rec in cat["recordings"]:
        if rec["role"] != "train":
            continue
        for f in sorted((TRAIN / rec["rec_id"]).glob("[0-9][0-9][0-9].flac")):
            rows.append({
                "path": f.relative_to(TRAIN).as_posix(),
                "reciter": rec["reciter"],
                "rec_id": rec["rec_id"],
                "style": rec["style"],
                "riwayah": rec["riwayah"],
                "surah": int(f.stem),
                "seconds": round(sf.info(str(f)).duration, 2),
                "name": names[rec["reciter"]],
            })
    with open(MANIFEST_FILE, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)
    hours = sum(r["seconds"] for r in rows) / 3600
    print(f"Manifest: {len(rows)} files, {hours:.1f} h, "
          f"{len({r['reciter'] for r in rows})} reciters -> {MANIFEST_FILE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="process only these rec_ids")
    ap.add_argument("--limit", type=int, help="process only the first N recordings")
    args = ap.parse_args()

    cat = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    recs = cat["recordings"]
    if args.only:
        recs = [r for r in recs if r["rec_id"] in set(args.only)]
    if args.limit:
        recs = recs[: args.limit]

    short, failed = [], []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(process, r) for r in recs]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="rec"):
            try:
                rec_id, seconds, problems = fut.result()
            except Exception as e:
                tqdm.write(f"  error: {e!r}")
                continue
            if any("failed" in p for p in problems):
                failed.append(rec_id)
            elif problems:
                tqdm.write(f"  {rec_id}: {len(problems)} problems: {problems[:4]}")
            rec = next(r for r in recs if r["rec_id"] == rec_id)
            if rec["role"] == "train" and seconds < 0.8 * BUDGET_MIN[rec["style"]] * 60:
                short.append((rec_id, round(seconds / 60, 1)))

    if failed:
        print(f"Network failures (re-run to retry): {failed}")
    if short:
        print(f"Recordings under 80% of budget (minutes): {short}")
    write_manifest(cat)


if __name__ == "__main__":
    main()
