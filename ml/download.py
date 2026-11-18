"""Download the per-ayah MP3s needed for Phase 1 from everyayah.com.

Usage:  python ml\\download.py
Safe to re-run: files already on disk are skipped.
"""
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

from config import (AYAH_COUNT, BASE_URL, CROSS, CROSS_SURAHS, ENROLL_SURAHS,
                    KNOWN, RAW, TEST_SURAHS, UNKNOWN)

WORKERS = 6
session = requests.Session()
session.headers["User-Agent"] = "QariID-research-prototype"


def jobs():
    """Yield (subfolder, filename) for every file the prototype needs."""
    plan = []
    for _, folder in KNOWN.values():
        plan.append((folder, ENROLL_SURAHS + TEST_SURAHS))
    for _, folder in UNKNOWN.values():
        plan.append((folder, TEST_SURAHS))
    for folder in CROSS.values():
        plan.append((folder, CROSS_SURAHS))
    for folder, surahs in plan:
        for s in surahs:
            for a in range(1, AYAH_COUNT[s - 1] + 1):
                yield folder, f"{s:03d}{a:03d}.mp3"


def fetch(job):
    folder, name = job
    dest = RAW / folder / name
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped", job
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            r = session.get(f"{BASE_URL}/{folder}/{name}", timeout=60)
            if r.status_code == 404:
                return "missing", job
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return "ok", job
        except requests.RequestException:
            continue
    return "failed", job


def main():
    todo = list(jobs())
    counts = {"ok": 0, "skipped": 0, "missing": 0, "failed": 0}
    problems = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for status, job in tqdm(pool.map(fetch, todo), total=len(todo), unit="file"):
            counts[status] += 1
            if status in ("missing", "failed"):
                problems.append((status, *job))
    print(counts)
    by_folder = {}
    for status, folder, _ in problems:
        by_folder.setdefault((folder, status), 0)
        by_folder[(folder, status)] += 1
    for (folder, status), n in sorted(by_folder.items()):
        print(f"  {status}: {n} files in {folder}")
    size = sum(f.stat().st_size for f in RAW.rglob("*.mp3")) / 1e9
    print(f"Total on disk: {size:.2f} GB in {RAW}")


if __name__ == "__main__":
    main()
