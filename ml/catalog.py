"""Build the Phase 2 dataset catalog: which reciters and recordings to fetch,
and the role each one plays (training, held-out test, enrollment).

Usage:  python ml\\catalog.py
Writes data/train/catalog.json and prints a summary. Deterministic for a given
mp3quran listing and SPLIT_SEED.
"""
import json
import random
from collections import Counter
from datetime import date

import requests

from config import (CATALOG_FILE, ENROLL_SURAHS, EVAL_SURAHS, EXTRA_UNKNOWN,
                    HELDOUT_RECORDING, HELDOUT_STYLE, KNOWN, LONG_SURAHS,
                    MAX_RECORDINGS_PER_RECITER, MIN_SURAHS_PER_RECORDING,
                    MP3QURAN_API, MP3QURAN_ID, QURANICAUDIO_EXTRA,
                    QURANICAUDIO_URL, SPLIT_SEED, TEST_SURAHS, UNKNOWN)

HEADERS = {"User-Agent": "QariID-research-prototype"}
EXCLUDED = set(EVAL_SURAHS) | set(LONG_SURAHS)


def style_of(moshaf_name):
    n = moshaf_name.lower()
    if "mo'lim" in n:
        return None              # teaching edition: includes other voices repeating
    if "mojawwad" in n or "mujawwad" in n:
        return "mujawwad"
    if "murattal" in n:
        return "murattal"
    return "other"


def train_surahs(available):
    return sorted(set(available) - EXCLUDED)


def priority(rec):
    """Lower sorts first: mujawwad, then Hafs murattal, then the rest."""
    if rec["style"] == "mujawwad":
        return 0
    if rec["style"] == "murattal" and rec["riwayah"] == 1:
        return 1
    return 2


def fetch_mp3quran():
    r = requests.get(MP3QURAN_API, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()["reciters"]


def build():
    phase1 = {v: k for k, v in MP3QURAN_ID.items()}
    reciters, recordings = {}, []

    for r in fetch_mp3quran():
        rid = f"mq{r['id']}"
        recs = []
        for m in r["moshaf"]:
            style = style_of(m["name"])
            if style is None:
                continue
            surahs = [int(s) for s in m["surah_list"].split(",") if s.strip()]
            recs.append({
                "rec_id": f"{rid}_m{m['id']}",
                "reciter": rid,
                "moshaf_id": m["id"],
                "source": "mp3quran",
                "url": m["server"],
                "style": style,
                "riwayah": m["rewaya_id"],
                "label": m["name"],
                "surahs": surahs,
            })
        if recs:
            reciters[rid] = {"name": r["name"].strip(), "phase1_id": phase1.get(r["id"])}
            recordings.extend(recs)

    for rid, (name, path, style) in QURANICAUDIO_EXTRA.items():
        reciters[rid] = {"name": name, "phase1_id": None}
        recordings.append({
            "rec_id": f"{rid}_main", "reciter": rid, "moshaf_id": None,
            "source": "quranicaudio", "url": f"{QURANICAUDIO_URL}/{path}",
            "style": style, "riwayah": 1, "label": f"quranicaudio {path}",
            "surahs": list(range(1, 115)),
        })

    by_reciter = {}
    for rec in recordings:
        rec["role"] = None
        rec["eval_surahs"] = []
        by_reciter.setdefault(rec["reciter"], []).append(rec)

    # 1. Held-out recordings of trained reciters (cross-style / cross-recording tests).
    heldout = {**{k: "heldout_style" for k in HELDOUT_STYLE},
               **{k: "heldout_recording" for k in HELDOUT_RECORDING}}
    for rec in recordings:
        if rec["moshaf_id"] in heldout:
            rec["role"] = heldout[rec["moshaf_id"]]
            rec["eval_surahs"] = list(TEST_SURAHS)

    # 2. Training recordings: up to N per reciter, by priority.
    for rid, recs in by_reciter.items():
        cands = [r for r in recs if r["role"] is None
                 and len(train_surahs(r["surahs"])) >= MIN_SURAHS_PER_RECORDING]
        cands.sort(key=lambda r: (priority(r), -len(r["surahs"])))
        # One per priority bucket first, so a reciter with both styles keeps
        # both (the pairing fine-tuning learns from); then fill remaining slots.
        firsts, seen = [], set()
        for rec in cands:
            if priority(rec) not in seen:
                seen.add(priority(rec))
                firsts.append(rec)
        picked = (firsts + [r for r in cands if r not in firsts])[:MAX_RECORDINGS_PER_RECITER]
        for rec in picked:
            rec["role"] = "train"

    # 3. Fully held-out "unknown" reciters: Phase 1 five plus random extras.
    def usable_for_test(rid):
        return [r for r in by_reciter[rid]
                if r["role"] == "train" and set(TEST_SURAHS) <= set(r["surahs"])]

    phase1_unknown = {f"mq{MP3QURAN_ID[k]}" for k in UNKNOWN}
    phase1_known = {f"mq{MP3QURAN_ID[k]}" for k in KNOWN}
    held_reciters = {r["reciter"] for r in recordings if r["role"] in heldout.values()}
    pool = sorted(rid for rid in by_reciter
                  if rid not in phase1_unknown | phase1_known | held_reciters
                  and len(usable_for_test(rid)) == 1
                  and sum(r["role"] == "train" for r in by_reciter[rid]) == 1
                  and not rid.startswith("qa"))
    extra = random.Random(SPLIT_SEED).sample(pool, EXTRA_UNKNOWN)
    unknown = sorted(phase1_unknown | set(extra))
    for rid in unknown:
        recs = usable_for_test(rid)
        if not recs:
            print(f"  warning: unknown reciter {rid} has no recording with surahs {TEST_SURAHS}")
        for rec in by_reciter[rid]:
            rec["role"] = None
        if recs:
            recs[0]["role"] = "unknown"
            recs[0]["eval_surahs"] = list(TEST_SURAHS)

    # 4. Enrollment audio for reciters that have held-out recordings: their main
    #    training recording on the enrollment surahs (never trained on).
    for rid in held_reciters:
        train = [r for r in by_reciter[rid] if r["role"] == "train"]
        train.sort(key=lambda r: (r["style"] != "murattal", r["riwayah"] != 1))
        if not train or not set(ENROLL_SURAHS) <= set(train[0]["surahs"]):
            print(f"  warning: no enrollment recording for {rid}")
            continue
        train[0]["eval_surahs"] = list(ENROLL_SURAHS)

    for rec in recordings:
        if rec["role"] in ("heldout_style", "heldout_recording", "unknown"):
            missing = set(rec["eval_surahs"]) - set(rec["surahs"])
            if missing:
                print(f"  warning: {rec['rec_id']} lacks eval surahs {sorted(missing)}")

    kept = [r for r in recordings if r["role"] is not None]
    used = {r["reciter"] for r in kept}
    for rid in reciters:
        roles = {r["role"] for r in kept if r["reciter"] == rid}
        reciters[rid]["split"] = "unknown" if "unknown" in roles else "train"
    reciters = {k: v for k, v in reciters.items() if k in used}
    return {"built": date.today().isoformat(), "seed": SPLIT_SEED,
            "reciters": reciters, "recordings": kept}


def main():
    cat = build()
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_FILE.write_text(json.dumps(cat, ensure_ascii=False, indent=1), encoding="utf-8")

    recs = cat["recordings"]
    train = [r for r in recs if r["role"] == "train"]
    print(f"Reciters: {len(cat['reciters'])} "
          f"({sum(v['split'] == 'train' for v in cat['reciters'].values())} train, "
          f"{sum(v['split'] == 'unknown' for v in cat['reciters'].values())} unknown)")
    print("Recordings by role:", dict(Counter(r["role"] for r in recs)))
    print("Training recordings by style:", dict(Counter(r["style"] for r in train)))
    per = Counter(r["reciter"] for r in train)
    print("Training recordings per reciter:", dict(sorted(Counter(per.values()).items())))
    mj = sorted({cat["reciters"][r["reciter"]]["name"] for r in train if r["style"] == "mujawwad"})
    print(f"Mujawwad reciters in training ({len(mj)}):", ", ".join(mj))
    for r in recs:
        if r["role"] not in ("train", "unknown"):
            print(f"  {r['role']:18s} {r['rec_id']:14s} {cat['reciters'][r['reciter']]['name']} | {r['label']}")
    print(f"Wrote {CATALOG_FILE}")


if __name__ == "__main__":
    main()
