"""Measure how well a model identifies reciters.

Usage:  python ml\\evaluate.py                         (pretrained model)
        python ml\\evaluate.py --model data\\checkpoints\\<run>\\best.pt
        python ml\\evaluate.py --seconds 10 --conditions clean phone_far

Enrollment vectors are built with the model under test, so every model is
scored the same way. Gallery: the 20 Phase 1 reciters (everyayah, surahs
36/55/67) plus the reciters that have held-out recordings (mp3quran, same surahs).

Reports, per clip length and simulated listening condition:
  top-1 / top-3  enrolled reciters, unseen surahs (18, 50) of the enrolled recording
  EER5 / EER30   enrolled vs non-enrolled: the 5 Phase 1 strangers / 30 held-out reciters
  cross          everyayah mujawwad recording of an enrolled (murattal) reciter
  style          held-out mujawwad of reciters never trained on in that style
  recording      held-out recording (other riwayah) of an enrolled reciter
Writes results\\eval_<tag>_<timestamp>.json.
"""
import argparse
import json
import time

import numpy as np
import soundfile as sf

from audio import CONDITIONS, degrade, load_surahs, split_clips
from config import (CATALOG_FILE, CROSS, CROSS_SURAHS, ENROLL_SURAHS, EVAL,
                    KNOWN, MP3QURAN_ID, RESULTS, TEST_SURAHS, UNKNOWN)

CLIP_SECONDS = (5, 10)
MAX_CLIPS = 60      # per reciter, per clip length
UNKNOWN30_CLIPS = 20


def eer(genuine, impostor):
    """Equal error rate and its threshold for 'is this an enrolled reciter?'"""
    if len(genuine) == 0 or len(impostor) == 0:
        return None, None
    genuine, impostor = np.sort(genuine), np.sort(impostor)
    t = np.unique(np.concatenate([genuine, impostor]))
    frr = np.searchsorted(genuine, t, side="left") / len(genuine)          # genuine < t: rejected
    far = 1 - np.searchsorted(impostor, t, side="left") / len(impostor)    # impostor >= t: accepted
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(t[i])


def load_eval(rec_id, surahs):
    parts = []
    for s in surahs:
        f = EVAL / rec_id / f"{s:03d}.flac"
        if f.exists():
            parts.append(sf.read(str(f), dtype="float32")[0])
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def phase2_sets():
    """Gallery additions and test audio from the Phase 2 catalog (if built)."""
    if not CATALOG_FILE.exists():
        return {}, {}, {}, {}
    cat = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    to_known = {f"mq{v}": k for k, v in MP3QURAN_ID.items() if k in KNOWN}
    enroll, style, recording, unknown = {}, {}, {}, {}
    for rec in cat["recordings"]:
        rid = rec["reciter"]
        gid = to_known.get(rid, rid)
        if rec["role"] == "train" and rec["eval_surahs"] == list(ENROLL_SURAHS) and gid not in KNOWN:
            enroll[gid] = load_eval(rec["rec_id"], ENROLL_SURAHS)
        elif rec["role"] == "heldout_style":
            style[gid] = load_eval(rec["rec_id"], TEST_SURAHS)
        elif rec["role"] == "heldout_recording":
            recording[gid] = load_eval(rec["rec_id"], TEST_SURAHS)
        elif rec["role"] == "unknown":
            unknown[rid] = load_eval(rec["rec_id"], TEST_SURAHS)
    drop = lambda d: {k: v for k, v in d.items() if len(v)}
    return drop(enroll), drop(style), drop(recording), drop(unknown)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="fine-tuned weights (.pt); default: pretrained")
    ap.add_argument("--tag", help="name used in the results file")
    ap.add_argument("--seconds", type=int, nargs="*", default=list(CLIP_SECONDS))
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    args = ap.parse_args()

    from embedder import Embedder        # heavy import; keep eer() importable cheaply
    embedder = Embedder(weights=args.model)
    tag = args.tag or ("pretrained" if not args.model else
                       "_".join(p for p in __import__("pathlib").Path(args.model).with_suffix("").parts[-2:]))

    extra_enroll, style_wav, rec_wav, unk30_wav = phase2_sets()
    enroll_wav = {r: load_surahs(f, ENROLL_SURAHS) for r, (_, f) in KNOWN.items()}
    enroll_wav.update(extra_enroll)
    ids = list(enroll_wav)
    centroids = np.stack([embedder.embed(split_clips(w, 10)).mean(0) for w in enroll_wav.values()])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    print(f"Model: {embedder.weights} on {embedder.device}; gallery of {len(ids)} reciters "
          f"({len(extra_enroll)} from Phase 2 catalog)\n")

    known_wav = {r: load_surahs(KNOWN[r][1], TEST_SURAHS) for r in KNOWN}
    unk5_wav = {r: load_surahs(f, TEST_SURAHS) for r, (_, f) in UNKNOWN.items()}
    cross_wav = {r: load_surahs(f, CROSS_SURAHS) for r, f in CROSS.items()}

    def gather(wavs, seconds, cap=MAX_CLIPS):
        clips, labels = [], []
        for r, wav in wavs.items():
            c = split_clips(wav, seconds, cap)
            if len(c):
                clips.append(c)
                labels += [r] * len(c)
        if not clips:
            return np.zeros((0, seconds * 16000), dtype=np.float32), np.array([])
        return np.concatenate(clips), np.array(labels)

    def sims(clips, cond):
        if len(clips) == 0:
            return np.zeros((0, len(ids)))
        return embedder.embed(degrade(clips, cond)) @ centroids.T

    def top1_of(sim, labels):
        if len(sim) == 0:
            return None, {}
        truth = np.array([ids.index(r) for r in labels])
        hit = sim.argmax(1) == truth
        return float(hit.mean()), {r: float(hit[labels == r].mean()) for r in dict.fromkeys(labels)}

    rows, details = [], {}
    for seconds in args.seconds:
        sets = {
            "known": gather(known_wav, seconds),
            "unk5": gather(unk5_wav, seconds),
            "unk30": gather(unk30_wav, seconds, UNKNOWN30_CLIPS),
            "cross": gather(cross_wav, seconds),
            "style": gather(style_wav, seconds),
            "recording": gather(rec_wav, seconds),
        }
        for cond in args.conditions:
            try:
                S = {k: sims(c, cond) for k, (c, _) in sets.items()}
            except Exception as e:
                print(f"{seconds}s {cond}: skipped ({e})")
                continue
            k_lab = sets["known"][1]
            truth = np.array([ids.index(r) for r in k_lab])
            order = np.argsort(-S["known"], axis=1)
            top1 = float((order[:, 0] == truth).mean())
            top3 = float((order[:, :3] == truth[:, None]).any(axis=1).mean())
            genuine = S["known"].max(axis=1)
            eer5, thr5 = eer(genuine, S["unk5"].max(axis=1)) if len(S["unk5"]) else (None, None)
            eer30, thr30 = eer(genuine, S["unk30"].max(axis=1)) if len(S["unk30"]) else (None, None)
            cross, per_cross = top1_of(S["cross"], sets["cross"][1])
            style, per_style = top1_of(S["style"], sets["style"][1])
            recording, per_rec = top1_of(S["recording"], sets["recording"][1])
            rows.append(dict(seconds=seconds, condition=cond, clips=len(k_lab), top1=top1,
                             top3=top3, eer5=eer5, thr5=thr5, eer30=eer30, thr30=thr30,
                             cross_top1=cross, style_top1=style, recording_top1=recording,
                             counts={k: len(v[1]) for k, v in sets.items()}))
            _, per_known = top1_of(S["known"], k_lab)
            confusions = {}
            for t, p in zip(truth, order[:, 0]):
                if t != p:
                    key = f"{ids[t]} -> {ids[p]}"
                    confusions[key] = confusions.get(key, 0) + 1
            details[f"{seconds}s/{cond}"] = dict(
                per_reciter=per_known, per_cross=per_cross, per_style=per_style,
                per_recording=per_rec,
                confusions=dict(sorted(confusions.items(), key=lambda kv: -kv[1])[:10]))

    def pct(v):
        return "   n/a" if v is None else f"{100 * v:5.1f}%"

    print(f"{'clip':>4} {'condition':<12} {'top-1':>6} {'top-3':>6} {'EER5':>6} {'EER30':>6} "
          f"{'thr30':>5} {'cross':>6} {'style':>6} {'recrd':>6}")
    for r in rows:
        thr = "  n/a" if r["thr30"] is None else f"{r['thr30']:.2f}"
        print(f"{r['seconds']:>3}s {r['condition']:<12} {pct(r['top1'])} {pct(r['top3'])} "
              f"{pct(r['eer5'])} {pct(r['eer30'])} {thr:>5} {pct(r['cross_top1'])} "
              f"{pct(r['style_top1'])} {pct(r['recording_top1'])}")
    if rows:
        print("clips per set:", rows[0]["counts"])

    for key in ("10s/clean", "10s/phone_far"):
        if key not in details:
            continue
        d = details[key]
        weakest = sorted(d["per_reciter"].items(), key=lambda kv: kv[1])[:5]
        print(f"\n[{key}] weakest reciters: " + ", ".join(f"{r} {100 * a:.0f}%" for r, a in weakest))
        if d["confusions"]:
            print(f"[{key}] top confusions: "
                  + ", ".join(f"{k} x{n}" for k, n in list(d["confusions"].items())[:5]))
        for name in ("per_cross", "per_style", "per_recording"):
            if d[name]:
                print(f"[{key}] {name[4:]}: "
                      + ", ".join(f"{r} {100 * a:.0f}%" for r, a in d[name].items()))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"eval_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(dict(model=embedder.weights, gallery=ids, summary=rows,
                                   details=details), indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
