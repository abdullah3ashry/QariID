"""Fine-tune the pretrained ECAPA-TDNN on Quran recitation (Phase 2).

Usage:  python ml\\finetune.py [--epochs 10] [--steps 1000] [--run NAME]
        python ml\\finetune.py --val-only          (score the pretrained model)

Data: data/train/manifest.csv (from fetch_train.py). One surah file per
recording is held out for validation; everything else is training data.

Training: balanced sampling (reciter uniform, then recording uniform, then file
by duration), random 3 s crops with random augmentation, AAM-softmax head
initialised from pretrained per-reciter centroids, bf16 autocast. For the first
--warmup-epochs only the top of the network (pooling + final layer) is trained.

Validation (each epoch, phone-far degradation, 10 s clips):
  same_top1   identify a reciter from another surah of the same recording
  cross_top1  identify a reciter from a *different* recording (other style or
              riwayah) than the one their gallery vector was built from
  mujawwad    cross_top1 restricted to mujawwad clips (murattal gallery)
  eer         verification EER over all clip-vs-gallery pairs
Checkpoints: data/checkpoints/<run>/epoch_XX.pt and best.pt (embedding_model
weights only; load with Embedder(weights=...)). History in history.json.
"""
import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
from speechbrain.lobes.models.ECAPA_TDNN import Classifier
from speechbrain.nnet.losses import AdditiveAngularMargin, LogSoftmaxWrapper
from torch.utils.data import DataLoader, Dataset

from audio import phone_far, split_clips
from augment import Augmenter
from config import DATA, MANIFEST_FILE, SAMPLE_RATE, SPLIT_SEED, TRAIN
from embedder import load_pretrained
from evaluate import eer

CKPT = DATA / "checkpoints"
VAL_CLIP_S = 10
VAL_CLIPS = 6          # per half of each validation file


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_manifest():
    with open(MANIFEST_FILE, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["seconds"] = float(r["seconds"])
    return rows


def split_val(rows, seed=SPLIT_SEED):
    """Hold out one file (>= 2 min) per recording that has at least 4 files."""
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["rec_id"]].append(r)
    rng = random.Random(seed)
    train, val = [], []
    for _, rs in sorted(by_rec.items()):
        rs = sorted(rs, key=lambda r: r["path"])
        cands = [r for r in rs if r["seconds"] >= 2 * VAL_CLIPS * VAL_CLIP_S]
        if len(rs) >= 4 and cands:
            v = rng.choice(cands)
            val.append(v)
            train += [r for r in rs if r is not v]
        else:
            train += rs
    return train, val


class CropDataset(Dataset):
    """Each item is an independent random crop; the index only seeds the RNG."""

    def __init__(self, rows, label_of, n_items, seconds, augment):
        self.n_items, self.n = n_items, int(seconds * SAMPLE_RATE)
        self.label_of, self.augment, self.epoch = label_of, augment, 0
        tree = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["seconds"] * SAMPLE_RATE > self.n + 1:
                tree[r["reciter"]][r["rec_id"]].append(r)
        self.reciters = sorted(tree)
        self.tree = {k: [(sorted(v[rec], key=lambda r: r["path"])) for rec in sorted(v)]
                     for k, v in tree.items()}

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        rng = np.random.default_rng((SPLIT_SEED, self.epoch, i))
        reciter = self.reciters[rng.integers(len(self.reciters))]
        recs = self.tree[reciter]
        files = recs[rng.integers(len(recs))]
        w = np.array([f["seconds"] for f in files])
        f = files[rng.choice(len(files), p=w / w.sum())]
        total = int(f["seconds"] * SAMPLE_RATE)
        start = int(rng.integers(0, max(1, total - self.n)))
        x, _ = sf.read(str(TRAIN / f["path"]), start=start, frames=self.n, dtype="float32")
        if len(x) < self.n:
            x = np.pad(x, (0, self.n - len(x)))
        if self.augment is not None:
            x = self.augment(x, rng)
        return torch.from_numpy(x), self.label_of[reciter]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(torch.nn.Module):
    def __init__(self, sb, n_classes):
        super().__init__()
        self.features = sb.mods.compute_features
        self.norm = sb.mods.mean_var_norm
        self.embedding_model = sb.mods.embedding_model
        self.head = Classifier(input_size=192, out_neurons=n_classes)

    def embed(self, wav):
        with torch.no_grad():
            f = self.features(wav)
            f = self.norm(f, torch.ones(wav.shape[0], device=wav.device))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=wav.is_cuda):
            e = self.embedding_model(f)
        return e.float()                         # [B, 1, 192]

    def forward(self, wav):
        return self.head(self.embed(wav))


@torch.no_grad()
def embed_clips(model, clips, device, batch=32):
    model.eval()
    out = []
    for i in range(0, len(clips), batch):
        x = torch.from_numpy(np.ascontiguousarray(clips[i:i + batch])).to(device)
        out.append(torch.nn.functional.normalize(model.embed(x).squeeze(1), dim=1).cpu().numpy())
    return np.concatenate(out)


TOP = ("mfa", "asp", "asp_bn", "fc")


def set_trainable(model, top_only):
    for name, p in model.embedding_model.named_parameters():
        p.requires_grad = (not top_only) or name.split(".")[0] in TOP


def train_mode(model, top_only):
    """model.train(), but frozen layers stay in eval mode so their BatchNorm
    running statistics do not drift during the top-only warm-up."""
    model.train()
    if top_only:
        for name, child in model.embedding_model.named_children():
            if name not in TOP:
                child.eval()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def build_val(val_rows):
    """Gallery from the 'primary' recording of each reciter; tests from both
    halves of the held-out files. Returns dict of clip arrays + labels."""
    by_reciter = defaultdict(list)
    for r in val_rows:
        by_reciter[r["reciter"]].append(r)
    rng = np.random.default_rng(0)
    enroll, enroll_lab, same, same_lab, cross, cross_lab, cross_mj = [], [], [], [], [], [], []
    for reciter, rs in sorted(by_reciter.items()):
        # primary = murattal Hafs if available, else first
        rs = sorted(rs, key=lambda r: (r["style"] != "murattal", r["riwayah"] != "1", r["rec_id"]))
        for k, r in enumerate(rs):
            wav = sf.read(str(TRAIN / r["path"]), dtype="float32")[0]
            clips = split_clips(wav, VAL_CLIP_S)
            half = len(clips) // 2
            first, second = clips[:half][:VAL_CLIPS], clips[half:][:VAL_CLIPS]
            if k == 0:
                enroll.append(first)
                enroll_lab += [reciter] * len(first)
                same.append(second)
                same_lab += [reciter] * len(second)
            else:
                cross.append(second)
                cross_lab += [reciter] * len(second)
                cross_mj += [r["style"] == "mujawwad"] * len(second)

    def deg(arrs):
        if not arrs:
            return np.zeros((0, VAL_CLIP_S * SAMPLE_RATE), dtype=np.float32)
        return np.stack([phone_far(c, rng) for c in np.concatenate(arrs)]).astype(np.float32)

    return dict(enroll=np.concatenate(enroll), enroll_lab=np.array(enroll_lab),
                same=deg(same), same_lab=np.array(same_lab),
                cross=deg(cross), cross_lab=np.array(cross_lab), cross_mj=np.array(cross_mj))


def validate(model, val, device):
    gallery_ids = sorted(set(val["enroll_lab"]))
    e = embed_clips(model, val["enroll"], device)
    G = np.stack([e[val["enroll_lab"] == g].mean(0) for g in gallery_ids])
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    idx = {g: i for i, g in enumerate(gallery_ids)}

    def score(clips, labels):
        if len(clips) == 0:
            return None, None, None
        s = embed_clips(model, clips, device) @ G.T
        truth = np.array([idx[l] for l in labels])
        top1 = (s.argmax(1) == truth)
        own = np.zeros_like(s, dtype=bool)
        own[np.arange(len(s)), truth] = True
        return top1, s[own], s[~own]

    same_top1, g1, i1 = score(val["same"], val["same_lab"])
    out = {"same_top1": float(same_top1.mean())}
    gen, imp = [g1], [i1]
    if len(val["cross"]):
        cross_top1, g2, i2 = score(val["cross"], val["cross_lab"])
        out["cross_top1"] = float(cross_top1.mean())
        mj = val["cross_mj"]
        out["mujawwad_top1"] = float(cross_top1[mj].mean()) if mj.any() else None
        gen.append(g2)
        imp.append(i2)
    rate, thr = eer(np.concatenate(gen), np.concatenate(imp))
    out.update(eer=rate, threshold=thr)
    parts = [out["same_top1"], out.get("cross_top1") or 0, out.get("mujawwad_top1") or 0, 1 - rate]
    out["score"] = float(np.mean(parts))
    return out


def fmt(m):
    def p(k):
        v = m.get(k)
        return "n/a" if v is None else f"{100 * v:.1f}%"
    return (f"same {p('same_top1')} | cross {p('cross_top1')} | mujawwad {p('mujawwad_top1')} "
            f"| EER {p('eer')} | score {m['score']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def init_head(model, ds, label_of, device, crops=16):
    """Head weights = mean pretrained embedding per reciter (clean crops)."""
    W = torch.zeros(len(label_of), 192)
    aug, ds.augment = ds.augment, None
    for reciter, lab in label_of.items():
        xs = []
        for k in range(crops):
            recs = ds.tree[reciter]
            files = recs[k % len(recs)]
            f = files[k % len(files)]
            total = int(f["seconds"] * SAMPLE_RATE)
            start = (k * 7919 * SAMPLE_RATE) % max(1, total - ds.n)
            x, _ = sf.read(str(TRAIN / f["path"]), start=start, frames=ds.n, dtype="float32")
            xs.append(np.pad(x, (0, ds.n - len(x))))
        e = embed_clips(model, np.stack(xs), device)
        W[lab] = torch.from_numpy(e.mean(0))
    ds.augment = aug
    model.head.weight.data.copy_(torch.nn.functional.normalize(W, dim=1).to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=time.strftime("ft_%Y%m%d_%H%M"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1000, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-5, help="backbone learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--aug", type=float, default=1.0, help="augmentation strength (0 = off)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SPLIT_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_manifest()
    train_rows, val_rows = split_val(rows)
    reciters = sorted({r["reciter"] for r in train_rows})
    label_of = {r: i for i, r in enumerate(reciters)}
    hours = sum(r["seconds"] for r in train_rows) / 3600
    print(f"Train: {len(train_rows)} files, {hours:.1f} h, {len(reciters)} reciters | "
          f"val: {len(val_rows)} files")

    sb = load_pretrained(device)
    model = Model(sb, len(reciters)).to(device)

    t = time.time()
    val = build_val(val_rows)
    print(f"Validation: {len(val['enroll_lab'])} enroll, {len(val['same_lab'])} same, "
          f"{len(val['cross_lab'])} cross ({int(val['cross_mj'].sum())} mujawwad) clips "
          f"[{time.time() - t:.0f}s]")
    base = validate(model, val, device)
    print(f"epoch 0 (pretrained): {fmt(base)}")
    if args.val_only:
        return

    augment = Augmenter(args.aug) if args.aug > 0 else None
    if augment:
        print("Augmentation:", augment.describe())
    ds = CropDataset(train_rows, label_of, args.steps * args.batch, args.crop, augment)
    init_head(model, ds, label_of, device)

    criterion = LogSoftmaxWrapper(AdditiveAngularMargin(margin=args.margin, scale=args.scale))
    opt = torch.optim.AdamW([
        {"params": model.embedding_model.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=2e-5)
    base_lrs = [g["lr"] for g in opt.param_groups]
    total, warm = args.epochs * args.steps, min(500, args.steps)

    run_dir = CKPT / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
    history = [{"epoch": 0, **base}]
    best = base["score"]
    step = 0
    for epoch in range(1, args.epochs + 1):
        top_only = epoch <= args.warmup_epochs
        set_trainable(model, top_only)
        ds.epoch = epoch
        dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
        train_mode(model, top_only)
        t, losses, correct, seen = time.time(), [], 0, 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device)
            lr_f = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_f
            out = model(x)
            loss = criterion(out, y.unsqueeze(1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
            correct += (out.squeeze(1).argmax(1) == y).sum().item()
            seen += len(y)
            step += 1
            if step % 200 == 0:
                print(f"  step {step}: loss {np.mean(losses[-200:]):.3f}, "
                      f"train acc {100 * correct / seen:.1f}%, "
                      f"{seen / (time.time() - t):.0f} crops/s", flush=True)
        m = validate(model, val, device)
        m.update(epoch=epoch, loss=float(np.mean(losses)), train_acc=correct / seen,
                 minutes=(time.time() - t) / 60)
        history.append(m)
        torch.save({"embedding_model": model.embedding_model.state_dict()},
                   run_dir / f"epoch_{epoch:02d}.pt")
        tag = ""
        if m["score"] > best:
            best = m["score"]
            torch.save({"embedding_model": model.embedding_model.state_dict()}, run_dir / "best.pt")
            tag = "  * best"
        print(f"epoch {epoch}: loss {m['loss']:.3f} | {fmt(m)} [{m['minutes']:.1f} min]{tag}",
              flush=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    print(f"Done. Best score {best:.4f} (pretrained {base['score']:.4f}). Checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
