# QariID — progress and handoff

Last updated: 2026-11-18

## Goal

A public mobile app that listens to about 10 seconds of Quran recitation (live,
radio or a recording) and names the reciter. "Shazam for qaris".

## Decisions made

| Topic | Decision |
|---|---|
| Technique | Speaker embeddings (ECAPA-TDNN via SpeechBrain), not audio fingerprinting |
| Matching | Cosine similarity against stored reciter vectors, with an "unknown" threshold |
| App | Flutter, Android + iOS |
| Inference | Server API for v1; model exported to ONNX so on-device/offline is possible later |
| Scope | About 50 reciters in v1 |
| Training | Fine-tune locally on the RTX 5050 Laptop GPU (8 GB). Fine-tuning is high priority |
| Learning from users | No automatic online learning; opt-in feedback reviewed offline |
| Git | Claude stages files only; the owner makes the commits |

Full plan: `C:\Users\user\.claude\plans\so-i-was-thinking-eager-teacup.md`

## Environment

- Project folder: `D:\QariID`
- System Python is 3.15, which has no PyTorch wheels. The project uses a
  uv-managed Python 3.12 venv at `D:\QariID\.venv`.
- torch 2.7.1+cu128, torchaudio 2.7.1, speechbrain 1.0.x, huggingface_hub < 1.0
  (pins are in `requirements.txt`). CUDA works on the RTX 5050.
- `uv` is installed per-user: run it as `python -m uv ...`.
- Git 2.56 is at `C:\Program Files\Git\cmd\git.exe` (on PATH in new terminals).
- ffmpeg is not installed. MP3 decoding and encoding go through `soundfile`.
- Run scripts from `D:\QariID` as `.venv\Scripts\python.exe ml\<script>.py`.

Rebuild the environment from scratch:

```powershell
python -m pip install --user uv
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt --index-strategy unsafe-best-match
```

## What exists

| File | Purpose |
|---|---|
| `ml/config.py` | Reciter lists (20 known, 5 unknown, 3 mujawwad), surah splits, paths |
| `ml/download.py` | Downloads per-ayah MP3s from everyayah.com; re-runnable |
| `ml/audio.py` | Decode to 16 kHz mono, silence removal, clipping, six simulated listening conditions |
| `ml/embedder.py` | `Embedder` class wrapping the pretrained model; `embed(clips)` returns 192-dim unit vectors |
| `ml/enroll.py` | Builds one vector per known reciter -> `data/reciters.npz` |
| `ml/evaluate.py` | Top-1/top-3 accuracy, EER against unknown reciters, cross-recording accuracy |
| `ml/check_multi.py` | Tests enrollment from two recordings against an unseen third |
| `results/*.json` | Saved output of the two evaluations below |

Data on disk (not in git): about 2 GB under `data/`
- `data/raw/<everyayah folder>/<SSSAAA>.mp3` — downloaded audio
- `data/wav16k/<folder>/<SSS>.flac` — decoded, silence-removed surahs (cache)
- `data/models/` — pretrained `speechbrain/spkrec-ecapa-voxceleb`
- `data/reciters.npz` — the 20 enrolled vectors

Data split used so far: enrollment on surahs 36, 55, 67; testing on surahs 18, 50.

## Results so far

### Phase 1: pretrained model, no training, 20 reciters

`results/phase1_20261117_022239.json` — 1,200 test clips per row.

| Condition (10 s clips) | Top-1 | Top-3 | EER vs. unknown reciters |
|---|---|---|---|
| Clean | 99.8% | 100% | 3.0% |
| Noise, 10 dB SNR | 99.3% | 100% | 6.0% |
| Reverb | 99.5% | 99.9% | 4.7% |
| Narrow band | 98.8% | 100% | 3.7% |
| Low-bitrate MP3 | 99.2% | 100% | 4.1% |
| Phone across a room | 97.8% | 99.8% | 9.7% |

- 5 s clips: 99.5% clean, 95.8% phone across a room. The app should record about 10 s.
- EER thresholds ranged from 0.50 to 0.63 depending on condition.
- Tested on the mujawwad recording of an enrolled reciter (10 s, clean):
  Al-Husary 100%, Abdul Basit 40%, Al-Minshawy 38%.

### Multi-recording enrollment check

`results/multi_enroll_20261117_023441.json` — 60 clips per reciter, tested on a
third recording never used for enrollment.

| Condition | Enrollment | Abdul Basit | Al-Husary | Al-Minshawy |
|---|---|---|---|---|
| Clean | Murattal only | 98% | 100% | 75% |
| Clean | Murattal + mujawwad averaged | 98% | 100% | 48% |
| Clean | Murattal + mujawwad separate | 98% | 100% | 65% |
| Phone across a room | Murattal only | 98% | 100% | 87% |
| Phone across a room | Murattal + mujawwad averaged | 48% | 100% | 88% |
| Phone across a room | Murattal + mujawwad separate | 98% | 100% | 85% |

### Conclusions

1. The pretrained model is already strong on murattal, including on unseen recordings.
2. Mujawwad recitation is the weak spot; the same reciter's mujawwad and murattal
   vectors sit far apart.
3. Adding a second recording at enrollment does not fix this, and averaging styles
   into one vector hurts. If a reciter needs several vectors, keep them separate.
4. Fine-tuning on data that contains both styles for the same reciters is the fix to try.

### Caveats

- All degradations are simulated; nothing has been tested with a real phone yet.
- Only five non-enrolled reciters were used for the "unknown" test.
- The cross-recording tests cover three reciters only, and 60-clip results swing
  by several points from noise.
- The Minshawy "third recording" (`Menshawi_32kbps`) may be the same murattal
  recording at a lower bitrate (median per-ayah duration difference 0.2 s).
- Not yet measured: whether separate-vector enrollment fixes mujawwad clips
  themselves on held-out surahs.
- Corrupt downloads skipped at decode time: `Muhammad_Ayyoub_128kbps/036005.mp3`,
  `Hani_Rifai_192kbps/018077.mp3`, `Yasser_Ad-Dussary_128kbps/018098.mp3`.
- Missing on the server: 4 files in `Menshawi_32kbps`, 5 in `warsh/warsh_Abdul_Basit_128kbps`.

## Phase 2 status (2026-11-18)

Built:

| File | Purpose |
|---|---|
| `ml/catalog.py` | Queries mp3quran.net, picks recordings and assigns roles -> `data/train/catalog.json` |
| `ml/fetch_train.py` | Downloads whole surahs, decodes to 16 kHz FLAC, removes silence, deletes MP3s; writes `data/train/manifest.csv` |
| `ml/augment.py` | Random augmentation chain (speed ±5%, reverb, noise 0–20 dB, band-limit, MP3 codec, gain) |
| `ml/finetune.py` | AAM-softmax fine-tuning with validation each epoch; checkpoints in `data/checkpoints/<run>/` |
| `ml/evaluate.py` | Now takes `--model`; builds enrollment with the model under test; adds style / recording / 30-unknown tests |
| `ml/embedder.py` | `Embedder(weights=...)` loads a fine-tuned checkpoint |

Dataset design (`ml/config.py`, Phase 2 section):
- Source: mp3quran.net (244 reciters) plus one quranicaudio.com mujawwad reciter (Al-Dulaimi).
  Teaching editions ("Mo'lim") are excluded because other voices repeat after the reciter.
- 219 reciters: 189 for training and 30 fully held out as "unknown". The 30 are the
  Phase 1 five plus 25 picked at random (seed 1234).
- 210 training recordings: up to 3 per reciter, one per style first so a reciter
  with mujawwad also keeps murattal. Budget per recording is 60 min of speech,
  or 120 min for mujawwad.
- Never trained on: surahs 18, 36, 50, 55, 67 (all evaluation surahs) for any
  reciter, and surahs 2–9 (too long).
- Held-out test recordings: mujawwad of Mustafa Ismail and Mahmoud Ali Al-Banna
  (their murattal is trained on), and a non-Hafs recording of Husary, Al-Hudhaify,
  Abdul Basit, Ibrahim Al-Dosari and Al-Tarabulsi. Their enrollment audio (surahs
  36/55/67 of the trained recording) and test audio (18/50) go to `data/eval/`.
- Mujawwad is scarce: only 7 reciters have mujawwad in training (Abdul Basit,
  Husary, Minshawi, Tablawi, Muaiqly, Saber Abdulhakam, Al-Dulaimi). quranicaudio
  adds almost nothing new. More mujawwad (Egyptian qaris on archive.org etc.)
  would be the next data improvement.
- Checked: the prepared mp3quran Husary audio matches the Phase 1 Husary vector
  (40/40 clips, similarity about 0.8).

Run order:

```powershell
.venv\Scripts\python.exe ml\catalog.py        # only to rebuild the catalog
.venv\Scripts\python.exe ml\fetch_train.py    # re-runnable; about 2 h, about 15 GB
.venv\Scripts\python.exe ml\finetune.py --val-only   # pretrained baseline on validation
.venv\Scripts\python.exe ml\finetune.py --run ft1
.venv\Scripts\python.exe ml\evaluate.py               # pretrained
.venv\Scripts\python.exe ml\evaluate.py --model data\checkpoints\ft1\best.pt
```

Notes:
- Real RIRs and noises (OpenSLR 28, `data/aug/RIRS_NOISES`) are optional;
  without them `augment.py` uses synthetic reverb and noise. The first download
  attempt stalled because the connection was shared with the dataset fetch.
- mpg123 prints "part2_3_length too large" for some mp3quran files: a few bad
  frames are skipped; harmless.
- During the top-only warm-up, frozen layers are kept in eval mode; otherwise
  their BatchNorm statistics drift towards Quran audio and shift every embedding.

## Phase 2 plan (original)

1. **Build the training dataset.**
   - Source: mp3quran.net API (hundreds of reciters, several recordings/riwayat
     for many of them), plus everyayah.com.
   - Target: 150–200 reciters, 1–2 hours each, so the model does not overfit to
     the 50 that will be enrolled.
   - Prefer reciters with more than one recording or style; label every file with
     reciter, recording and style (murattal / mujawwad / other).
   - mp3quran files are whole surahs, not ayahs: the loader needs a path for that.
   - Check each source's terms of use.
2. **Define the splits.**
   - Held-out surahs for every reciter.
   - Some reciters fully held out for the "unknown" test (more than five).
   - Some recordings held out for cross-recording and cross-style tests.
3. **Write `ml/finetune.py`.**
   - Start from the pretrained ECAPA weights, AAM-softmax loss.
   - 3 s random crops, batch about 32, mixed precision, low learning rate, early
     layers frozen at first.
   - On-the-fly augmentation: noise (MUSAN), room impulse responses, speed
     perturbation, band-limiting and codec simulation.
4. **Extend `ml/evaluate.py`** to accept a model path, so the pretrained and
   fine-tuned models are scored on identical test sets, including the mujawwad
   and third-recording sets.
5. **Accept the fine-tuned model only if** it beats the pretrained baseline on
   mujawwad, degraded and unknown-reciter tests without losing murattal accuracy.
6. **Enroll the final 50 reciters** and fix the "unknown" threshold.
7. **Write `ml/export.py`**: export to ONNX, verify the outputs match PyTorch,
   and produce an int8-quantized copy for a future offline build.

Then: Phase 3 server (FastAPI + ONNX Runtime, `/identify`, `/reciters`,
`/feedback`), Phase 4 Flutter app, Phase 5 release.

## Open items for the owner

- Make the first commit. Everything is staged on branch `main`; nothing is committed.
- The global git identity on this machine is "Abdullah Abdelsamea Mohamed"
  (`s-abdullah.abdelsamiaa@zewailcity.edu.eg`). Change it before committing if
  that is not the intended author.
- Create the GitHub repository, then `git remote add origin <url>` and
  `git push -u origin main`.
- Choose the 50 reciters for v1 (or approve a proposed list).
- Terms of use: neither mp3quran.net nor everyayah.com publishes terms for
  their audio. Training a model on it for a public app should be cleared with
  them (both have contact pages) before release; the app must not redistribute
  the audio itself.
- A real-phone test set (recitations played through a speaker and recorded on a
  phone) would replace the simulated conditions; it needs someone to record it.

## How to resume with Claude

Open Claude Code in `D:\QariID` and say: "Read PROGRESS.md and continue with Phase 2."
