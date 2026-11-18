# QariID

Identify a Quran reciter from a few seconds of audio — "Shazam for qaris".

A speaker-embedding model (ECAPA-TDNN) turns a clip into a voice vector; each
reciter is stored as one or more vectors; a clip is matched by cosine similarity,
with an "unknown" answer when nothing is close enough.

## Status

Phase 1 (proof of concept with a pretrained model, 20 reciters) is done.
Results are in `results/`.

## Setup (Windows)

PyTorch has no wheels for Python 3.15 yet, so the project uses its own Python 3.12:

```powershell
python -m pip install --user uv
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt --index-strategy unsafe-best-match
```

## Running

```powershell
.venv\Scripts\python.exe ml\download.py      # ~1.6 GB of audio from everyayah.com into data\
.venv\Scripts\python.exe ml\enroll.py        # build one voice vector per reciter
.venv\Scripts\python.exe ml\evaluate.py      # accuracy under clean and degraded conditions
.venv\Scripts\python.exe ml\check_multi.py   # does enrolling from two recordings help on a third?
```

## Layout

| Path | Contents |
|---|---|
| `ml/config.py` | Reciter list, surah splits, paths |
| `ml/audio.py` | Decoding, silence removal, clipping, simulated degradations |
| `ml/embedder.py` | Wrapper around the pretrained ECAPA-TDNN model |
| `ml/download.py`, `enroll.py`, `evaluate.py`, `check_multi.py` | Pipeline scripts |
| `results/` | Evaluation output (JSON) |
| `data/` | Audio, caches and model files (not in git) |

## Roadmap

1. Proof of concept (done)
2. Fine-tuning on Quran recitation with multiple recordings per reciter
3. Inference server (FastAPI + ONNX Runtime)
4. Flutter app (Android + iOS)
5. Release

Audio is used for research only and is never redistributed; the app ships voice
vectors, not recordings.
