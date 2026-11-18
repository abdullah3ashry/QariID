"""Turns audio clips into voice vectors using a pretrained ECAPA-TDNN model."""
import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from config import MODELS

PRETRAINED = "speechbrain/spkrec-ecapa-voxceleb"


def load_pretrained(device):
    return EncoderClassifier.from_hparams(
        source=PRETRAINED,
        savedir=str(MODELS / PRETRAINED.replace("/", "_")),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,   # Windows: avoid symlinks
    )


class Embedder:
    """weights: optional path to a fine-tuned embedding_model state dict
    (written by finetune.py); None uses the pretrained model."""

    def __init__(self, weights=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_pretrained(self.device)
        self.weights = str(weights) if weights else "pretrained"
        if weights:
            state = torch.load(weights, map_location=self.device, weights_only=True)
            state = state.get("embedding_model", state)
            self.model.mods.embedding_model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def embed(self, clips, batch_size=32):
        """clips: array [N, samples] of equal-length 16 kHz clips.

        Returns L2-normalised vectors [N, 192].
        """
        out = []
        for i in range(0, len(clips), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(clips[i:i + batch_size]))
            emb = self.model.encode_batch(batch.to(self.device)).squeeze(1)
            out.append(torch.nn.functional.normalize(emb, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 192), dtype=np.float32)
