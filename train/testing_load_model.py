import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA2VEC_DIR = _REPO_ROOT / "pretraining" / "data2vec"
sys.path.insert(0, str(_DATA2VEC_DIR))

from genome_encoder.genome_bert import mae_genome_base  # noqa: E402


def load_pretrained_weights(model, pretrained_weights: str) -> None:
    if not os.path.isfile(pretrained_weights):
        raise FileNotFoundError(pretrained_weights)
    checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {pretrained_weights} (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")


if __name__ == "__main__":
    ckpt = os.environ.get("PRETRAINED_CHECKPOINT", "")
    if not ckpt:
        raise SystemExit("Set PRETRAINED_CHECKPOINT=/path/to/checkpoint.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mae_genome_base(seq_length=6400000, patch_size=768).to(device)
    load_pretrained_weights(model, ckpt)
