from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from hh4b_transformer.models.transformer import SimpleTransformer

# TODO: REVIEW

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--x", required=True, help=".npy input features")
    p.add_argument("--y", required=True, help=".npy labels")
    args = p.parse_args()

    X = np.load(args.x)
    y = np.load(args.y)

    model = SimpleTransformer(in_dim=X.shape[1])
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X).float()).numpy()
        prob = 1 / (1 + np.exp(-logits))

    auc = roc_auc_score(y, prob)
    print(f"ROC AUC: {auc:.4f}")
    Path("outputs").mkdir(exist_ok=True)
    np.save("outputs/prob.npy", prob)


if __name__ == "__main__":
    main()
