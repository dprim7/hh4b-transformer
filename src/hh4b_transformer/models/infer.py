from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

# TODO: REVIEW

def predict_torchscript(model_path: str | Path, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    m = torch.jit.load(str(model_path), map_location="cpu").eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).float()
        with torch.no_grad():
            logit = m(xb)
            prob = torch.sigmoid(logit).cpu().numpy()
        outs.append(prob)
    return np.concatenate(outs)


def predict_onnx(model_path: str | Path, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path))
    outs = []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size].astype(np.float32)
        prob = sess.run(None, {"x": xb})[0]
        if prob.ndim == 2 and prob.shape[1] == 1:
            prob = prob[:, 0]
        outs.append(prob)
    return np.concatenate(outs)
