from __future__ import annotations

from typing import List, Dict

import numpy as np
from HH4b.utils import get_feat

# TODO: REVIEW

def build_matrix(df, features: List[str]) -> np.ndarray:
    cols = []
    for f in features:
        v = get_feat(df, f) if f not in df else df[f].to_numpy()
        if v is None:
            continue
        arr = v
        if arr.ndim == 1:
            arr = arr[:, None]
        cols.append(arr)
    if not cols:
        return np.empty((len(df), 0), dtype=np.float32)
    return np.concatenate(cols, axis=1)
