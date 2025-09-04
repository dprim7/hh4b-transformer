from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

# TODO: REVIEW

class EventDataset(Dataset):
    def __init__(
        self,
        events_dict: Dict[str, "pd.DataFrame"],
        features: List[str],
        *,
        weight_key: str = "finalWeight",
        sig_keys: List[str] | None = None,
    ) -> None:
        super().__init__()
        self.features = features
        self.weight_key = weight_key
        self.samples = []
        self.X = []
        self.y = []
        self.w = []

        if sig_keys is None:
            sig_keys = ["hh4b", "vbfhh4b"]

        for sample, df in events_dict.items():
            xs = []
            for f in features:
                v = df.get(f)
                if v is None:
                    # support multi-index columns via string keys used in HH4b.utils.get_feat
                    try:
                        v = df[f].to_numpy()
                    except Exception:
                        continue
                arr = v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v)
                if arr.ndim == 1:
                    arr = arr[:, None]
                xs.append(arr)
            if not xs:
                continue
            Xs = np.concatenate(xs, axis=1)
            ys = np.ones(len(df)) if any(k in sample for k in sig_keys) else np.zeros(len(df))
            ws = df[self.weight_key].to_numpy().squeeze()

            self.samples.append(sample)
            self.X.append(Xs)
            self.y.append(ys)
            self.w.append(ws)

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        self.w = np.concatenate(self.w, axis=0)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": torch.from_numpy(self.X[idx]).float(),
            "y": torch.tensor(self.y[idx]).long(),
            "w": torch.tensor(self.w[idx]).float(),
        }
