from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split, TensorDataset

from hh4b_transformer.data.hh4b_loader import load_year
from hh4b_transformer.features.build import build_matrix
from hh4b_transformer.models.transformer import SimpleTransformer

# TODO: REVIEW

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--features", default="configs/features.yaml")
    p.add_argument("--model", default="configs/model.yaml")
    p.add_argument("--train", default="configs/train.yaml")
    p.add_argument("--outdir", default="artifacts/run")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg_data = yaml.safe_load(Path(args.data).read_text())
    cfg_feats = yaml.safe_load(Path(args.features).read_text())
    cfg_model = yaml.safe_load(Path(args.model).read_text())
    cfg_train = yaml.safe_load(Path(args.train).read_text())

    rng = np.random.default_rng(cfg_train.get("seed", 42))

    # Load and stack years
    Xs, ys, ws = [], [], []
    for year in cfg_data["years"]:
        events = load_year(
            data_root=cfg_data["data_root"],
            year=year,
            txbb_version=cfg_data["txbb_version"],
            bdt_version=cfg_data["bdt_version"],
            load_systematics=cfg_data["load_systematics"],
            reorder_txbb=cfg_data["reorder_txbb"],
        )
        for sample, df in events.items():
            X = build_matrix(df, cfg_feats["inputs"])
            y = np.ones(len(df)) if ("hh4b" in sample or "vbfhh4b" in sample) else np.zeros(len(df))
            w = df[cfg_feats.get("weight_key", "finalWeight")].to_numpy().squeeze()
            Xs.append(X); ys.append(y); ws.append(w)

    X = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)

    # dataset and split
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(w).float())
    n = len(ds)
    n_val = int(cfg_data["splits"]["val_fraction"] * n)
    n_test = int(cfg_data["splits"]["test_fraction"] * n)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(cfg_train.get("seed", 42)))

    bs = cfg_train["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)

    model_cfg = cfg_model["model"]
    model = SimpleTransformer(in_dim=X.shape[1], d_model=model_cfg["d_model"], n_layers=model_cfg["n_layers"], dropout=model_cfg["dropout"]).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_train["optimizer"]["lr"], weight_decay=cfg_train["optimizer"].get("weight_decay", 0.0))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    max_steps = cfg_train["train"]["max_steps"]
    log_every = cfg_train["train"].get("log_every", 100)
    step = 0
    while step < max_steps:
        for xb, yb, wb in train_loader:
            opt.zero_grad(set_to_none=True)
            logit = model(xb)
            loss = (bce(logit, yb) * wb).mean()
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                print(f"step {step} loss {loss.item():.4f}")
            if step >= max_steps:
                break

    # save checkpoint
    torch.save(model.state_dict(), outdir / "model.pt")
    (outdir / "meta.json").write_text(json.dumps({
        "features_version": cfg_feats["version"],
        "inputs": cfg_feats["inputs"],
        "data": cfg_data,
        "model": model_cfg,
        "train": cfg_train,
    }, indent=2))


if __name__ == "__main__":
    main()
