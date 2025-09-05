# hh4b-transformer

A lightweight, experiment-focused repository for training transformer-based models for the HH→4b analysis. This repo is intentionally separate from the main `HH4b` analysis to keep ML dependencies, experiments, and artifacts isolated, while reusing `HH4b` for standard I/O and physics conventions.

## Goals
- Clean training environment (Torch/JAX/etc.) without impacting `HH4b` users.
- Reuse `HH4b` data loading/normalization for consistency (weights, selections, naming).
- Export CPU-friendly models (TorchScript/ONNX) for downstream inference in `HH4b`.
- Track features and model metadata for reproducibility.

Details about Data, Architecure, Features etc. can be found in [PLAN.md](docs/PLAN.md)


# Development

## Virtual Environment for Development

Create and activate virtual environment, install repo as package, install pre-commit.
First time install, in order:
```
micromamba create -n hh4b-part -f environment.yml -y
micromamba activate hh4b-part
python -m pip install -e .
pre-commit install 
```
Activate the micromamba environment in each new session.

## Data expectations
- Data lives outside this repo. Expected structure (as produced by `HH4b`):
```
<data_root>/<year>/<sample>/{parquet,pickles}/...
```
- Example path:
```
/ceph/cms/store/user/dprimosc/bbbb/skimmer/24Sep25_v12v2_private_signal/2023BPix/JetMET_Run2023D/parquet/out_55.parquet
```
or on pvc:
```
```


## Design choices (reasoning)
- Separate repo: isolates ML deps, accelerates iteration, simplifies CI.
- Use `HH4b` as a library: ensures consistent weights, selections, and sample naming.
- Config-first: all paths, features, and hparams are YAML-driven for reproducibility.
- Minimal code surface: small, testable modules (loader, features, model, export).

## Plan
1. MVP: load normalized events via `HH4b`, build features, train baseline transformer.
2. Export compact CPU inference artifact + metadata (feature version, data tag, `HH4b` commit).
3. Add a tiny inference hook in `HH4b` to annotate events with the new score.
4. Iterate on features/architecture; keep changes isolated here.

## Getting started
- Create an environment and install dependencies.
- Install `HH4b` in the same environment (editable or pinned commit) so imports work.
- Fill out `configs/data.yaml` and `configs/features.yaml` to match your ntuples.
- Run `src/hh4b_transformer/train.py` (MVP CLI is minimal) and iterate.

## Notes
- Keep large artifacts in `artifacts/` (gitignored).
- Track model tags and metadata in `registry/models.json`.
