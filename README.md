# hh4b-transformer

A lightweight repository for training transformer-based models for the HH→4b analysis on the NRP-Nautilus cluster via Kubernetes. 


Details about Data, Architecture, Features etc. can be found in the [overview](docs/overview.md#notes).

For tasks that are yet to be completed,  see the list of [TODOs](docs/overview.md#TODO).

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
- Data lives outside this repo. Expected structure (as in `HH4b`):
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
