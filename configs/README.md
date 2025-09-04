# configs

Centralized YAML configuration for data paths, features, model hyperparameters, and exports. This keeps runs reproducible and portable across environments.

- `data.yaml`: data roots/tags, years, loader options
- `features.yaml`: feature list/version, targets, weight key
- `model.yaml`: architecture/hparams
- `train.yaml`: optimizer/schedule/seed/logging
- `export.yaml`: export format and metadata
