# registry

Stores lightweight metadata mapping model tags to exported artifacts and their feature/data provenance. This is optional, but useful to make inference in `HH4b` reproducible.

Example `models.json` entry:
```json
{
  "v1.0.0": {
    "artifact": "artifacts/v1.0.0/model.onnx",
    "features": "v1",
    "hh4b_commit": "abc123",
    "data_tag": "24Sep25_v12v2_private_signal"
  }
}
```
