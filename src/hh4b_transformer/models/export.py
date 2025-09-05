from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch  # type: ignore

# TODO: REVIEW


def export_torchscript(
    model: torch.nn.Module, example: torch.Tensor, out_path: str | Path, meta: Dict[str, Any]
) -> Path:
    model.eval()
    ts = torch.jit.trace(model, example)
    out_path = Path(out_path)
    ts.save(str(out_path))
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out_path


def export_onnx(
    model: torch.nn.Module,
    example: torch.Tensor,
    out_path: str | Path,
    meta: Dict[str, Any],
    opset: int = 17,
) -> Path:
    model.eval()
    out_path = Path(out_path)
    torch.onnx.export(
        model,
        example,
        f=str(out_path),
        input_names=["x"],
        output_names=["logit"],
        opset_version=opset,
        dynamic_axes={"x": {0: "batch"}, "logit": {0: "batch"}},
    )
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out_path
