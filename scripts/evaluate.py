#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clip_fo3d.config import load_config, project_path
from clip_fo3d.dataset import list_scenes, voxelize_scene
from clip_fo3d.metrics import confusion_matrix, summarize_confusion
from clip_fo3d.model import build_model
from infer import text_weights


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CLIP-FO3D checkpoint on ScanNet validation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(PROJECT / "configs/inference.yaml"))
    parser.add_argument("--point-root")
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--output", default="outputs/scannet_val.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or cfg["device"])
    point_root = Path(args.point_root).resolve() if args.point_root else project_path(
        cfg, cfg["paths"]["point_root"]
    )
    clip_checkpoint = Path(args.clip_checkpoint).resolve() if args.clip_checkpoint else project_path(
        cfg, cfg["paths"]["clip_checkpoint"]
    )
    split = cfg["evaluation"]["split"]
    scenes = list_scenes(point_root, split)
    if len(scenes) != 312:
        raise SystemExit(f"Expected all 312 ScanNet validation scenes, found {len(scenes)}")

    class_names = cfg["evaluation"]["class_names"]
    classifier = text_weights(
        clip_checkpoint, class_names, device, cfg["evaluation"]["prompt_ensemble"]
    )
    model = build_model(cfg["model"]).to(device).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    amp_enabled = device.type == "cuda" and cfg["evaluation"].get("precision") == "bfloat16"

    total = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for index, scene in enumerate(scenes, 1):
        scene_root = point_root / split / scene
        sample = voxelize_scene(scene_root, cfg["model"]["voxel_size"])
        grid = sample["grid_coord"].to(device)
        features = sample["features"].to(device)
        batch = torch.zeros(len(grid), dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            prediction_features = model(grid, features, batch)
        voxel_features = F.normalize(prediction_features.float(), dim=-1)
        voxel_prediction = (voxel_features @ classifier).argmax(dim=1).cpu()
        prediction = voxel_prediction[sample["inverse"]].numpy()
        labels = np.load(scene_root / "segment20.npy").reshape(-1)
        total += confusion_matrix(
            prediction, labels, len(class_names), cfg["evaluation"]["ignore_index"]
        )
        print(json.dumps({"progress": f"{index}/312", "scene": scene, **summarize_confusion(total)}))

    result = {"num_scenes": 312, "class_names": class_names, **summarize_confusion(total)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
