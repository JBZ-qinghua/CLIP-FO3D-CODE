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

from clip_fo3d.config import load_config, project_path
from clip_fo3d.dataset import voxelize_scene
from clip_fo3d.model import build_model
from clip_fo3d.prompts import IMAGENET_TEMPLATES, SINGLE_TEMPLATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP-FO3D inference on one preprocessed point cloud")
    parser.add_argument("--scene-root", required=True, help="Directory containing coord/color/normal .npy files")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(PROJECT / "configs/inference.yaml"))
    parser.add_argument("--clip-checkpoint", help="OpenAI CLIP ViT-B/16 checkpoint")
    parser.add_argument("--class-name", action="append", help="Custom text query; repeat for multiple classes")
    parser.add_argument("--single-template", action="store_true")
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--output")
    return parser.parse_args()


@torch.inference_mode()
def text_weights(checkpoint: Path, names: list[str], device: torch.device, ensemble: bool) -> torch.Tensor:
    import clip

    model, _ = clip.load(str(checkpoint), device=device, jit=False)
    templates = IMAGENET_TEMPLATES if ensemble else SINGLE_TEMPLATE
    encoded = []
    for name in names:
        tokens = clip.tokenize([template.format(name) for template in templates]).to(device)
        features = F.normalize(model.encode_text(tokens).float(), dim=-1)
        encoded.append(F.normalize(features.mean(dim=0), dim=0))
    return torch.stack(encoded, dim=1)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or cfg["device"])
    scene_root = Path(args.scene_root).resolve()
    for name in ("coord.npy",):
        if not (scene_root / name).is_file():
            raise SystemExit(f"Missing input file: {scene_root / name}")

    class_names = args.class_name or cfg["evaluation"]["class_names"]
    clip_checkpoint = (
        Path(args.clip_checkpoint).resolve()
        if args.clip_checkpoint
        else project_path(cfg, cfg["paths"]["clip_checkpoint"])
    )
    classifier = text_weights(clip_checkpoint, class_names, device, not args.single_template)

    model = build_model(cfg["model"]).to(device).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    sample = voxelize_scene(scene_root, cfg["model"]["voxel_size"])
    grid = sample["grid_coord"].to(device)
    inputs = sample["features"].to(device)
    batch = torch.zeros(len(grid), dtype=torch.long, device=device)
    amp_enabled = device.type == "cuda" and cfg["evaluation"].get("precision") == "bfloat16"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        prediction_features = model(grid, inputs, batch)
    voxel_features = F.normalize(prediction_features.float(), dim=-1)
    scores = voxel_features @ classifier
    voxel_confidence, voxel_prediction = scores.max(dim=1)
    inverse = sample["inverse"].numpy()
    prediction = voxel_prediction.cpu().numpy()[inverse].astype(np.int16)
    confidence = voxel_confidence.cpu().numpy()[inverse].astype(np.float16)

    output = Path(args.output) if args.output else scene_root.parent / f"{scene_root.name}_clip_fo3d.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, np.ndarray] = {
        "prediction": prediction,
        "confidence": confidence,
        "class_names": np.asarray(class_names),
    }
    if args.save_features:
        result["features"] = voxel_features.cpu().numpy()[inverse].astype(np.float16)
    np.savez_compressed(output, **result)
    print(json.dumps({
        "scene": scene_root.name,
        "points": int(len(prediction)),
        "classes": class_names,
        "output": str(output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
