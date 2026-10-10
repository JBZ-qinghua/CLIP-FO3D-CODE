from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .dense_clip import DenseCLIPExtractor, crop_boxes


def natural_key(path: Path) -> list[object]:
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", path.stem)]


def scene_views(image_root: Path, split: str, scene: str, max_views: int | None = None) -> list[tuple[Path, Path]]:
    scene_root = image_root / split / scene
    colors = sorted((scene_root / "color").glob("*.png"), key=natural_key)
    pairs = [(path, scene_root / "correspondence" / f"{path.stem}.npy") for path in colors]
    pairs = [(image, corr) for image, corr in pairs if corr.is_file()]
    return pairs if max_views is None else pairs[:max_views]


def _valid_correspondence(correspondence: np.ndarray, width: int, height: int, num_points: int) -> np.ndarray:
    correspondence = np.asarray(correspondence)
    if correspondence.ndim != 2 or correspondence.shape[1] != 3:
        raise ValueError(f"Expected correspondence Nx3 [x,y,point], got {correspondence.shape}")
    correspondence = np.rint(correspondence).astype(np.int64, copy=False)
    valid = (
        (correspondence[:, 0] >= 0) & (correspondence[:, 0] < width)
        & (correspondence[:, 1] >= 0) & (correspondence[:, 1] < height)
        & (correspondence[:, 2] >= 0) & (correspondence[:, 2] < num_points)
    )
    correspondence = correspondence[valid]
    # The Pointcept preprocessing may emit multiple mesh pixels that map to the
    # same processed point. Its dataset loader stores one pixel per point by
    # assignment (the last occurrence wins), which also matches the paper's
    # point-to-pixel rather than pixel-to-point projection definition.
    _, reverse_index = np.unique(correspondence[::-1, 2], return_index=True)
    keep = (len(correspondence) - 1 - reverse_index)
    return correspondence[np.sort(keep)]


def project_scene(
    extractor: DenseCLIPExtractor,
    point_root: str | Path,
    image_root: str | Path,
    split: str,
    scene: str,
    scales: Sequence[float] = (1.0, 0.5, 0.25),
    stride_ratio: float = 0.5,
    max_views: int | None = None,
    view_batch_size: int = 4,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Extract and project dense CLIP features for one ScanNet scene.

    Crop contributions are divided by each pixel's overlap count before point
    aggregation. This reproduces pixel-space multi-scale averaging without ever
    materializing the several-GB HxWx512 feature map.
    """
    point_root = Path(point_root)
    image_root = Path(image_root)
    coord = np.load(point_root / split / scene / "coord.npy", mmap_mode="r")
    num_points = int(coord.shape[0])
    point_sum = torch.zeros((num_points, extractor.output_dim), dtype=torch.float32)
    point_pixels = torch.zeros(num_points, dtype=torch.float32)
    views = scene_views(image_root, split, scene, max_views=max_views)
    if not views:
        raise FileNotFoundError(f"No RGB/correspondence pairs found for {split}/{scene}")

    iterator = tqdm(total=len(views), desc=f"project {split}/{scene}", disable=not show_progress)
    used_correspondences = 0
    for view_start in range(0, len(views), view_batch_size):
        states = []
        for image_path, correspondence_path in views[view_start:view_start + view_batch_size]:
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
            width, height = image.size
            correspondence = _valid_correspondence(
                np.load(correspondence_path, mmap_mode="r"), width, height, num_points
            )
            if correspondence.size == 0:
                continue
            xs, ys, point_ids = correspondence.T
            coverage = np.zeros(len(correspondence), dtype=np.uint8)
            for x0, y0, x1, y1 in crop_boxes(width, height, scales, stride_ratio):
                coverage += ((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).astype(np.uint8)
            if np.any(coverage == 0):
                raise RuntimeError("Crop generator left some pixels uncovered")
            states.append((image, xs, ys, point_ids, coverage))

        if states:
            regions_batch = extractor.region_crops_batch(
                [state[0] for state in states], scales=scales, stride_ratio=stride_ratio
            )
            for (_, xs, ys, point_ids, coverage), regions in zip(states, regions_batch):
                for region in regions:
                    x0, y0, x1, y1 = region.box_xyxy
                    inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
                    rows = np.flatnonzero(inside)
                    if rows.size == 0:
                        continue
                    local_labels = region.labels[ys[rows] - y0, xs[rows] - x0]
                    for start in range(0, len(rows), 25_000):
                        stop = min(start + 25_000, len(rows))
                        features = region.features[torch.from_numpy(local_labels[start:stop]).long()].float()
                        weights = torch.from_numpy(
                            1.0 / coverage[rows[start:stop]].astype(np.float32)
                        ).unsqueeze(1)
                        ids = torch.from_numpy(point_ids[rows[start:stop]])
                        point_sum.index_add_(0, ids, features * weights)
                point_pixels.index_add_(0, torch.from_numpy(point_ids), torch.ones(len(point_ids)))
                used_correspondences += len(point_ids)
        iterator.update(min(view_batch_size, len(views) - view_start))
    iterator.close()

    valid = point_pixels > 0
    targets = torch.zeros_like(point_sum)
    targets[valid] = point_sum[valid] / point_pixels[valid, None]
    targets[valid] = F.normalize(targets[valid], dim=-1)
    stats = {
        "num_points": num_points,
        "valid_points": int(valid.sum()),
        "views": len(views),
        "correspondences": used_correspondences,
        "scales": [float(scale) for scale in scales],
        "stride_ratio": float(stride_ratio),
    }
    return targets, valid, stats


def save_targets(path: str | Path, features: torch.Tensor, valid: torch.Tensor, stats: dict[str, object], dtype: str = "float16") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = features.cpu().numpy().astype(np.float16 if dtype == "float16" else np.float32)
    # A scene takes minutes to generate. Publish it atomically so an interrupted
    # process can never leave a corrupt file that looks complete to a restart.
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, features=array, valid=valid.cpu().numpy(), **stats)
    temporary.replace(path)


def load_targets(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        return payload["features"].astype(np.float32), payload["valid"].astype(bool)


def verify_target_set(
    point_root: str | Path,
    image_root: str | Path,
    target_root: str | Path,
    split: str,
    scales: Sequence[float],
    stride_ratio: float,
) -> dict[str, object]:
    """Verify that every point scene has a target built from every exported view."""
    point_root, image_root, target_root = map(Path, (point_root, image_root, target_root))
    scenes = sorted(
        path.name for path in (point_root / split).glob("scene*")
        if (path / "coord.npy").is_file()
    )
    missing, invalid = [], []
    for scene in scenes:
        target_path = target_root / split / f"{scene}.npz"
        if not target_path.is_file():
            missing.append(scene)
            continue
        expected_views = len(scene_views(image_root, split, scene))
        try:
            with np.load(target_path) as payload:
                valid = (
                    expected_views > 0
                    and int(payload["views"]) == expected_views
                    and np.allclose(payload["scales"], scales)
                    and np.isclose(float(payload["stride_ratio"]), stride_ratio)
                )
        except (KeyError, OSError, ValueError):
            valid = False
        if not valid:
            invalid.append(scene)
    return {
        "split": split,
        "expected_scenes": len(scenes),
        "complete_scenes": len(scenes) - len(missing) - len(invalid),
        "missing_scenes": missing,
        "invalid_scenes": invalid,
    }
