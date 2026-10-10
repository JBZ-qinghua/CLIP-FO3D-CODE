from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def elastic_distortion(
    coord: np.ndarray,
    parameters: tuple[tuple[float, float], ...] = ((0.2, 0.4), (0.8, 1.6)),
) -> np.ndarray:
    """OpenScene/BPNet-style smooth coordinate distortion in metric space."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import convolve

    result = coord
    for granularity, magnitude in parameters:
        coord_min = result.min(axis=0)
        noise_dim = ((result - coord_min).max(axis=0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)
        kernels = (
            np.ones((3, 1, 1, 1), dtype=np.float32) / 3,
            np.ones((1, 3, 1, 1), dtype=np.float32) / 3,
            np.ones((1, 1, 3, 1), dtype=np.float32) / 3,
        )
        for _ in range(2):
            for kernel in kernels:
                noise = convolve(noise, kernel, mode="constant", cval=0)
        axes = [
            np.linspace(lower - granularity, lower + granularity * (size - 2), size)
            for lower, size in zip(coord_min, noise_dim)
        ]
        interpolator = RegularGridInterpolator(axes, noise, bounds_error=False, fill_value=0)
        result = result + interpolator(result).astype(np.float32) * magnitude
    return result.astype(np.float32, copy=False)

def list_scenes(scannet_root: str | Path, split: str) -> list[str]:
    root = Path(scannet_root) / split
    return sorted(path.name for path in root.glob("scene*") if (path / "coord.npy").is_file())


def load_targets(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        return payload["features"].astype(np.float32), payload["valid"].astype(bool)


def voxelize_scene(
    scene_path: str | Path,
    voxel_size: float,
    target_path: str | Path | None = None,
    augment: bool = False,
) -> dict[str, torch.Tensor | str]:
    scene_path = Path(scene_path)
    coord = np.load(scene_path / "coord.npy").astype(np.float32)
    coord_for_grid = coord
    if augment:
        if np.random.random() < 0.95:
            coord_for_grid = elastic_distortion(coord_for_grid)
        angles = np.array([
            np.random.uniform(-np.pi / 64, np.pi / 64),
            np.random.uniform(-np.pi / 64, np.pi / 64),
            np.random.uniform(-np.pi, np.pi),
        ])
        cx, cy, cz = np.cos(angles); sx, sy, sz = np.sin(angles)
        rotations = [
            np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]),
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]),
            np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]),
        ]
        np.random.shuffle(rotations)
        rotation = (rotations[0] @ rotations[1] @ rotations[2]).astype(np.float32)
        scale = np.float32(np.random.uniform(0.9, 1.1))
        coord_for_grid = coord_for_grid @ rotation.T * scale
    grid = np.floor(
        (coord_for_grid - coord_for_grid.min(axis=0, keepdims=True)) / voxel_size
    ).astype(np.int32)
    unique_grid, unique_index, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
    if augment and np.random.random() < 0.95:
        for axis in (0, 1):
            if np.random.random() < 0.5:
                unique_grid[:, axis] = unique_grid[:, axis].max() - unique_grid[:, axis]
    num_voxels = len(unique_grid)

    result: dict[str, torch.Tensor | str] = {
        "scene": scene_path.name,
        "grid_coord": torch.from_numpy(unique_grid),
        "features": torch.ones((num_voxels, 3), dtype=torch.float32),
        "inverse": torch.from_numpy(inverse.astype(np.int64)),
    }
    label_path = scene_path / "segment20.npy"
    if label_path.is_file():
        labels = np.load(label_path).reshape(-1).astype(np.int64)
        result["point_labels"] = torch.from_numpy(labels)
    if target_path is not None:
        targets, valid = load_targets(target_path)
        if len(targets) != len(coord):
            raise ValueError(f"Target/point mismatch in {scene_path.name}: {len(targets)} != {len(coord)}")
        result["targets"] = torch.from_numpy(targets[unique_index])
        result["valid"] = torch.from_numpy(valid[unique_index])
    return result


class ScanNetDistillationDataset(Dataset):
    def __init__(
        self, scannet_root: str | Path, target_root: str | Path, split: str,
        voxel_size: float, augment: bool = True,
    ) -> None:
        self.scannet_root = Path(scannet_root)
        self.target_root = Path(target_root)
        self.split = split
        self.voxel_size = float(voxel_size)
        self.augment = bool(augment)
        available = {path.stem for path in (self.target_root / split).glob("*.npz")}
        self.scenes = [scene for scene in list_scenes(self.scannet_root, split) if scene in available]
        if not self.scenes:
            raise FileNotFoundError(f"No extracted targets under {self.target_root / split}")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        scene = self.scenes[index]
        return voxelize_scene(
            self.scannet_root / self.split / scene,
            self.voxel_size,
            self.target_root / self.split / f"{scene}.npz",
            augment=self.augment,
        )


def collate_sparse(samples: Iterable[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    samples = list(samples)
    counts = torch.tensor([len(sample["grid_coord"]) for sample in samples], dtype=torch.long)
    batch = torch.repeat_interleave(torch.arange(len(samples)), counts)
    return {
        "scene": [str(sample["scene"]) for sample in samples],
        "grid_coord": torch.cat([sample["grid_coord"] for sample in samples]),
        "features": torch.cat([sample["features"] for sample in samples]),
        "targets": torch.cat([sample["targets"] for sample in samples]),
        "valid": torch.cat([sample["valid"] for sample in samples]),
        "batch": batch,
    }
