from __future__ import annotations

import io
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


COLOR_COMPRESSION = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


@dataclass
class RGBDFrame:
    camera_to_world: np.ndarray
    color_data: bytes
    depth_data: bytes

    @classmethod
    def load(cls, handle) -> "RGBDFrame":
        camera_to_world = np.asarray(struct.unpack("f" * 16, handle.read(64)), dtype=np.float32).reshape(4, 4)
        handle.read(16)  # color/depth timestamps
        color_size = struct.unpack("Q", handle.read(8))[0]
        depth_size = struct.unpack("Q", handle.read(8))[0]
        return cls(camera_to_world, handle.read(color_size), handle.read(depth_size))


class SensorData:
    """Minimal ScanNet v4 .sens reader used for the paper's 10-frame export."""

    def __init__(self, path: str | Path) -> None:
        with Path(path).open("rb") as handle:
            version = struct.unpack("I", handle.read(4))[0]
            if version != 4:
                raise ValueError(f"Unsupported ScanNet .sens version {version}")
            sensor_name_length = struct.unpack("Q", handle.read(8))[0]
            self.sensor_name = handle.read(sensor_name_length).decode("utf-8", errors="replace")
            self.intrinsic_color = np.asarray(struct.unpack("f" * 16, handle.read(64)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack("f" * 16, handle.read(64)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack("f" * 16, handle.read(64)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack("f" * 16, handle.read(64)), dtype=np.float32).reshape(4, 4)
            self.color_compression = COLOR_COMPRESSION[struct.unpack("i", handle.read(4))[0]]
            self.depth_compression = DEPTH_COMPRESSION[struct.unpack("i", handle.read(4))[0]]
            self.color_width = struct.unpack("I", handle.read(4))[0]
            self.color_height = struct.unpack("I", handle.read(4))[0]
            self.depth_width = struct.unpack("I", handle.read(4))[0]
            self.depth_height = struct.unpack("I", handle.read(4))[0]
            self.depth_shift = struct.unpack("f", handle.read(4))[0]
            frame_count = struct.unpack("Q", handle.read(8))[0]
            self.frames = [RGBDFrame.load(handle) for _ in range(frame_count)]

    def decode_color(self, frame: RGBDFrame) -> np.ndarray:
        if self.color_compression not in {"jpeg", "png"}:
            raise ValueError(f"Unsupported color compression: {self.color_compression}")
        image = imageio.imread(io.BytesIO(frame.color_data))
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        return image[..., :3]

    def decode_depth(self, frame: RGBDFrame) -> np.ndarray:
        if self.depth_compression == "zlib_ushort":
            raw = zlib.decompress(frame.depth_data)
        elif self.depth_compression == "raw_ushort":
            raw = frame.depth_data
        else:
            raise ValueError(f"Unsupported depth compression: {self.depth_compression}")
        return np.frombuffer(raw, dtype=np.uint16).reshape(self.depth_height, self.depth_width)


def project_points_with_depth(
    coords_world: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    rendered_depth: np.ndarray,
    depth_tolerance: float,
) -> np.ndarray:
    """Project points to one RGB frame and reject occluded points by mesh depth."""
    if not np.isfinite(camera_to_world).all():
        return np.full((1, 3), -1, dtype=np.int32)
    world_to_camera = np.linalg.inv(camera_to_world)
    coords_camera = coords_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = coords_camera[:, 2]
    positive = z > 1e-5
    pixels_h = coords_camera @ intrinsic[:3, :3].T
    u = np.rint(pixels_h[:, 0] / np.maximum(z, 1e-5)).astype(np.int64)
    v = np.rint(pixels_h[:, 1] / np.maximum(z, 1e-5)).astype(np.int64)
    height, width = rendered_depth.shape
    inside = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    point_ids = np.flatnonzero(inside)
    if point_ids.size == 0:
        return np.full((1, 3), -1, dtype=np.int32)
    surface_depth = rendered_depth[v[point_ids], u[point_ids]]
    visible = np.isfinite(surface_depth) & (np.abs(z[point_ids] - surface_depth) <= depth_tolerance)
    point_ids = point_ids[visible]
    if point_ids.size == 0:
        return np.full((1, 3), -1, dtype=np.int32)
    return np.column_stack((u[point_ids], v[point_ids], point_ids)).astype(np.int32)


def prepare_scene_10frames(
    raw_scene: str | Path,
    point_scene: str | Path,
    output_scene: str | Path,
    frame_gap: int = 10,
    depth_tolerance: float = 0.05,
    export_depth: bool = False,
) -> dict[str, object]:
    import camtools as ct
    import open3d as o3d

    if frame_gap != 10:
        raise ValueError("CLIP-FO3D paper protocol requires frame_gap=10")
    raw_scene, point_scene, output_scene = map(Path, (raw_scene, point_scene, output_scene))
    scene = raw_scene.name
    sens_path = raw_scene / f"{scene}.sens"
    mesh_path = raw_scene / f"{scene}_vh_clean_2.ply"
    coord_path = point_scene / "coord.npy"
    for required in (sens_path, mesh_path, coord_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    sensor = SensorData(sens_path)
    coords = np.load(coord_path).astype(np.float32)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    color_dir = output_scene / "color"
    pose_dir = output_scene / "pose"
    correspondence_dir = output_scene / "correspondence"
    intrinsic_dir = output_scene / "intrinsic"
    depth_dir = output_scene / "depth"
    for directory in (color_dir, pose_dir, correspondence_dir, intrinsic_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if export_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)
    np.save(intrinsic_dir / "intrinsic.npy", sensor.intrinsic_color)

    used = 0
    visible_pairs = 0
    for frame_id in range(0, len(sensor.frames), frame_gap):
        frame = sensor.frames[frame_id]
        color_path = color_dir / f"{frame_id}.png"
        pose_path = pose_dir / f"{frame_id}.npy"
        corr_path = correspondence_dir / f"{frame_id}.npy"
        if not color_path.is_file():
            imageio.imwrite(color_path, sensor.decode_color(frame))
        np.save(pose_path, frame.camera_to_world)
        if export_depth and not (depth_dir / f"{frame_id}.png").is_file():
            imageio.imwrite(depth_dir / f"{frame_id}.png", sensor.decode_depth(frame))

        if not corr_path.is_file():
            if np.isfinite(frame.camera_to_world).all():
                rendered_depth = ct.raycast.mesh_to_im_depth(
                    mesh=mesh,
                    K=sensor.intrinsic_color[:3, :3],
                    T=np.linalg.inv(frame.camera_to_world),
                    height=sensor.color_height,
                    width=sensor.color_width,
                )
                correspondence = project_points_with_depth(
                    coords, sensor.intrinsic_color, frame.camera_to_world,
                    rendered_depth, depth_tolerance,
                )
            else:
                correspondence = np.full((1, 3), -1, dtype=np.int32)
            np.save(corr_path, correspondence)
        else:
            correspondence = np.load(corr_path, mmap_mode="r")
        used += 1
        if not np.array_equal(correspondence, np.full((1, 3), -1)):
            visible_pairs += len(correspondence)

    metadata = {
        "scene": scene,
        "frame_gap": frame_gap,
        "source_frames": len(sensor.frames),
        "exported_frames": used,
        "visible_point_frame_pairs": visible_pairs,
        "color_size": [sensor.color_width, sensor.color_height],
        "depth_tolerance_m": depth_tolerance,
    }
    (output_scene / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def verify_10frame_export(point_root: str | Path, image_root: str | Path, splits=("train", "val")) -> dict[str, object]:
    point_root, image_root = Path(point_root), Path(image_root)
    report: dict[str, object] = {"frame_gap": 10, "splits": {}}
    for split in splits:
        point_scenes = sorted(path.name for path in (point_root / split).glob("scene*") if path.is_dir())
        missing, invalid, total_frames = [], [], 0
        for scene in point_scenes:
            scene_root = image_root / split / scene
            color_ids = sorted(int(path.stem) for path in (scene_root / "color").glob("*.png"))
            corr_ids = sorted(int(path.stem) for path in (scene_root / "correspondence").glob("*.npy"))
            pose_ids = sorted(int(path.stem) for path in (scene_root / "pose").glob("*.npy"))
            if not color_ids:
                missing.append(scene)
                continue
            metadata_path = scene_root / "metadata.json"
            if not metadata_path.is_file():
                invalid.append(scene)
            else:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                expected_ids = list(range(0, int(metadata.get("source_frames", 0)), 10))
                if (
                    metadata.get("frame_gap") != 10
                    or int(metadata.get("exported_frames", -1)) != len(expected_ids)
                    or color_ids != expected_ids
                ):
                    invalid.append(scene)
            if color_ids != corr_ids or color_ids != pose_ids or not (scene_root / "intrinsic/intrinsic.npy").is_file():
                invalid.append(scene)
            total_frames += len(color_ids)
        report["splits"][split] = {
            "expected_scenes": len(point_scenes), "complete_scenes": len(point_scenes) - len(set(missing + invalid)),
            "frames": total_frames, "missing_scenes": missing, "invalid_scenes": sorted(set(invalid)),
        }
    return report
