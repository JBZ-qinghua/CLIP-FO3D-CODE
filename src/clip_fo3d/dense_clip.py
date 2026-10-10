from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic


CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)


def _convert_clip_compute_weights(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Convert CLIP matmul weights while leaving LayerNorm parameters in fp32."""
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            module.weight.data = module.weight.data.to(dtype)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(dtype)
        if isinstance(module, torch.nn.MultiheadAttention):
            for attribute in (
                "in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight",
                "in_proj_bias", "bias_k", "bias_v",
            ):
                tensor = getattr(module, attribute, None)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)
        for attribute in ("text_projection", "proj"):
            tensor = getattr(module, attribute, None)
            if tensor is not None:
                tensor.data = tensor.data.to(dtype)


@dataclass(frozen=True)
class RegionCrop:
    """One crop's paper-faithful local-token features."""

    box_xyxy: tuple[int, int, int, int]
    labels: np.ndarray
    features: torch.Tensor


def crop_boxes(width: int, height: int, scales: Sequence[float], stride_ratio: float = 0.5) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for scale in scales:
        crop_w = max(1, min(width, round(width * float(scale))))
        crop_h = max(1, min(height, round(height * float(scale))))
        step_x = max(1, round(crop_w * stride_ratio))
        step_y = max(1, round(crop_h * stride_ratio))
        xs = list(range(0, max(width - crop_w, 0) + 1, step_x))
        ys = list(range(0, max(height - crop_h, 0) + 1, step_y))
        if not xs or xs[-1] != width - crop_w:
            xs.append(width - crop_w)
        if not ys or ys[-1] != height - crop_h:
            ys.append(height - crop_h)
        boxes.extend((x, y, x + crop_w, y + crop_h) for y in ys for x in xs)
    return boxes


class DenseCLIPExtractor:
    """Dense CLIP ViT-B/16 extractor from Section 3.1 of CLIP-FO3D.

    The original CLIP tokens never attend to the added local tokens. Each local
    token attends only to patch tokens assigned to its SLIC super-pixel, matching
    equations (1)-(2) in the paper.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cuda",
        n_segments: int = 50,
        compactness: float = 10.0,
        slic_sigma: float = 1.0,
        crop_batch_size: int = 256,
        precision: str = "float16",
        slic_workers: int = 8,
    ) -> None:
        try:
            import clip
        except ImportError as exc:
            raise RuntimeError("OpenAI CLIP is required; install the project environment first") from exc

        self.device = torch.device(device)
        self.model, _ = clip.load(str(checkpoint), device=self.device, jit=False)
        if precision == "float16":
            # clip.load keeps CUDA convolution/attention weights in fp16 while
            # preserving LayerNorm parameters in fp32 for numerical stability.
            pass
        elif precision == "bfloat16":
            self.model.float()
            _convert_clip_compute_weights(self.model, torch.bfloat16)
        elif precision == "float32":
            self.model.float()
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        visual = self.model.visual
        if not hasattr(visual, "transformer") or visual.input_resolution != 224:
            raise ValueError("CLIP-FO3D requires the OpenAI CLIP ViT-B/16 visual encoder")
        self.visual = visual
        self.input_resolution = int(visual.input_resolution)
        self.patch_size = int(visual.conv1.kernel_size[0])
        self.grid_size = self.input_resolution // self.patch_size
        self.n_segments = int(n_segments)
        self.compactness = float(compactness)
        self.slic_sigma = float(slic_sigma)
        self.crop_batch_size = int(crop_batch_size)
        self.precision = precision
        self.slic_workers = int(slic_workers)
        if self.crop_batch_size < 1:
            raise ValueError("crop_batch_size must be positive")
        if self.slic_workers < 1:
            raise ValueError("slic_workers must be positive")
        self._slic_pool = ThreadPoolExecutor(max_workers=self.slic_workers)

    @property
    def output_dim(self) -> int:
        return int(self.visual.proj.shape[1])

    @staticmethod
    def _image_array_cpu(image: Image.Image, input_resolution: int) -> np.ndarray:
        image = image.convert("RGB").resize(
            (input_resolution, input_resolution), resample=Image.Resampling.BICUBIC
        )
        return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        array = self._image_array_cpu(image, self.input_resolution)
        tensor = torch.from_numpy(array.copy()).unsqueeze(0).to(self.device).float().div_(255.0)
        return (tensor - CLIP_MEAN.to(self.device)) / CLIP_STD.to(self.device)

    @staticmethod
    def _contiguous_labels(labels: np.ndarray) -> np.ndarray:
        _, inverse = np.unique(labels, return_inverse=True)
        return inverse.reshape(labels.shape).astype(np.int32)

    def _segment(self, image: Image.Image) -> np.ndarray:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        labels = slic(
            array,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.slic_sigma,
            start_label=0,
            channel_axis=-1,
            enforce_connectivity=True,
        )
        return self._contiguous_labels(labels)

    @torch.inference_mode()
    def encode_local_tokens(self, image: Image.Image, labels: np.ndarray) -> torch.Tensor:
        """Return one normalized 512-D CLIP feature per SLIC region."""
        return self.encode_local_tokens_batch([image], [labels])[0]

    @staticmethod
    def _paper_attention(
        attention: torch.nn.MultiheadAttention,
        sequence: torch.Tensor,
        original_len: int,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the paper's two disjoint attention paths with fused SDPA.

        Original CLIP tokens attend only to original tokens. Each local token
        attends only to the patch tokens assigned to its SLIC region.
        """
        length, batch, width = sequence.shape
        heads = attention.num_heads
        head_dim = width // heads
        qkv = F.linear(sequence, attention.in_proj_weight, attention.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(1, 0, 2).reshape(batch, length, heads, head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        original = F.scaled_dot_product_attention(
            q[:, :, :original_len], k[:, :, :original_len], v[:, :, :original_len], dropout_p=0.0
        )
        local_q = q[:, :, original_len:]
        patch_k = k[:, :, 1:original_len]
        patch_v = v[:, :, 1:original_len]
        local = F.scaled_dot_product_attention(
            local_q, patch_k, patch_v, attn_mask=local_mask, dropout_p=0.0
        )
        combined = torch.cat((original, local), dim=2)
        combined = combined.transpose(1, 2).reshape(batch, length, width).permute(1, 0, 2)
        return F.linear(combined, attention.out_proj.weight, attention.out_proj.bias)

    @torch.inference_mode()
    def encode_local_tokens_batch(
        self, images: Sequence[Image.Image], labels_batch: Sequence[np.ndarray]
    ) -> list[torch.Tensor]:
        """Batched local-token forwarding with an independent mask per crop."""
        if len(images) != len(labels_batch) or not images:
            raise ValueError("images and labels_batch must have the same non-zero length")
        # Resize/normalize in parallel, then make one host-to-device transfer.
        # Large ScanNet batches contain ~1,000 crops, so transferring each crop
        # separately leaves the GPU idle for most of the preprocessing phase.
        arrays = self._slic_pool.map(
            lambda image: self._image_array_cpu(image, self.input_resolution), images
        )
        # Transfer compact uint8 input once, then normalize in FP32 on the GPU.
        x = torch.from_numpy(np.stack(list(arrays))).to(self.device).float().div_(255.0)
        x = (x - CLIP_MEAN.to(self.device)) / CLIP_STD.to(self.device)
        x = x.to(dtype=self.visual.conv1.weight.dtype)
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        batch, patches, width = x.shape

        cls = self.visual.class_embedding.to(x.dtype).view(1, 1, width)
        original = torch.cat((cls.expand(batch, 1, width), x), dim=1)
        original = original + self.visual.positional_embedding.to(x.dtype).unsqueeze(0)

        patch_groups, represented_batch = [], []
        for labels in labels_batch:
            label_tensor = torch.from_numpy(labels.astype(np.float32))[None, None]
            patch_labels = F.interpolate(
                label_tensor, size=(self.grid_size, self.grid_size), mode="nearest"
            ).flatten().long().to(self.device)
            represented = torch.unique(patch_labels, sorted=True)
            if represented.numel() == 0:
                raise RuntimeError("SLIC produced no represented regions")
            represented_batch.append(represented.detach().cpu().numpy())
            patch_groups.append(torch.searchsorted(represented, patch_labels))

        local_counts = [len(represented) for represented in represented_batch]
        max_local = max(local_counts)
        local = cls.expand(batch, max_local, width).clone()
        local = local + self.visual.positional_embedding[0].to(x.dtype).view(1, 1, width)
        sequence = torch.cat((original, local), dim=1)
        sequence = self.visual.ln_pre(sequence).permute(1, 0, 2)

        original_len = patches + 1
        local_mask = torch.full(
            (batch, 1, max_local, patches), -10_000.0,
            dtype=sequence.dtype, device=sequence.device,
        )
        for batch_id, (patch_group, num_local) in enumerate(zip(patch_groups, local_counts)):
            local_ids = torch.arange(num_local, device=self.device)[:, None]
            local_mask[batch_id, 0, :num_local] = torch.where(
                patch_group[None] == local_ids,
                torch.zeros((), dtype=sequence.dtype, device=self.device),
                torch.full((), -10_000.0, dtype=sequence.dtype, device=self.device),
            )
            if num_local < max_local:
                local_mask[batch_id, 0, num_local:, 0] = 0
        for block in self.visual.transformer.resblocks:
            normalized = block.ln_1(sequence)
            attended = self._paper_attention(
                block.attn, normalized, original_len, local_mask
            )
            sequence = sequence + attended
            sequence = sequence + block.mlp(block.ln_2(sequence))

        local_features = sequence[original_len:].permute(1, 0, 2)
        local_features = self.visual.ln_post(local_features)
        if self.visual.proj is not None:
            local_features = local_features @ self.visual.proj
        local_features = F.normalize(local_features.float(), dim=-1)
        if not torch.isfinite(local_features).all():
            raise FloatingPointError(
                f"Non-finite dense CLIP features with precision={self.precision}; "
                "set dense_clip.precision to float32"
            )

        # Return features indexed by every original SLIC label. Labels without a
        # ViT patch inherit the closest represented label's feature.
        results = []
        for batch_id, (labels, represented, num_local) in enumerate(
            zip(labels_batch, represented_batch, local_counts)
        ):
            all_ids = np.arange(int(labels.max()) + 1)
            if np.array_equal(all_ids, represented):
                nearest = all_ids
            else:
                flat = labels.ravel()
                counts = np.bincount(flat, minlength=len(all_ids)).clip(min=1)
                yy, xx = np.indices(labels.shape)
                centroids = np.column_stack((
                    np.bincount(flat, weights=xx.ravel(), minlength=len(all_ids)) / counts,
                    np.bincount(flat, weights=yy.ravel(), minlength=len(all_ids)) / counts,
                ))
                represented_centroids = centroids[represented]
                nearest = (
                    (centroids[:, None] - represented_centroids[None]) ** 2
                ).sum(axis=-1).argmin(axis=1)
            results.append(
                local_features[batch_id, :num_local][torch.as_tensor(nearest, device=self.device)]
            )
        return results

    def iter_region_crops(
        self,
        image: Image.Image,
        scales: Sequence[float] = (1.0, 0.5, 0.25),
        stride_ratio: float = 0.5,
    ) -> Iterator[RegionCrop]:
        yield from self.region_crops_batch([image], scales, stride_ratio)[0]

    def region_crops_batch(
        self,
        images: Sequence[Image.Image],
        scales: Sequence[float] = (1.0, 0.5, 0.25),
        stride_ratio: float = 0.5,
    ) -> list[list[RegionCrop]]:
        """Extract crops from several views together to improve GPU occupancy."""
        descriptors: list[tuple[int, tuple[int, int, int, int], Image.Image]] = []
        for image_id, image in enumerate(images):
            for box in crop_boxes(*image.size, scales, stride_ratio):
                descriptors.append((image_id, box, image.crop(box)))
        labels_all = list(self._slic_pool.map(self._segment, [item[2] for item in descriptors]))
        results: list[list[RegionCrop]] = [[] for _ in images]
        for start in range(0, len(descriptors), self.crop_batch_size):
            batch_descriptors = descriptors[start:start + self.crop_batch_size]
            batch_labels = labels_all[start:start + self.crop_batch_size]
            features_batch = self.encode_local_tokens_batch(
                [item[2] for item in batch_descriptors], batch_labels
            )
            feature_sizes = [len(features) for features in features_batch]
            features_cpu = torch.cat(features_batch).cpu().split(feature_sizes)
            for (image_id, box, _), labels, features in zip(
                batch_descriptors, batch_labels, features_cpu
            ):
                results[image_id].append(RegionCrop(box, labels, features))
        return results

    def dense_feature_map(
        self,
        image: Image.Image,
        scales: Sequence[float] = (1.0, 0.5, 0.25),
        stride_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Materialize HxWx512 features; intended for small images/debugging."""
        width, height = image.size
        total = torch.zeros((height, width, self.output_dim), dtype=torch.float32)
        count = torch.zeros((height, width, 1), dtype=torch.float32)
        for region in self.iter_region_crops(image, scales, stride_ratio):
            x0, y0, x1, y1 = region.box_xyxy
            feature_map = region.features[torch.from_numpy(region.labels).long()]
            total[y0:y1, x0:x1] += feature_map
            count[y0:y1, x0:x1] += 1
        return F.normalize(total / count.clamp_min(1), dim=-1)
