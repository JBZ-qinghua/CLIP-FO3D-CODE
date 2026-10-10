from __future__ import annotations

import torch
import torch.nn.functional as F


def negative_cosine_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    if valid is not None:
        prediction = prediction[valid]
        target = target[valid]
    if prediction.numel() == 0:
        return prediction.sum() * 0.0
    return -F.cosine_similarity(prediction.float(), target.float(), dim=-1).mean()

