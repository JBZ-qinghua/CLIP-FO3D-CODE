import unittest

import numpy as np
import torch

from clip_fo3d.dense_clip import crop_boxes
from clip_fo3d.losses import negative_cosine_loss
from clip_fo3d.metrics import confusion_matrix, summarize_confusion


class CoreTests(unittest.TestCase):
    def test_fused_attention_matches_explicit_mask(self):
        from clip_fo3d.dense_clip import DenseCLIPExtractor

        torch.manual_seed(0)
        batch, original_len, local_count, width, heads = 2, 5, 3, 16, 4
        attention = torch.nn.MultiheadAttention(width, heads)
        sequence = torch.randn(original_len + local_count, batch, width)
        groups = [torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 1, 0])]
        counts = [3, 2]
        explicit = torch.ones(batch, original_len + local_count, original_len + local_count, dtype=torch.bool)
        explicit[:, :original_len, :original_len] = False
        local_mask = torch.full((batch, 1, local_count, original_len - 1), -10_000.0)
        for batch_id, (group, count) in enumerate(zip(groups, counts)):
            for local_id in range(count):
                explicit[batch_id, original_len + local_id, 1:original_len][group == local_id] = False
                local_mask[batch_id, 0, local_id, group == local_id] = 0
            for local_id in range(count, local_count):
                explicit[batch_id, original_len + local_id, original_len + local_id] = False
                local_mask[batch_id, 0, local_id, 0] = 0
        explicit = explicit[:, None].expand(batch, heads, -1, -1).reshape(
            batch * heads, original_len + local_count, original_len + local_count
        )
        reference = attention(sequence, sequence, sequence, attn_mask=explicit, need_weights=False)[0]
        fused = DenseCLIPExtractor._paper_attention(attention, sequence, original_len, local_mask)
        valid = torch.zeros(original_len + local_count, batch, dtype=torch.bool)
        valid[:original_len] = True
        for batch_id, count in enumerate(counts):
            valid[original_len:original_len + count, batch_id] = True
        self.assertTrue(torch.allclose(reference[valid], fused[valid], atol=1e-6, rtol=1e-5))

    def test_configured_crop_count_and_coverage(self):
        boxes = crop_boxes(400, 300, (1.0, 0.5, 0.25), 0.5)
        self.assertEqual(len(boxes), 59)
        coverage = np.zeros((300, 400), dtype=np.int16)
        for x0, y0, x1, y1 in boxes:
            coverage[y0:y1, x0:x1] += 1
        self.assertGreaterEqual(coverage.min(), 3)

    def test_negative_cosine_loss(self):
        prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = prediction.clone()
        self.assertTrue(torch.allclose(negative_cosine_loss(prediction, target), torch.tensor(-1.0)))
        self.assertTrue(torch.allclose(
            negative_cosine_loss(prediction, -target, torch.tensor([True, False])), torch.tensor(1.0)
        ))

    def test_metrics_ignore_label(self):
        target = np.array([0, 0, 1, 1, -1])
        pred = np.array([0, 1, 1, 1, 0])
        matrix = confusion_matrix(pred, target, 2)
        result = summarize_confusion(matrix)
        self.assertEqual(matrix.tolist(), [[1, 1], [0, 2]])
        self.assertTrue(np.isclose(result["allAcc"], 0.75))


if __name__ == "__main__":
    unittest.main()
