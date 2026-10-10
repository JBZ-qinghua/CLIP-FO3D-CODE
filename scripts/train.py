#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from clip_fo3d.config import load_config, project_path
from clip_fo3d.dataset import ScanNetDistillationDataset, collate_sparse
from clip_fo3d.losses import negative_cosine_loss
from clip_fo3d.model import build_model
from clip_fo3d.projection import verify_target_set


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill projected dense CLIP features into Mink16UNet34C"
    )
    parser.add_argument("--config", default=str(PROJECT / "configs/scannet.yaml"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("float32", "bfloat16"))
    parser.add_argument("--resume")
    parser.add_argument("--output-root", help="Override checkpoint output directory")
    parser.add_argument("--distributed", action="store_true", help="Use torchrun/DDP")
    parser.add_argument("--sync-bn", action="store_true", help="Synchronize sparse BN statistics across ranks")
    parser.add_argument("--stop-file", help="Save an immediate resumable checkpoint when this file appears")
    parser.add_argument("--allow-incomplete-targets", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    distributed = args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        if not torch.cuda.is_available():
            raise SystemExit("NCCL distributed training requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device(args.device or cfg["device"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed = int(cfg["seed"])
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    train_cfg = cfg["train"]
    precision = args.precision or train_cfg.get("precision", "float32")
    amp_enabled = device.type == "cuda" and precision == "bfloat16"
    steps = args.steps or train_cfg["steps"]
    point_root = project_path(cfg, cfg["paths"]["point_root"])
    target_root = project_path(cfg, cfg["paths"]["target_root"])
    verification_error = None
    if not args.allow_incomplete_targets and rank == 0:
        report = verify_target_set(
            point_root,
            project_path(cfg, cfg["paths"]["image_root"]),
            target_root,
            train_cfg["split"],
            cfg["dense_clip"]["scales"],
            cfg["dense_clip"]["stride_ratio"],
        )
        if report["complete_scenes"] != report["expected_scenes"]:
            verification_error = (
                "Teacher target verification failed: "
                f"{report['complete_scenes']}/{report['expected_scenes']} complete. "
                "Run scripts/extract_targets.py for the full training split."
            )
    if distributed:
        messages = [verification_error]
        dist.broadcast_object_list(messages, src=0)
        verification_error = messages[0]
    if verification_error:
        raise SystemExit(verification_error)
    dataset = ScanNetDistillationDataset(
        point_root,
        target_root,
        train_cfg["split"], cfg["model"]["voxel_size"], augment=train_cfg["augment"],
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
        seed=seed, drop_last=True,
    ) if distributed else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size or train_cfg["batch_size"],
        shuffle=sampler is None, sampler=sampler,
        num_workers=args.workers if args.workers is not None else train_cfg["workers"],
        collate_fn=collate_sparse, drop_last=True, pin_memory=True,
    )
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise SystemExit("Training loader is empty; check the target set and batch size")
    model = build_model(cfg["model"]).to(device)
    if distributed and args.sync_bn:
        import MinkowskiEngine as ME
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=train_cfg["lr"], momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_cfg["lr_step"], gamma=train_cfg["lr_gamma"]
    )
    start_step = 0
    resume_data_epoch = None
    resume_batch_offset = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        resume_data_epoch = checkpoint.get("data_epoch")
        resume_batch_offset = checkpoint.get("next_batch_index")

    train_model = DistributedDataParallel(
        model, device_ids=[device.index], output_device=device.index,
        broadcast_buffers=args.sync_bn, gradient_as_bucket_view=True,
        static_graph=True,
    ) if distributed else model

    output = Path(args.output_root).resolve() if args.output_root else project_path(cfg, cfg["paths"]["output_root"])
    stop_file = Path(args.stop_file).resolve() if args.stop_file else output / "STOP_REQUESTED"
    paused_marker = output / "PAUSED"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        # A marker belongs to the previous launcher invocation.  Clear it only
        # after this invocation has successfully initialized all training state.
        paused_marker.unlink(missing_ok=True)
    if distributed:
        dist.barrier()
    train_model.train()
    step = start_step
    log_time = time.perf_counter()
    log_step = step
    data_epoch = int(resume_data_epoch) if resume_data_epoch is not None else start_step // steps_per_epoch
    batch_offset = int(resume_batch_offset) if resume_batch_offset is not None else start_step % steps_per_epoch
    if batch_offset >= steps_per_epoch:
        data_epoch += batch_offset // steps_per_epoch
        batch_offset %= steps_per_epoch
    pause_requested = False
    while step < steps and not pause_requested:
        if sampler is not None:
            sampler.set_epoch(data_epoch)
        for batch_index, batch_data in enumerate(loader):
            if batch_index < batch_offset:
                continue
            if step >= steps:
                break
            grid = batch_data["grid_coord"].to(device, non_blocking=True)
            features = batch_data["features"].to(device, non_blocking=True)
            batch = batch_data["batch"].to(device, non_blocking=True)
            targets = batch_data["targets"].to(device, non_blocking=True)
            valid = batch_data["valid"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                prediction = train_model(grid, features, batch)
            if distributed:
                local_loss_sum = -F.cosine_similarity(
                    prediction[valid].float(), targets[valid].float(), dim=-1
                ).sum()
                global_valid = valid.sum(dtype=torch.long)
                dist.all_reduce(global_valid, op=dist.ReduceOp.SUM)
                # DDP averages gradients across ranks. Multiplication by the
                # world size makes the resulting gradient the exact global
                # per-voxel mean rather than a mean of per-rank means.
                loss = local_loss_sum * world_size / global_valid.clamp_min(1)
            else:
                loss = negative_cosine_loss(prediction.float(), targets, valid)
                global_valid = valid.sum(dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step(); scheduler.step(); step += 1
            pause_flag = torch.tensor(
                [int(rank == 0 and stop_file.is_file())], dtype=torch.uint8, device=device
            )
            if distributed:
                dist.broadcast(pause_flag, src=0)
            pause_requested = bool(pause_flag.item())
            if step == 1 or step % train_cfg["log_every"] == 0:
                now = time.perf_counter()
                if distributed:
                    global_loss_sum = local_loss_sum.detach().clone()
                    dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
                    log_loss = global_loss_sum / global_valid.clamp_min(1)
                else:
                    log_loss = loss.detach()
                if rank == 0:
                    print(json.dumps({"step": step, "loss": float(log_loss),
                                      "lr": scheduler.get_last_lr()[0],
                                      "valid_voxels": int(global_valid),
                                      "world_size": world_size,
                                      "global_batch_size": (args.batch_size or train_cfg["batch_size"]) * world_size,
                                      "seconds_per_step": (now - log_time) / max(step - log_step, 1)}),
                          flush=True)
                log_time, log_step = now, step
            if step % train_cfg["save_every"] == 0 or step == steps or pause_requested:
                if distributed:
                    dist.barrier()
                if rank == 0:
                    path = output / f"checkpoint_step_{step}.pth"
                    next_data_epoch = data_epoch
                    next_batch_index = batch_index + 1
                    if next_batch_index >= steps_per_epoch:
                        next_data_epoch += 1
                        next_batch_index = 0
                    atomic_torch_save(
                        {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "config": cfg,
                         "world_size": world_size,
                         "global_batch_size": (args.batch_size or train_cfg["batch_size"]) * world_size,
                         "data_epoch": next_data_epoch,
                         "next_batch_index": next_batch_index,
                         "paused": pause_requested}, path,
                    )
                    atomic_torch_save(
                        {"step": step, "model": model.state_dict(), "config": cfg,
                         "world_size": world_size,
                         "global_batch_size": (args.batch_size or train_cfg["batch_size"]) * world_size},
                        output / "latest.pth",
                    )
                    if pause_requested:
                        stop_file.unlink(missing_ok=True)
                        paused_marker.touch()
                        print(json.dumps({"event": "pause_checkpoint_saved", "step": step,
                                          "checkpoint": str(path)}), flush=True)
                if distributed:
                    dist.barrier()
                if pause_requested:
                    break
        batch_offset = 0
        data_epoch += 1
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
