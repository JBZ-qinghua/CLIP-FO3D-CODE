#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${CLIP_FO3D_PYTHON:-${project_root}/.conda/bin/python}"
output_root="${project_root}/outputs/scannet"
preflight_root="${project_root}/outputs/mink34c_preflight_8gpu"
log_file="${project_root}/logs/train_mink16unet34c_80k_8gpu.log"
evaluation_log="${project_root}/logs/evaluate_mink16unet34c_scannet_val.log"
result_file="${output_root}/scannet_val.json"

cd "${project_root}"
mkdir -p "${output_root}" "${preflight_root}" logs

"${python_bin}" -c 'import torch, MinkowskiEngine; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'

torchrun_args=(
  -m torch.distributed.run --standalone --nproc_per_node=8
  scripts/train.py --distributed --batch-size 1
)

# Validate the exact eight-rank forward/backward, rank-local normalization,
# gradient reduction, optimizer, scheduler, and rank-zero checkpoint path.
if [[ ! -f "${preflight_root}/checkpoint_step_1.pth" ]]; then
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
    "${python_bin}" "${torchrun_args[@]}" \
      --steps 1 --workers 0 --output-root "${preflight_root}" \
      2>&1 | tee "${project_root}/logs/train_mink16unet34c_preflight_8gpu.log"
fi

resume_args=()
latest_checkpoint="$(find "${output_root}" -maxdepth 1 -type f \
  -name 'checkpoint_step_*.pth' -print | sort -V | tail -n 1)"
if [[ -n "${latest_checkpoint}" ]]; then
  resume_args=(--resume "${latest_checkpoint}")
fi

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  "${python_bin}" "${torchrun_args[@]}" \
    --steps 80000 --workers 2 --output-root "${output_root}" \
    "${resume_args[@]}" >>"${log_file}" 2>&1

# Workers exit normally so torchrun does not turn a deliberate pause into a
# ChildFailedError.  The marker lets the outer supervisor distinguish a clean
# pause from completed training.
if [[ -f "${output_root}/PAUSED" ]]; then
  exit 75
fi

"${python_bin}" - <<'PY'
import torch

payload = torch.load("outputs/scannet/latest.pth", map_location="cpu", weights_only=False)
if int(payload.get("step", -1)) != 80000:
    raise SystemExit(f"Expected final step 80000, found {payload.get('step')}")
if int(payload.get("world_size", -1)) != 8:
    raise SystemExit(f"Expected world size 8, found {payload.get('world_size')}")
PY

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  "${python_bin}" scripts/evaluate.py \
    --checkpoint "${output_root}/latest.pth" --device cuda:0 \
    --output "${result_file}" \
    2>&1 | tee "${evaluation_log}"

date -u '+[%Y-%m-%dT%H:%M:%SZ] 8-GPU 80K training and full validation complete' \
  | tee -a "${log_file}"
