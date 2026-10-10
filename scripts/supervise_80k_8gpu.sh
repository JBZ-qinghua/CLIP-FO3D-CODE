#!/usr/bin/env bash
set -uo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${project_root}/scripts/run_80k_8gpu.sh"
status_log="${project_root}/logs/train_mink16unet34c_8gpu_supervisor.log"
final_checkpoint="${project_root}/outputs/scannet/latest.pth"
max_attempts=4

mkdir -p "${project_root}/logs"
for attempt in $(seq 1 "${max_attempts}"); do
  printf '[%s] CLIP-FO3D 8-GPU run attempt %d/%d\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${attempt}" "${max_attempts}" \
    | tee -a "${status_log}"
  if bash "${runner}"; then
    if [[ -f "${final_checkpoint}" ]]; then
      printf '[%s] final 80K checkpoint verified\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "${status_log}"
      exit 0
    fi
    printf '[%s] runner exited successfully but the final checkpoint is missing\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "${status_log}"
  else
    status=$?
    if (( status == 75 )); then
      printf '[%s] training paused cleanly after writing an immediate checkpoint\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "${status_log}"
      exit 75
    fi
    printf '[%s] runner exited with status %d\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${status}" | tee -a "${status_log}"
  fi
  if (( attempt < max_attempts )); then
    sleep 60
  fi
done

printf '[%s] CLIP-FO3D 8-GPU pipeline failed after %d attempts\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${max_attempts}" | tee -a "${status_log}"
exit 1
