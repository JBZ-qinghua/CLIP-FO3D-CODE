#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stop_file="${project_root}/outputs/scannet/STOP_REQUESTED"

training_found=0
for process_dir in /proc/[0-9]*; do
  [[ "$(readlink -f "${process_dir}/cwd" 2>/dev/null || true)" == "${project_root}" ]] || continue
  while IFS= read -r -d '' argument; do
    if [[ "${argument}" == "scripts/train.py" || "${argument}" == "${project_root}/scripts/train.py" ]]; then
      training_found=1
      break 2
    fi
  done <"${process_dir}/cmdline" 2>/dev/null || true
done

if (( training_found == 0 )); then
  echo 'No CLIP-FO3D training worker is currently running in this project.' >&2
  exit 1
fi

mkdir -p "$(dirname "${stop_file}")"
touch "${stop_file}"
echo "Pause requested: ${stop_file}"
echo 'All ranks will finish the current step, save a resumable checkpoint, and exit.'
