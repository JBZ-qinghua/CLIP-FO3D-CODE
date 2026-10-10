#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 PROCESSED_SCANNET_ROOT RAW_SCANNET_ROOT [CLIP_VIT_B16_PT]" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
processed_root="$(realpath "$1")"
raw_root="$(realpath "$2")"

mkdir -p "${project_root}/data" "${project_root}/weights"
ln -sfn "${processed_root}" "${project_root}/data/scannet"
ln -sfn "${raw_root}" "${project_root}/data/scannet_raw"
if [[ $# -eq 3 ]]; then
  clip_checkpoint="$(realpath "$3")"
  ln -sfn "${clip_checkpoint}" "${project_root}/weights/ViT-B-16.pt"
fi

echo "Linked processed and raw ScanNet data into ${project_root}/data"
