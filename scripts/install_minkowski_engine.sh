#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo 'Activate the project Conda environment before running this script.' >&2
  exit 1
fi

python_bin="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "Python not found under CONDA_PREFIX=${CONDA_PREFIX}" >&2
  exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compatibility_patch="${project_root}/scripts/minkowski_engine_torch2.patch"
if [[ ! -f "${compatibility_patch}" ]]; then
  echo "Required compatibility patch is missing: ${compatibility_patch}" >&2
  exit 1
fi

build_root="$(mktemp -d)"
cleanup() {
  rm -rf -- "${build_root}"
}
trap cleanup EXIT

git clone --quiet https://github.com/NVIDIA/MinkowskiEngine.git "${build_root}/MinkowskiEngine"
cd "${build_root}/MinkowskiEngine"
git checkout --quiet 9f81ae66b33b883cd08ee4f64d08cf633608b118
git apply "${compatibility_patch}"

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-12}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

"${python_bin}" setup.py bdist_wheel \
  --force_cuda \
  --blas=openblas \
  --blas_include_dirs="${CONDA_PREFIX}/include"
wheel_files=(dist/minkowskiengine-*.whl)
if [[ ${#wheel_files[@]} -ne 1 || ! -f "${wheel_files[0]}" ]]; then
  echo 'Expected exactly one MinkowskiEngine wheel after compilation.' >&2
  exit 1
fi
PIP_DISABLE_PIP_VERSION_CHECK=1 "${python_bin}" -m pip install \
  --no-deps --no-index "${wheel_files[0]}"
"${python_bin}" -c 'import MinkowskiEngine as ME; print("MinkowskiEngine", ME.__version__)'
