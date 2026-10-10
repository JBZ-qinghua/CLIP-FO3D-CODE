# CLIP-FO3D

**English** | [简体中文](README_zh-CN.md)

CLIP-FO3D training, evaluation, and open-vocabulary 3D point-cloud inference on ScanNet. The pipeline builds dense CLIP point-level teacher features from RGB-D views sampled every 10 frames and distills them into a sparse 3D network based on MinkowskiEngine. At inference time, the model uses only 3D coordinates—RGB images are not required.

## Features

- Strict ScanNet protocol with RGB-D frames sampled every 10 frames for offline teacher-target generation.
- Dense CLIP feature distillation into a 512-dimensional Mink16UNet34C backbone.
- Eight-GPU DDP training with periodic checkpoints, safe pause, and automatic resume.
- Evaluation on all 312 scenes in the ScanNet validation split.
- Open-vocabulary inference for individual scenes and custom class vocabularies.
- Pretrained ScanNet weights and a fixed evaluation record.

## Environment Setup

Requirements:

- Linux and Conda
- An NVIDIA driver compatible with CUDA 11.8
- Eight GPUs are recommended for training; fewer GPUs can be used for target extraction, evaluation, and inference.

~~~bash
git clone https://github.com/JBZ-qinghua/CLIP-FO3D-CODE.git CLIP-FO3D
cd CLIP-FO3D

PIP_INDEX_URL=https://pypi.org/simple \
  conda env create -p ./.conda -f environment.yml
conda activate "$PWD/.conda"

bash scripts/install_minkowski_engine.sh

PIP_INDEX_URL=https://pypi.org/simple \
  python -m pip install --no-build-isolation \
  "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

python -m pip install -e . --no-build-isolation --no-deps
python scripts/check_environment.py
~~~

Download the OpenAI CLIP ViT-B/16 weights:

~~~bash
mkdir -p weights
python -c "import clip; clip.load('ViT-B/16', device='cpu', download_root='weights')"
test -f weights/ViT-B-16.pt
~~~

## Data Preparation

Access to the raw ScanNet dataset must be requested under the official ScanNet terms. Training requires preprocessed point clouds, such as those exported with Pointcept, together with the raw RGB-D .sens sequences and meshes:

~~~text
<PROCESSED_SCANNET>/train/sceneXXXX_XX/{coord.npy,color.npy,normal.npy}
<PROCESSED_SCANNET>/val/sceneXXXX_XX/{coord.npy,segment20.npy}
<RAW_SCANNET>/scans/sceneXXXX_XX/sceneXXXX_XX.sens
<RAW_SCANNET>/scans/sceneXXXX_XX/sceneXXXX_XX_vh_clean_2.ply
~~~

Create project-local symbolic links, then export RGB-D views and point-to-pixel correspondences every 10 frames:

~~~bash
bash scripts/setup_links.sh <PROCESSED_SCANNET> <RAW_SCANNET> weights/ViT-B-16.pt

python scripts/prepare_scannet10.py \
  --raw-root data/scannet_raw \
  --point-root data/scannet \
  --output-root data/scannet10/images \
  --splits train val \
  --workers 32

python scripts/verify_protocol.py
~~~

The protocol check should report all 1,201 training scenes and 312 validation scenes as complete, with frame_gap set to 10.

## Teacher-Target Generation

Teacher targets are used only during training. Completed scenes are skipped, so interrupted extraction jobs can be restarted directly.

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/extract_targets_multi_gpu.sh train

python scripts/verify_protocol.py --check-targets
~~~

Teacher features are written to targets/scannet/train/*.npz. The integrity check must pass before training begins.

## Training

The default configuration runs for 80,000 steps with SGD, an initial learning rate of 0.8, and a learning-rate multiplier of 0.99 every 1,000 steps. The eight-GPU runner performs a DDP preflight check before training, evaluation, and result export:

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/run_80k_8gpu.sh
~~~

To launch DDP training only:

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train.py --distributed --batch-size 1 --workers 2
~~~

Full checkpoints are stored at outputs/scannet/checkpoint_step_*.pth. Each checkpoint contains the model, optimizer, learning-rate scheduler, training step, and data-sampler position. The runner resumes automatically from the complete checkpoint with the highest step number.

To pause a running job safely, execute the following command in another terminal:

~~~bash
bash scripts/request_training_pause.sh
~~~

All ranks finish the current step, write a resumable checkpoint, and exit. Restart scripts/run_80k_8gpu.sh to resume after a planned pause or unexpected interruption.

## ScanNet Validation Evaluation

Evaluation covers all 312 scenes in the official validation split, using 19 classes and excluding other furniture:

~~~bash
python scripts/evaluate.py \
  --checkpoint weights/clip_fo3d_scannet.pth \
  --point-root data/scannet \
  --config configs/inference.yaml \
  --output outputs/scannet_val.json
~~~

A fixed evaluation record is included at [results/scannet_val.json](results/scannet_val.json).

## Open-Vocabulary Inference

The input directory must contain at least coord.npy, an [N, 3] array of point coordinates in meters. Predictions correspond one-to-one with the input points.

~~~text
data/example_scene/
└── coord.npy
~~~

Run inference with the ScanNet 19-class vocabulary:

~~~bash
python scripts/infer.py \
  --scene-root data/example_scene \
  --checkpoint weights/clip_fo3d_scannet.pth \
  --config configs/inference.yaml \
  --output outputs/example_scene.npz
~~~

Use a custom class vocabulary:

~~~bash
python scripts/infer.py \
  --scene-root data/example_scene \
  --checkpoint weights/clip_fo3d_scannet.pth \
  --class-name wall --class-name chair --class-name monitor \
  --output outputs/custom_vocab.npz
~~~

The output NPZ contains prediction, confidence, and class_names. With --save-features, it also stores a 512-dimensional feature for each point.

## Pretrained Weights

The trained checkpoint weights/clip_fo3d_scannet.pth is approximately 145 MB and is tracked with Git LFS through .gitattributes.

After cloning the repository, install Git LFS and retrieve the checkpoint with:

~~~bash
git lfs install
git lfs pull
~~~

Datasets, teacher targets, training outputs, logs, and local Conda environments are excluded through .gitignore.

## Project Structure

~~~text
configs/       Training and inference configurations
scripts/       Data preparation, teacher extraction, training, evaluation, and inference
src/           Models, projection, distillation, and dataset code
tests/         Unit tests
weights/       Pretrained and trained model weights
results/       Fixed evaluation records
~~~

## Testing

~~~bash
python -m pytest -q
~~~

## License

This project is released under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for notices covering third-party components and model weights. Use of ScanNet data is subject to the ScanNet license.

## Citation

~~~bibtex
@inproceedings{zhang2023clipfo3d,
  title={CLIP-FO3D: Learning Free Open-World 3D Scene Representations from 2D Dense Features},
  author={Zhang, Cheng and others},
  booktitle={ICCV},
  year={2023}
}
~~~
