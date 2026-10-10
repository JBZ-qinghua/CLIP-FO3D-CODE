# CLIP-FO3D

CLIP-FO3D 的 ScanNet 训练、评测与开放词汇三维点云推理实现。项目先从每 10 帧抽取的 RGB-D 视图构建稠密 CLIP 点级教师特征，再将其蒸馏到 MinkowskiEngine 稀疏 3D 网络；最终推理只使用三维坐标，不需要 RGB 图像。

## 特性

- 严格按每 10 帧采样 ScanNet 原始 RGB-D 视频并构建离线教师目标；
- 使用 Dense CLIP 特征蒸馏训练 512 维 Mink16UNet34C；
- 提供 8 卡 DDP 训练、周期 checkpoint、安全暂停与自动续训；
- 支持 ScanNet 全部 312 个验证场景的评测；
- 支持单场景与自定义类别词表的开放词汇推理；
- 包含已训练的 ScanNet 权重与固定评测记录。

## 环境配置

要求 Linux、Conda、支持 CUDA 11.8 的 NVIDIA GPU 驱动。推荐使用 8 张 GPU 训练；单卡或少卡也可用于教师目标提取、评测和推理。

```bash
git clone <YOUR_REPOSITORY_URL> CLIP-FO3D
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
```

下载 OpenAI CLIP ViT-B/16 权重：

```bash
mkdir -p weights
python -c "import clip; clip.load('ViT-B/16', device='cpu', download_root='weights')"
test -f weights/ViT-B-16.pt
```

## 数据准备

ScanNet 原始数据需要按官方协议申请。训练需要 Pointcept 等预处理工具导出的点云，以及原始 RGB-D `.sens` 序列与网格：

```text
<PROCESSED_SCANNET>/train/sceneXXXX_XX/{coord.npy,color.npy,normal.npy}
<PROCESSED_SCANNET>/val/sceneXXXX_XX/{coord.npy,segment20.npy}
<RAW_SCANNET>/scans/sceneXXXX_XX/sceneXXXX_XX.sens
<RAW_SCANNET>/scans/sceneXXXX_XX/sceneXXXX_XX_vh_clean_2.ply
```

建立项目内软链接，并导出训练集、验证集的每 10 帧 RGB-D 视图与点像对应：

```bash
bash scripts/setup_links.sh <PROCESSED_SCANNET> <RAW_SCANNET> weights/ViT-B-16.pt

python scripts/prepare_scannet10.py \
  --raw-root data/scannet_raw \
  --point-root data/scannet \
  --output-root data/scannet10/images \
  --splits train val \
  --workers 32

python scripts/verify_protocol.py
```

协议检查应显示训练集 1,201 个场景和验证集 312 个场景均完整，且 `frame_gap` 为 10。

## 生成教师目标

教师目标只用于训练阶段。脚本会跳过已完成场景，因此可以中断后直接重新执行。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/extract_targets_multi_gpu.sh train

python scripts/verify_protocol.py --check-targets
```

教师特征写入 `targets/scannet/train/*.npz`。开始训练前必须通过完整性检查。

## 训练

默认训练配置为 80,000 steps、SGD、初始学习率 0.8，并且每 1,000 step 将学习率乘以 0.99。8 卡运行器会先执行一次 DDP 预检，然后训练、评测并保存结果：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/run_80k_8gpu.sh
```

仅启动 DDP 训练时：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train.py --distributed --batch-size 1 --workers 2
```

完整 checkpoint 位于 `outputs/scannet/checkpoint_step_*.pth`，其中包含模型、优化器、学习率调度器、step 和数据采样位置。运行器会自动选择编号最大的完整 checkpoint 续训。

若要安全暂停正在运行的训练，请在另一个终端执行：

```bash
bash scripts/request_training_pause.sh
```

所有 rank 会完成当前 step、写入可恢复 checkpoint 后退出。临时中断或机器异常后，重新运行 `scripts/run_80k_8gpu.sh` 即可恢复。

## ScanNet 验证集评测

评测遍历官方验证划分的全部 312 个场景，使用 19 个类别并排除 `other furniture`：

```bash
python scripts/evaluate.py \
  --checkpoint weights/clip_fo3d_scannet.pth \
  --point-root data/scannet \
  --config configs/inference.yaml \
  --output outputs/scannet_val.json
```

仓库中提供固定评测记录：[results/scannet_val.json](results/scannet_val.json)。

## 单场景开放词汇推理

输入目录至少需要 `coord.npy`，数组形状为 `[N, 3]`，坐标单位为米。预测与输入点一一对应。

```text
data/example_scene/
└── coord.npy
```

使用 ScanNet 19 类词表：

```bash
python scripts/infer.py \
  --scene-root data/example_scene \
  --checkpoint weights/clip_fo3d_scannet.pth \
  --config configs/inference.yaml \
  --output outputs/example_scene.npz
```

使用自定义类别词表：

```bash
python scripts/infer.py \
  --scene-root data/example_scene \
  --checkpoint weights/clip_fo3d_scannet.pth \
  --class-name wall --class-name chair --class-name monitor \
  --output outputs/custom_vocab.npz
```

输出 NPZ 包含 `prediction`、`confidence`、`class_names`；添加 `--save-features` 时还会写入每点 512 维特征。

## 预训练权重与 GitHub 发布

`weights/clip_fo3d_scannet.pth` 是训练完成的模型权重，约 145 MB，超过普通 GitHub 单文件大小限制。仓库已通过 `.gitattributes` 配置 Git LFS：

```bash
git lfs install
git add .gitattributes weights/clip_fo3d_scannet.pth
git add .
git commit -m "Release CLIP-FO3D"
git push
```

也可以将该权重作为 GitHub Release 附件发布。数据集、教师目标、训练输出、日志和本机 Conda 环境均已在 `.gitignore` 中排除。

## 项目结构

```text
configs/       训练与推理配置
scripts/       数据准备、教师目标、训练、评测与推理入口
src/           CLIP-FO3D 模型、投影、蒸馏与数据集代码
tests/         单元测试
weights/        预训练/训练完成权重
results/        固定评测记录
```

## 测试

```bash
python -m pytest -q
```

## License

本项目以 [MIT License](LICENSE) 发布。第三方组件与模型权重的说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。使用 ScanNet 数据须遵守其数据许可。

## Citation

```bibtex
@inproceedings{zhang2023clipfo3d,
  title={CLIP-FO3D: Learning Free Open-World 3D Scene Representations from 2D Dense Features},
  author={Zhang, Cheng and others},
  booktitle={ICCV},
  year={2023}
}
```

