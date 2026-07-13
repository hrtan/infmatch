# InfMatch: Dataset Distillation by Influence Matching

Official implementation of **Influence Matching (Inf-Match)** for dataset distillation.

Instead of aligning training *process* surrogates (per-step gradients or trajectories),
Inf-Match aligns the final *outcome* of training: it learns a compact synthetic set
whose influence on the converged model parameters matches that of the full dataset.

For the method, theory, and full experimental results, please refer to the paper:
**`InfMatch.pdf`** (see the [Citation](#citation) below). This README only covers how to
run the released code.

## Repository structure

```text
.
├── cifar10_ipc50.yaml      # example config (CIFAR-10, IPC=50)
├── pretrain/               # Stage 1: trajectory pretraining
│   ├── pretrain.py
│   └── run_pretrain.sh
├── condense/               # Stage 2: dataset condensation
│   ├── condense.py
│   └── run_condense.sh
└── evaluation/             # Stage 3: evaluate the distilled dataset
    ├── evaluation.py
    └── run_eval.sh
```

## Requirements

- Python 3.8+
- PyTorch (with CUDA) and `torchvision`
- `PyYAML`
- Multi-GPU training is launched via `torchrun`.

## Configuration

All settings (dataset, network, optimization, condensation, matching, etc.) are read
from a single YAML config. Before running, edit the paths in `cifar10_ipc50.yaml` to
match your environment:

- `dataset.data_dir` — path to the dataset
- `save_path.save_dir` — where results are written
- `save_path.pretrain_dir` — where pretrained trajectories / models are stored

## Usage

The pipeline runs in three stages. Each stage reads the same config file.

```bash
# Stage 1 — pretrain trajectories
cd pretrain
bash run_pretrain.sh

# Stage 2 — condense (distill) the dataset
cd ../condense
bash run_condense.sh                       # uses ../cifar10_ipc50.yaml by default
# or specify a config explicitly:
# bash run_condense.sh ../cifar10_ipc50.yaml

# Stage 3 — evaluate the distilled dataset
cd ../evaluation
bash run_eval.sh                           # auto-locates the latest condense run
# or evaluate a specific distilled file:
# bash run_eval.sh /abs/path/to/data_xxxxx.pt
```

You can override the GPUs and number of processes via environment variables, e.g.
`GPUS=0,1,2,3 NPROC=4 bash run_condense.sh`.

## Configs / TODO

Currently only the CIFAR-10 (IPC=50) config `cifar10_ipc50.yaml` is provided as a
working example.

- [ ] Additional configs (other datasets, IPC settings, and fusion variants)

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tan2026infmatch,
  title     = {Dataset Distillation by Influence Matching},
  author    = {Tan, Haoru and Wang, Wang and Wu, Sitong and Wu, Xiuzhe and Sun, Yang-Tian and Chang, Chirui and Zhang, Shaofeng and Qi, Xiaojuan},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
