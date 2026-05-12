# AgriPath: A Systematic Exploration of Architectural Trade-offs for Crop Disease Classification

> **Accepted for Publication at Transactions of Machine Learning Research (TMLR), 2026 --- Camera-Ready Manuscript in progress**

[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/hamzamooraj99/AgriPath-LF16-30k)
[![W&B](https://img.shields.io/badge/Weights%20&%20Biases-Experiments-orange?logo=weightsandbiases)](https://wandb.ai/hhm2000-heriot-watt-university/AgriPath-Paper/overview)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-red?logo=arxiv)](https://arxiv.org/abs/2603.13354)
[![License](https://img.shields.io/github/license/hamzamooraj99/AgriPath-Publication)](LICENSE)

---

## Overview

**AgriPath** is a benchmark study for multi-class crop disease classification on a unified 65-class dataset spanning 16 crops and two imaging conditions: controlled laboratory images and real-world field images.

The repository covers three model families:

| Architecture Family | Representative Models |
|---|---|
| Convolutional Neural Networks (CNN) | ResNet-50, ConvNeXt-Tiny |
| Contrastive Vision-Language Models | CLIP, SigLIP |
| Generative Vision-Language Models (VLM) | SmolVLM-500M, Qwen2.5-VL-3B, Qwen2.5-VL-7B |

In addition to training and evaluation code, the repo now includes dataset preparation utilities, modular CNN training/evaluation scripts, and analysis notebooks used for figures and error analysis.

---

## Dataset

The primary benchmark dataset is hosted on Hugging Face:

```text
hamzamooraj99/AgriPath-LF16-30k
```

Related derived datasets used by parts of the pipeline:

- `hamzamooraj99/AgriPath-LF16-30k-LAB`
- `hamzamooraj99/AgriPath-LF16-30k-FIELD`
- `hamzamooraj99/AgriPath-CNN`

Core dataset properties:

- ~30,000 images
- 65 crop-disease classes
- 16 crops
- Two sources: `lab` and `field`
- Standard `train`, `validation`, and `test` splits
- Common fields include `image`, `crop`, `disease`, `source`, `crop_disease_label`, and `numeric_label`

---

## Repository Structure

```text
AgriPath-Publication/
|-- dataset_scripts/                        # Dataset modifiers and split-generation utilities
|   |-- clean_hash_leakage_test_split.py    # Removes flagged test leakage samples and pushes a cleaned dataset
|   |-- custom_labels.py                    # Adds crop_disease_label / numeric_label columns
|   |-- detect_leakage_imagehash.py         # Detects duplicate and near-duplicate split leakage with perceptual hashes
|   |-- downsampler_split.py                # Builds the LF16-30k benchmark from the larger source dataset
|   `-- lab_field_separator.py              # Creates LAB and FIELD dataset variants
|-- model_scripts/
|   |-- cnn/
|   |   |-- cnn_lightning.py          # Modular Lightning data module + CNN training entry point
|   |   `-- summary_writer.py         # Artifact-based CNN evaluation and per-class summary logging
|   |-- train/
|   |   |-- train_clip.py             # Linear-probe training for CLIP/SigLIP backbones
|   |   |-- train_peft.py             # SmolVLM LoRA fine-tuning with PEFT/TRL
|   |   |-- train_unsloth.py          # Qwen2.5-VL LoRA fine-tuning with Unsloth
|   |   |-- configs/                  # Final training configs by model family
|   |   `-- sweep_configs/            # W&B sweep configs for hyperparameter search
|   `-- eval/
|       |-- baseline_evaluator.py     # Random and majority-class baselines
|       |-- eval_clip.py              # Linear-probe evaluation for CLIP/SigLIP heads
|       |-- zs_eval_clip.py           # Zero-shot CLIP/SigLIP evaluation
|       |-- eval_peft.py              # SmolVLM LoRA / frozen-vision evaluation
|       |-- eval_unsloth.py           # Qwen2.5-VL LoRA / zero-shot evaluation
|       |-- configs/                  # Evaluation configs (current + legacy)
|       |-- helper_scripts/           # Small evaluation helpers
|       `-- run_zs_evals/             # Convenience shell launchers for zero-shot runs
|-- analysis/
|   |-- dataset_figures.ipynb         # Dataset statistics and paper figures
|   |-- cnn_inference.ipynb           # CNN inference and qualitative inspection
|   |-- clip_inference.ipynb          # CLIP/SigLIP inference analysis
|   |-- resnet50_lightning.py         # Legacy analysis snapshot of the older CNN script
|   |-- diagrams/                     # Saved plots and visual assets
|   |-- error analysis/               # Error CSVs, heatmaps, and confusion-matrix notebook
|   `-- parse_outputs/                # VLM output post-processing
|-- requirements.txt
`-- README.md
```

---

## Recent Codebase Updates

The CNN workflow has been reorganised to support a more modular setup:

- `model_scripts/cnn/resnet50_lightning.py` was renamed to `model_scripts/cnn/cnn_lightning.py`
- `cnn_lightning.py` now supports both `resnet50` and `convnext` backbones behind a shared Lightning module
- `summary_writer.py` was refactored to consume exported W&B checkpoint artifacts and evaluate all saved CNN experiments in a uniform way
- `dataset_scripts/` was added to document and reproduce the benchmark dataset preparation workflow

The older `analysis/resnet50_lightning.py` file is still present as a legacy analysis copy, but the maintained training entry point is `model_scripts/cnn/cnn_lightning.py`.

---

## Models and Training

### 1. CNN Baselines

The CNN pipeline is built around a reusable Lightning data module and a generic `CNNLightningModel` that accepts different torchvision backbones.

Currently supported backbones:

- `resnet50`
- `convnext`

Example training run:

```bash
python model_scripts/cnn/cnn_lightning.py --model resnet50 --data main --max_epochs 10
```

Other valid dataset targets:

- `--data lab`
- `--data field`

This script runs a small grid over batch sizes and learning rates, trains a checkpoint for each combination, and logs the resulting artifacts to Weights & Biases.

To evaluate logged CNN checkpoints and write summary tables:

```bash
python model_scripts/cnn/summary_writer.py --model resnet50 --exp main --org <wandb-entity> --artifact_version 0
```

> NOTE: Training and evaluation for ConvNeXt-Tiny is underway and not complete as yet.

### 2. CLIP / SigLIP

The contrastive pipeline supports both zero-shot evaluation and frozen-backbone linear probing.

Zero-shot example:

```bash
python model_scripts/eval/zs_eval_clip.py --checkpoint google/siglip-base-patch16-224 --model SigLIP
```

Linear-probe training example:

```bash
python model_scripts/train/train_clip.py --model google/siglip-base-patch16-224 --run_name SigLIP_google_patch16 --base SigLIP
```

Linear-probe evaluation example:

```bash
python model_scripts/eval/eval_clip.py --checkpoint google/siglip-base-patch16-224 --head_artifact <wandb-artifact> --lr 1e-3 --model_name SigLIP
```

### 3. Generative VLMs

The repo includes LoRA fine-tuning and evaluation pipelines for:

| Model | Training Backend |
|---|---|
| SmolVLM-500M-Instruct | PEFT / TRL |
| Qwen2.5-VL-3B-Instruct | Unsloth |
| Qwen2.5-VL-7B-Instruct | Unsloth |

Supported training regimes:

| Regime | Description | Typical Config Suffix |
|---|---|---|
| `full_lora` | Full dataset | `*_full_lora.yaml` |
| `lab_lora` | Lab-only training | `*_lab.yaml` |
| `field_lora` | Field-only training | `*_field.yaml` |
| `train_frozen_vision` | Language-side tuning with frozen vision layers | `*_fv.yaml` |

Example training runs:

```bash
python model_scripts/train/train_unsloth.py --config model_scripts/train/configs/qwen3/qwen3_full_lora.yaml
python model_scripts/train/train_peft.py --config model_scripts/train/configs/smol/smol_full_lora.yaml
```

Example evaluation runs:

```bash
python model_scripts/eval/eval_unsloth.py --config model_scripts/eval/configs/lora_evals/qwen3/qwen3_full/charmed.yaml
python model_scripts/eval/eval_peft.py --config model_scripts/eval/configs/lora_evals/smol/smol_full/v9.yaml
```

---

## Evaluation Protocol

All major model families are evaluated across three test views:

| Split | Description |
|---|---|
| `Main` | Full held-out test set |
| `Lab` | Controlled laboratory subset |
| `Field` | Real-world field subset |

Reported metrics include:

- Macro precision
- Macro recall
- Macro F1
- Balanced accuracy
- Per-class scores
- Confusion matrices

The repo also includes two simple baselines:

```bash
python model_scripts/eval/baseline_evaluator.py
```

---

## Dataset Preparation Utilities

The new `dataset_scripts/` directory contains the dataset construction helpers used to prepare benchmark-ready dataset variants:

- `downsampler_split.py`: downsamples the larger `AgriPath-LF16` source dataset into the balanced `AgriPath-LF16-30k` benchmark
- `lab_field_separator.py`: derives source-specific LAB and FIELD variants from the full benchmark
- `custom_labels.py`: adds `crop_disease_label` and `numeric_label` columns for downstream model pipelines
- `detect_leakage_imagehash.py`: uses perceptual image hashes to flag intra-split hard duplicates and train-to-validation/test near-duplicate leakage, writing the flagged pairs to CSV
- `clean_hash_leakage_test_split.py`: reads a leakage report, removes flagged test samples, preserves the train/validation splits, and pushes a cleaned dataset variant to Hugging Face Hub

These scripts are primarily data-engineering utilities and are meant to be run selectively when rebuilding or extending the dataset assets on Hugging Face Hub.

---

## Analysis Assets

The `analysis/` directory contains notebooks and saved outputs used for inspection and paper figures:

| Path | Purpose |
|---|---|
| `analysis/dataset_figures.ipynb` | Dataset distribution plots |
| `analysis/cnn_inference.ipynb` | CNN inference and failure inspection |
| `analysis/clip_inference.ipynb` | CLIP/SigLIP inference analysis |
| `analysis/error analysis/conf_mat.ipynb` | Confusion-matrix and error analysis |
| `analysis/parse_outputs/csv_fix.py` | Post-processing for generated VLM outputs |

---

## Installation

Install the pinned environment with:

```bash
pip install -r requirements.txt
```

Key dependencies include:

| Package | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `pytorch-lightning` | CNN and linear-probe training loops |
| `transformers` | Backbone loading, processors, and VLM support |
| `datasets` | Hugging Face dataset access |
| `peft` | LoRA adapters |
| `trl` | SFT training utilities |
| `unsloth` | Efficient Qwen VLM fine-tuning |
| `wandb` | Experiment tracking and artifact management |

> **Note:** several scripts assume a CUDA-capable GPU and a configured `WANDB_API_KEY`.

---

## Configuration

Most VLM training and evaluation entry points are config-driven. Example fields:

```yaml
model_name: unsloth/Qwen2.5-VL-3B-Instruct
run_name: Qwen2.5-VL_3B_FLoRA
trc: false
job_type: full_lora
r: 128
learning_rate: 1.5e-4
weight_decay: 0.05
save_repo: hamzamooraj99/AgriPath-Qwen3B-LoRA
```

Training configs live under `model_scripts/train/configs/` and evaluation configs live under `model_scripts/eval/configs/`.

---

## Experiment Tracking

Experiments are logged with Weights & Biases. Before running the artifact-aware scripts, export your API key:

```bash
export WANDB_API_KEY=<your_api_key>
```

Tracked assets include:

- CNN checkpoint bundles
- Linear-probe classifier heads
- LoRA adapters and evaluation summaries

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file in this repository.
