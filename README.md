<div align="center">
<h1 align="center">🌍 M2Change 🌍</h1>

<h3>Beyond Bi-temporal and Unimodal: A Multimodal-Temporal Benchmark for Building Damage Assessment in Conflict Zones</h3>

[Kan Wei]<sup>1</sup>, [Jing Yao]<sup>1 *</sup>, [Jiahui Cui]<sup>1</sup>, [Xinyu Zhao]<sup>1</sup>, [Lei Wang]<sup>1 *</sup>, [Gemine Vivone]<sup>2</sup>, [Pedram Ghamisi]<sup>3</sup>

<sup>1</sup> Aerospace Information Research Institute, Chinese Academy of Sciences  
<sup>2</sup> National Research Council, CNR-IMAA, Italy  
<sup>3</sup> Helmholtz-Zentrum Dresden-Rossendorf, Germany  

[![Paper](https://img.shields.io/badge/Paper-Information_Fusion-cyan)](https://doi.org/10.1016/j.inffus.2026.104402) [![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Weikan0425/M2Change) [![License](https://img.shields.io/badge/License-RAIL-green)](#ethics--license)

[**Overview**](#-overview) | [**To-Do List**](#-to-do-list) | [**Model Weights**](#-model-weights) | [**Get Started**](#-lets-get-started) | [**Citation**](#-citation) 

</div>

## 📝 To-Do List
We are continuously updating this repository. Here is our current progress:

| Status | Task | Description |
| :---: | :--- | :--- |
| ✅ | **Dataset Release** | Publicly release M2Change-CZ1 and M2Change-CZ2 on Hugging Face. |
| ✅ | **Model Release** | Open-source the core architecture of MTCNet (CAA & STAD modules). |
| ✅ | **Weights Release** | Provide pre-trained weights for MTCNet-Base on both datasets. |
| ✅ | **Inference Code** | Release the complete test pipeline and evaluation scripts (`Val.py`). |
| ⬜️ | **Training Code** | Release the full end-to-end training code (`Train.py`). |

---

## 🔭 Overview

Practical and accurate building damage assessment in conflict zones is a critical yet challenging task. Conventional change detection methods rely heavily on bi-temporal, homogeneous optical data, which is often unavailable post-event due to cloud cover or acquisition constraints. 

**M2Change** introduces a novel **multimodal, multi-temporal paradigm**:
* **The Challenge:** Bridging the massive modality gap between pre-event high-resolution (1m) optical imagery and post-event medium-resolution (10m) SAR time series.
* **The Dataset:** Features two distinct conflict scenarios: **M2Change-CZ1** (spatially extensive sparse damage) and **M2Change-CZ2** (densely populated concentrated destruction).
* **The Solution (MTCNet):** We propose the Multimodal-Temporal Coupling Network (MTCNet), featuring a **Change-aware Alignment (CAA)** module to bridge the modality gap, and a **Spatio-Temporal Attention Differentiation (STAD)** module to enrich difference features with deep temporal context.

<p align="center">
  <img src="./overview.png" alt="M2Change Dataset Overview" width="100%">
</p>

---

## 📦 Model Weights

We provide the pre-trained weights of our **MTCNet-Base** model for both subsets of the M2Change dataset. You can download them directly from the links below:

| Dataset | Model | F1-Score | IoU | Download Link |
| :---  | :---: | :---: | :---: | :---: |
| **M2Change-CZ1** | MTCNet-Base | 69.19% | 52.90% | [Baidu Netdisk](https://pan.baidu.com/s/1p5cxyYW1uy_5e3XB5lMErg?pwd=emax)/[Google Drive](https://drive.google.com/file/d/1tODgix16u4vooli_cn_yqaOKnJJX9Enc/view?usp=drive_link) |
| **M2Change-CZ2** | MTCNet-Base | 68.39% | 51.96% | [Baidu Netdisk](https://pan.baidu.com/s/1xGeO40tifCQekSbXezFFVw?pwd=rrm8)/[Google Drive](https://drive.google.com/file/d/1DIWsTPx_nRjS_J_qoppgfiu3SUbKdN3-/view?usp=drive_link)|

*(Note: Please place the downloaded `.pth` files in their respective directories as defined in `Config.py`.)*

---

## 🚀 Let's Get Started

### `A. Installation`
Clone this repository and install the required dependencies:

```bash
git clone [https://github.com/Weikan0425/M2Change.git](https://github.com/Weikan0425/M2Change.git)
cd M2Change

# Install dependencies
pip install -r requirements.txt

```

### `B. Data Preparation`

1. Download the M2Change dataset from [Hugging Face](https://huggingface.co/datasets/Kerwin0425/M2Change).
2. Organize the downloaded data following the directory structure expected by `Config.py`.



```text
MCD/DATA/
├── M2Change-CZ1/
│   ├── train/
│   └── test/
└── M2Change-CZ2/
    ├── train/
    └── test/
```

### `C. Configuration`

All hyperparameters, dataset paths, and saving directories are centralized in `Config.py`. You can modify parameters such as `batch_size`, `learning_rate`, and `n_times` (temporal sequence length) directly in this file before running the scripts.

### `D. Training`

You can train the model from scratch using the unified `Train.py` script. Specify the dataset using the `--dataset` argument:

```bash
# Train on M2Change-CZ1  using GPU 0
CUDA_VISIBLE_DEVICES=0 python Train.py --dataset M2Change-CZ1

# Train on M2Change-CZ2  using GPU 1
CUDA_VISIBLE_DEVICES=1 python Train.py --dataset M2Change-CZ2

```

*Tip: To run the training process in the background, use `nohup CUDA_VISIBLE_DEVICES=0 python Train.py --dataset M2Change-CZ1 > train_m2Change-cz1.log 2>&1 &`.*

### `E. Inference & Evaluation`

To evaluate the model or generate change maps using the pre-trained weights, run `Val.py`:

```bash
# Evaluate on M2Change-CZ1 dataset
CUDA_VISIBLE_DEVICES=0 python Val.py --dataset M2Change-CZ1

# Evaluate on M2Change-CZ2 dataset
CUDA_VISIBLE_DEVICES=1 python Val.py --dataset M2Change-CZ2

```

**Overriding Default Paths:** If you want to test a custom model checkpoint or save visualization results to a different folder without editing `Config.py`, use the optional flags:

```bash
python Val.py --dataset M2Change-CZ1 --model_path /path/to/custom_weights.pth --save_dir /path/to/save_results/

```

The script will automatically generate pixel-wise prediction masks (`.png`) and error maps (highlighting True Positives, False Positives, and False Negatives) in the specified output directory.

---

## 📜 Citation

If you find the M2Change dataset or MTCNet useful in your research, please consider citing our paper:

```bibtex
@article{wei2026beyond,
  title={Beyond bi-temporal and unimodal: A multimodal-temporal coupling network for change detection in conflict zones},
  author={Wei, Kan and Yao, Jing and Cui, Jiahui and Zhao, Xinyu and Wang, Lei and Vivone, Gemine and Ghamisi, Pedram},
  journal={Information Fusion},
  volume={135},
  pages={104402},
  year={2026},
  publisher={Elsevier}
}

```

## ⚖️ Ethics & License

This dataset and the associated models are released under a **Responsible AI License (RAIL)**, which strictly prohibits military, intelligence, or surveillance applications. It is fundamentally designed as a decision-support tool for area-level humanitarian assessment. Precise geolocations have been technically anonymized to align with the highest Responsible AI standards.

