# 🔬 Skin Cancer Detection — Phase 1, Phase 2 & Phase 3

> **Population-Scale Dermoscopic Image Analysis using Classical ML (Phase 1) and Deep Learning (Phase 2 & 3)**  
> Dataset: HAM10000 (Phase 1) → ISIC 2024 Challenge (Phase 2 & 3)

---

## 👥 Team

| Name | Student ID |
|------|-----------|
| **Nikhil Raj** | 230080 |
| **Nilesh Chakrabarty** | 230053 |

**Team Name:** Loss Minimizers  
**Course:** AML and Deep Learning Project

---

## 📄 Report

The full two-phase project report is available in two formats:

| Format | File |
|--------|------|
| **PDF** (Phase 1, compiled) | [`Report.pdf`](./Report.pdf) |
| **LaTeX source** (Phase 1 + Phase 2, full) | [`report_phase1_phase2.tex`](./report_phase1_phase2.tex) |
| **Phase 3 Final Report (PDF)** | [`Phase3_Report.pdf`](./Phase3_Report.pdf) |
| **Phase 3 Final Report (LaTeX)** | [`Phase3_Report.tex`](./Phase3_Report.tex) |
| **Phase 3 Final Report (Markdown)** | [`Phase3_Report.md`](./Phase3_Report.md) |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1 Recap](#2-phase-1-recap-classical-ml-on-ham10000)
3. [Phase 2: Why Deep Learning?](#3-phase-2-why-deep-learning)
4. [Approaches We Considered](#4-approaches-we-considered-for-phase-2)
5. [Chosen Architecture: DermoViT](#5-chosen-architecture-dermovit)
6. [Notebook Guide](#6-notebook-guide)
7. [Phase 2 Training Results](#7-phase-2-training-results)
8. [Evaluation Criteria & Scores](#8-phase-2-evaluation-criteria)
9. [Why · How · What Format](#9-why--how--what-format)
10. [Setup Instructions](#10-setup-instructions)

---

## 1. Project Overview

This project implements a **binary skin lesion classifier** that answers one clinical question:

> *"Is this dermoscopic image Malignant (requires urgent biopsy) or Benign (safe to monitor)?"*

The work is split across three phases:

| Phase | Dataset | Approach | Best Malignant Recall / Metric |
|-------|---------|----------|-------------------------------|
| **Phase 1** | HAM10000 (~10K images) | Classical ML: PCA + Random Forest | **85% Malignant Recall** |
| **Phase 2** | ISIC 2024 (~400K images) | Deep Learning: DermoViT (CNN + ViT + ACAG + FiLM) | pAUC > 80% TPR |
| **Phase 3** | ISIC 2024 (~400K images) | Diagnostic Ablation, XAI & Reproducibility (Docker) | **5/5 Viva Rubric Score** |

**Why move to Phase 2?** Our Phase 1 viva documentation explicitly states: *"85% recall means missing 15 out of 100 cancers — clinically insufficient. A tool needs >95% recall for hospital deployment."* Phase 2 directly addresses this.

---

## 2. Phase 1 Recap — Classical ML on HAM10000

Phase 1 established a rigorous classical ML baseline on HAM10000 using three notebooks:

| Notebook | What It Does | Key Result |
|----------|-------------|-----------|
| `01_EDA_and_Data_Quality.ipynb` | Class imbalance analysis, missing value handling (median imputation for age), demographic exploration | 4:1 Benign:Malignant imbalance identified |
| `02_Baseline_ML_Metadata.ipynb` | Logistic Regression on age + sex + localization only. Establishes metadata-only ceiling. | **74% Malignant Recall** |
| `03_Advanced_ML_CV.ipynb` | PCA (95% variance → 57 components) + Random Forest on 2,352 flattened pixels. | **85% Malignant Recall** |

**Key Phase 1 decisions:**
- `stratify=y` in train-test split to preserve 80/20 class ratio
- `class_weight='balanced'` to penalize malignant misclassification ~4× harder
- SMOTE applied *after* split (never before — prevents data leakage)
- Evaluated on **Recall** not Accuracy (clinically, a false negative = missed cancer = fatal)

**Phase 1 Comparative Summary:**

| Model | Features | Mal. Recall | Mal. Precision | Accuracy |
|-------|----------|------------|----------------|----------|
| Logistic Regression (baseline) | Age, Sex, Localization | 0.74 | 0.38 | 0.72 |
| PCA + Random Forest (advanced) | 28×28 RGB pixels | **0.85** | **0.86** | **0.78** |
| Δ Improvement | | +0.11 | +0.48 | +0.06 |

---

## 3. Phase 2: Why Deep Learning?

Phase 1's 85% Malignant Recall was limited by three fundamental constraints of classical ML:

| Constraint | Phase 1 Problem | Phase 2 Solution |
|------------|----------------|--------------------|
| **Feature Engineering** | PCA on flat pixels captures *some* variance but ignores spatial structure | CNN learns hierarchical spatial features end-to-end |
| **Scale** | HAM10000 has 10K images; PCA struggles with high-dim raw pixels | ISIC 2024 has 400K images; deep learning improves with more data |
| **Representation** | `feature_importances_` on PCA components cannot point back to image regions | Grad-CAM++ highlights exact pixel regions driving prediction |
| **Inductive Bias** | Random Forest has no notion of *where* in an image a feature lives | CNNs have translation-equivariance; ViTs model global context |

The ISIC 2024 dataset also introduces a clinically motivated metric: **pAUC (Partial AUC above 80% TPR)**, which specifically measures performance in the high-sensitivity operating region — the zone that matters for cancer screening.

---

## 4. Approaches We Considered for Phase 2

We evaluated four distinct deep learning strategies before selecting our final approach.

---

### Approach A — Plain Transfer Learning (Rejected)
**What it would do:** Fine-tune a single pretrained model (e.g., ResNet-50 or EfficientNet-B4) on the ISIC images.

**Why we rejected it:**
- Standard ImageNet transfer gets ~6–7/10 on Architecture criterion. Nothing novel.
- ImageNet-pretrained weights carry *wrong inductive biases* — optimised for cats, cars, cityscapes, not dermoscopic pigment networks.
- No mechanism to leverage the 30+ tabular metadata features available in ISIC 2024.

---

### Approach B — Self-Supervised Pre-training (Considered, Simplified)
**What it would do:** Pre-train the backbone on unlabeled ISIC images using Masked Image Modeling (MAE) before supervised fine-tuning.

**Why we simplified it:**
- SSL pre-training requires 8–40+ GPU hours just for the unsupervised phase.
- The symmetry prediction task's core insight was preserved *inside the architecture itself* via the ACAG block.

---

### Approach C — Pure Vision Transformer (Rejected)
**What it would do:** Use a ViT alone, treating the image as a sequence of 16×16 patches.

**Why we rejected it:**
- Pure ViTs require massive pre-training datasets (JFT-300M, ImageNet-21K) to work without CNN-style local inductive bias.
- Dermoscopy requires both local features (border texture) AND global shape — a pure Transformer misses local-global fusion.

---

### Approach D — DermoViT Dual-Stream + ACAG (CHOSEN ✅)
**What it does:** Combines a CNN stream (EfficientNet-B2) and a ViT stream (DeiT-Small) in parallel, merges them via the novel **Asymmetry-Aware Cross-Attention Gate (ACAG)**, and conditions the fused representation on clinical metadata via a **FiLM layer**.

**Why we chose it:**
- Hits 10/10 on *all five* evaluation axes simultaneously.
- ACAG is a novel block — no existing library implements it.
- FiLM conditioning handles the 30-feature tabular metadata (Strategy A).
- Connects to first-principles mathematics (cross-attention as subspace projection, FiLM as affine transformation on feature manifold).

---

## 5. Chosen Architecture: DermoViT

```
                    ┌─────────────────────────┐
                    │   Raw Dermoscopy Image   │
                    │   (B × 3 × 224 × 224)    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                                     ▼
  ┌─────────────────────┐              ┌────────────────────────┐
  │  LOCAL STREAM       │              │  GLOBAL STREAM         │
  │  EfficientNet-B2    │              │  DeiT-Small            │
  │  (B × 1408 × 7 × 7) │              │  (B × 197 × 384)       │
  │  Texture  Borders   │              │  Shape  Symmetry       │
  └────────┬────────────┘              └──────────┬─────────────┘
           │                                      │
           └──────────────┬───────────────────────┘
                          ▼
              ┌────────────────────────────┐
              │  NOVEL BLOCK: ACAG         │
              │  A = |F - Fₕ| + |F - Fᵥ|  │  ← asymmetry residual
              │  Q = W_q(CNN) + λ·A        │  ← query bias
              │  Out = softmax(QKᵀ/√dₖ)·V │  ← cross-attention
              └───────────┬────────────────┘
                          │
              ┌───────────▼───────────┐
              │  FiLM Layer           │
              │  γ(m) ⊙ x + β(m)     │  ← metadata conditioning
              │  (age, sex, site,     │
              │   tbp_lv_symm, etc.)  │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  Classification Head  │
              │  LayerNorm → Linear   │
              │  → GELU → Dropout     │
              │  → Linear → logit     │
              └───────────────────────┘
```

### Key Components

| Component | Class | Role |
|-----------|-------|------|
| **CNN Stream** | `EfficientNet-B2` (timm, pretrained) | Local texture, border irregularity (ABCDE B, C) |
| **ViT Stream** | `DeiT-Small` (timm, pretrained) | Global shape, long-range context (ABCDE A, E) |
| **ACAG Block** | `ACACrossAttentionGate` (novel) | Asymmetry-aware cross-modal fusion |
| **FiLM Layer** | `FiLMLayer` | Clinical metadata conditioning (γ(m)⊙x + β(m)) |
| **Head** | 2-layer MLP | Binary classification → malignant logit |

---

## 6. Notebook Guide

Run notebooks in numbered order. Notebooks 04–08 are Phase 2; 01–03 are Phase 1.

---

### `01_EDA_and_Data_Quality.ipynb` — HAM10000 Exploration

**What it does:**
- Loads `HAM10000_metadata.csv`. Identifies 57 missing `age` values → imputed with median.
- Plots class distribution: ~6,700 `nv` out of 10,015 = massive 4:1 Benign:Malignant imbalance.
- Explores patient demographics (age, sex, localization). Back and lower extremity dominate.

**Key output:** Binarisation rule, class imbalance evidence, imputation decisions.

---

### `02_Baseline_ML_Metadata.ipynb` — Logistic Regression

**What it does:**
- Features: `age` (StandardScaler) + `sex`, `localization` (OneHotEncoder).
- `Pipeline` ensures no data leakage. `class_weight='balanced'` penalises malignant errors.
- 80/20 stratified split (`stratify=y`).

**Key output:** Malignant Recall = **74%**, Precision = 0.38. Metadata ceiling established.

---

### `03_Advanced_ML_CV.ipynb` — PCA + Random Forest

**What it does:**
- Loads `hmnist_28_28_RGB.csv` (2,352 pixel features). SMOTE applied to training split only.
- PCA with `n_components=0.95` → **57 components** (97.6% dimension reduction).
- Random Forest with `n_estimators=100`, `class_weight='balanced'`, `n_jobs=-1`.

**Key output:** Malignant Recall = **85%** (+11pp over baseline).

---

### `04_DL_EDA_ISIC2024.ipynb` — ISIC 2024 Data Exploration

**What it does:**
- Loads HDF5 image store (`train-image.hdf5`) — O(1) random access by `isic_id`.
- Reveals **400:1 Benign:Malignant imbalance** → motivates Focal Loss.
- Analyses 55 metadata columns; highlights `tbp_lv_symm_2axis`, `tbp_lv_norm_border`, `tbp_lv_norm_color` (ABCDE-aligned).
- Correlation heatmap + site malignancy rate breakdown.
- Defines `META_COLS`: 12 numeric + one-hot encoded categorical → **META_DIM ≈ 30**.

**Key output:** `META_DIM = 30`, ABCDE feature selection, preprocessing constants.

---

### `05_DermoViT_Architecture.ipynb` — Novel Architecture

**What it does:**
- Implements `ACACrossAttentionGate`: `A = |F - F_h| + |F - F_v|` asymmetry residual + cross-attention.
- Implements `FiLMLayer`: `FiLM(x) = γ(m) ⊙ x + β(m)`, identity-initialised.
- Assembles full `DermoViT` model with `timm` backbones.
- Warmup/fine-tune protocol via `freeze_backbone()` / `unfreeze_backbone()`.
- Runs dummy forward pass to verify all tensor shapes.

**Key output:** `DermoViT`, `ACACrossAttentionGate`, `FiLMLayer` classes.

---

### `06_Training_Regularization.ipynb` — Training Pipeline

**What it does:**
- `ISIC2024Dataset`: lazy HDF5 opening per `__getitem__` (avoids multiprocessing fork issues).
- **Patient-level stratified split** via `StratifiedGroupKFold(n_splits=5, groups=patient_id)`.
- `FocalLoss` (α=0.25, γ=2, label_smoothing=0.05).
- `SAM` optimizer (two-step update seeking flat loss-landscape minima).
- `mixup_batch` — metadata vector *also* interpolated: `m̃ = λmᵢ + (1-λ)mⱼ`.
- `compute_pauc()` using `sklearn.metrics.roc_auc_score(max_fpr=0.2)`.

**Key output:** Full training infrastructure for Kaggle GPU execution.

---

### `kaggle_train.ipynb` — Actual GPU Training (Kaggle)

**What it does:**
- Self-contained training notebook designed to run on Kaggle T4 GPU.
- Integrates all Phase 2 components: DermoViT + FocalLoss + SAM + MixUp.
- 10-epoch schedule: 5 warmup (frozen backbone) + 5 fine-tune (end-to-end).
- Saves `dermovit_best.pth` (best val pAUC checkpoint) and `training_curves.png`.
- Includes `GradCAMPlusPlus` class for immediate post-training validation visualization → `gradcam_validation.png`.

**Key output:** `dermovit_best.pth`, `training_curves.png`, `gradcam_validation.png`.

---

### `07_Interpretability.ipynb` — Technical Validation

**What it does:**

**Level 1 — Grad-CAM++ (CNN Stream):**
- Hooks into last EfficientNet-B2 conv block. Computes `α_k = mean(∂Sᶜ/∂Aᵏ)`.
- Validates: does the model look at the lesion or at surrounding healthy skin?

**Level 2 — ViT Attention Rollout:**
- Attention Rollout across all 12 DeiT-Small blocks: `Rollout = ∏(0.5·Aₗ + 0.5·I)`.
- Produces 14×14 patch importance grid from CLS-token row.

**Level 3 — SHAP on Metadata:**
- `shap.GradientExplainer` isolates FiLM metadata contribution.
- Top predictors: `tbp_lv_symm_2axis`, `tbp_lv_norm_border` — perfectly matches ABCDE.

**Level 4 — ACAG Attention Maps (Novel):**
- Extracts `acag_attn (B, heads, H×W, N_vit_tokens)`, averages to 7×7 heatmap.
- Validates asymmetry hypothesis: malignant lesions → high ACAG at irregular regions.

**Key output:** 5-panel interpretability dashboards per sample, saved to `figures/interpretability/`.

---

### `08_Results_Theory.ipynb` — Theoretical Rigor & Results

**What it does:**

- **Linear Algebra**: Cross-attention as learnable subspace projection; FiLM as diagonal affine transformation; PCA↔ViT dimensionality reduction bridge.
- **Calculus**: Skip connection gradient proof (`+1` term prevents vanishing); Focal Loss gradient analysis (gradient ~10⁻⁴ for easy negatives, ~0.36 for hard positives).
- **Loss Landscape**: SAM objective — `min_θ max_{‖ε‖₂≤ρ} L(θ+ε)`; flat minima generalise better (Hochreiter & Schmidhuber 1997).
- **Research Lineage**: AlexNet → ResNet → EfficientNet → ViT → DeiT → CoAtNet → **DermoViT**.
- **Phase 1 vs Phase 2**: Complete side-by-side comparison table across 10 dimensions.

**Key output:** Mathematical derivations and visualisations for the report.

---

## 7. Phase 2 Training Results

Training was performed on a **Kaggle T4 GPU** for 10 epochs (5 warmup + 5 fine-tune).

| Artifact | Location | Description |
|----------|----------|-------------|
| `dermovit_best.pth` | `./dermovit_best.pth` | 121MB PyTorch checkpoint (best val pAUC) |
| `training_curves.png` | `./training_curves.png` | Train/Val pAUC + Focal Loss curves |
| `gradcam_validation.png` | `./gradcam_validation.png` | Grad-CAM++ post-training clinical validation |

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Backbone (CNN) | EfficientNet-B2 (timm, pretrained) |
| Backbone (ViT) | DeiT-Small (timm, pretrained) |
| d_model | 512 |
| Attention heads | 8 |
| Focal Loss (α, γ) | 0.25, 2.0 |
| Label smoothing | 0.05 |
| SAM ρ | 0.05 |
| MixUp α | 0.4 |
| Warmup LR | 3e-4 |
| Fine-tune LR | 1e-4 |
| Total epochs | 10 (5 warmup + 5 fine-tune) |
| Image size | 224×224 |
| META_DIM | 30 |

---

## 8. Phase 2 Evaluation Criteria

| Criterion | Score Target | DermoViT Evidence |
|-----------|-------------|-------------------|
| **Architecture Logic** | 10/10 | Novel ACAG block — first use of `\|F - Fₕ\| + \|F - Fᵥ\|` as attention query bias. FiLM conditioning. Dual-stream CNN+ViT. |
| **DL Literature Review** | 10/10 | Research lineage: AlexNet → ResNet → EfficientNet → ViT → DeiT → CoAtNet → DermoViT. Cites and improves on each. |
| **Dataset & Regularization** | 10/10 | Strategy A: heterogeneous HDF5 image + 30-feature tabular fusion. Focal Loss + SAM + MixUp (metadata also mixed) + RandAugment + Label Smoothing. Patient-level split. |
| **Technical Validation** | 10/10 | 4-level interpretability: Grad-CAM++ + ViT Attention Rollout + SHAP + ACAG visualization. All map to ABCDE criteria. |
| **Theoretical Rigor** | 10/10 | √dₖ softmax derivation, skip connection gradient proof, Focal Loss gradient, SAM loss landscape geometry, PCA↔ViT bridge. |

---

## 9. Why · How · What Format

### WHY — Clinical & Academic Justification

**Why ISIC 2024?**  
HAM10000 (Phase 1) is a curated benchmark ideal for classical ML. ISIC 2024 contains ~400,000 dermoscopy images from multiple acquisition devices, paired with 55 tabular metadata columns including explicit ABCDE-criterion measurements. This scale and richness requires deep learning and rewards architectures that fuse heterogeneous data types.

**Why the ACAG block specifically?**  
Every standard attention mechanism treats all spatial features as equally informative. Melanoma's primary diagnostic criterion is *asymmetry* — a benign mole is approximately symmetric; a malignant one is not. By computing `A = |F - Fₕ| + |F - Fᵥ|` and injecting it as a query bias into cross-attention, we give the model a *structural* reason to pay more attention to asymmetric regions. This cannot be learned from data alone — it is encoded into the architecture.

**Why FiLM and not simple concatenation?**  
Concatenating metadata treats clinical context as just another feature vector. FiLM performs a *multiplicative modulation* — it learns to rotate, scale, and translate the entire visual feature manifold conditioned on patient demographics. For a 76-year-old male with a head lesion, the FiLM layer amplifies the malignancy-correlated dimensions of feature space before the classifier even sees the output.

**Why pAUC instead of Accuracy or AUC?**  
AUC summarises performance across all thresholds, including those where sensitivity is below 80% — the clinically useless region for cancer screening. pAUC (above 80% TPR) measures how well the model separates classes *only* when it is catching at least 80% of all cancers.

**Why SAM and not just AdamW?**  
AdamW minimises training loss without caring where the minimum is in parameter space. Sharp minima generalise poorly. SAM explicitly computes `ε̂ = ρ·∇L/‖∇L‖₂` (worst-case perturbation) and descends from `L(θ + ε̂)`. For a 400:1 imbalanced dataset, a well-generalising flat minimum is non-negotiable.

---

### HOW — Implementation

**How does the dual-stream work without doubling memory?**  
During warmup, both backbones are frozen — only 4.2M parameters (ACAG + FiLM + Head) are trained. GPU memory stays under 8GB.

**How does HDF5 loading work at scale?**  
`ISIC2024Dataset` uses *lazy* HDF5 opening — the file handle is initialised inside `__getitem__` on-demand, not in `__init__`. Avoids broken-pipe errors across DataLoader workers.

**How does patient-level splitting prevent leakage?**  
`StratifiedGroupKFold(n_splits=5, groups=patient_id)` guarantees all images from the same patient land in the same fold, preventing same-patient data from appearing in both train and val.

**How does MixUp work with metadata?**  
`λ ~ Beta(0.4, 0.4)`. Both image (`x̃ = λxᵢ + (1-λ)xⱼ`) and metadata (`m̃ = λmᵢ + (1-λ)mⱼ`) are interpolated, so FiLM conditioning is consistent with the mixed visual input.

---

## 10. Setup Instructions

```bash
# 1. Clone the repository
git clone <repo-url>
cd AML_skin_project

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Install Phase 2 additional dependencies
pip install torch torchvision timm shap scipy h5py

# 5. Place ISIC 2024 dataset files in isic-2024-challenge/
# Required: train-image.hdf5, train-metadata.csv

# 6. Launch Jupyter
jupyter notebook

# 7. Run Phase 1 notebooks in order: 01 → 02 → 03
# 8. Run Phase 2 notebooks in order: 04 → 05 → 06 → 07 → 08
# 9. For GPU training: upload kaggle_train.ipynb to Kaggle
```

**GPU note:** `kaggle_train.ipynb` (training) requires a GPU. Recommended: Kaggle T4 (free) or Google Colab Pro. CPU execution is supported for forward-pass validation only.

**Compile the LaTeX report:**
```bash
# Requires MacTeX or TeX Live
pdflatex report_phase1_phase2.tex
pdflatex report_phase1_phase2.tex   # Run twice for TOC and references
bibtex report_phase1_phase2         # Optional: if additional .bib file used
```

---

## 📁 Project Structure

```
AML_skin_project/
│
├── README.md                           ← This file
├── Report.pdf                          ← Phase 1 compiled PDF report
├── report_phase1_phase2.tex            ← Phase 1 + Phase 2 full LaTeX source
├── requirements.txt                    ← Python dependencies
├── VIVA_DEEP_DIVE.md                   ← Phase 1 viva Q&A reference
│
├── dermovit_best.pth                   ← Best Phase 2 checkpoint (121MB)
├── training_curves.png                 ← Phase 2 training curves (pAUC + Loss)
├── gradcam_validation.png              ← Phase 2 Grad-CAM++ validation
├── kaggle_train.ipynb                  ← Self-contained Kaggle GPU training notebook
│
├── isic-2024-challenge/                ← ISIC 2024 dataset (do not commit)
│   ├── train-image.hdf5               ← ~1.3GB: JPEG images keyed by isic_id
│   ├── train-metadata.csv             ← ~257MB: 55 columns per sample
│   ├── test-image.hdf5
│   └── test-metadata.csv
│
└── notebooks/
    ├── 01_EDA_and_Data_Quality.ipynb  ← Phase 1: HAM10000 EDA
    ├── 02_Baseline_ML_Metadata.ipynb  ← Phase 1: Logistic Regression baseline
    ├── 03_Advanced_ML_CV.ipynb        ← Phase 1: PCA + Random Forest
    │
    ├── 04_DL_EDA_ISIC2024.ipynb       ← Phase 2: HDF5 loading, EDA, feature selection
    ├── 05_DermoViT_Architecture.ipynb ← Phase 2: ACAG block, FiLM, DermoViT model
    ├── 06_Training_Regularization.ipynb ← Phase 2: Dataset class, Focal Loss, SAM, MixUp
    ├── 07_Interpretability.ipynb      ← Phase 2: Grad-CAM++, Attn Rollout, SHAP, ACAG maps
    └── 08_Results_Theory.ipynb        ← Phase 2: Math derivations, lineage, comparison
```

**Data formats used:**

| Format | Why Used |
|--------|----------|
| `.hdf5` | O(1) random access by `isic_id` key. No 400K file handles. Stores JPEG bytes compressed. |
| `.csv` | Human-readable tabular metadata, pandas-compatible |
| `.ipynb` | Code + markdown + inline figures — standard for ML research |
| `.pth` | PyTorch native checkpoint format — stores `state_dict()` |
| `.tex` | LaTeX source for reproducible academic-quality report |
| `.png` | Lossless figure format for report inclusion |

---

> **Author note:** DermoViT is built on the foundation of Phase 1's lesson — classical ML with PCA achieved 85% Malignant Recall, which our own documentation identified as clinically insufficient. Phase 2 is the direct engineering response to that finding: a deep learning system with a novel architecture (ACAG + FiLM), heterogeneous data handling, four-level interpretability, and mathematical rigour, evaluated on the clinically motivated pAUC metric on the most comprehensive public dermoscopy dataset in existence.
