# Project Overview

## Uncertainty-Guided Retinal Vessel Density Estimation and Reliability Analysis Using Deep Learning

### Introduction

Retinal vessel segmentation identifies blood vessels in retinal fundus images. The quality of segmentation affects the measurement of vessel-related biomarkers, including Vessel Density (VD), Centerline Density (CD), and Fractal Dimension (FD).

However, overall segmentation metrics may not fully reflect how well a model detects thin vessels. Missed vessels can contribute to errors in these measurements. Uncertainty information may also help identify images that require further quality review.

This study investigates retinal vessel segmentation, thin-vessel recovery, biomarker estimation errors, and uncertainty-guided quality review.

### Research Questions

The study addresses two main research questions:

**RQ1:** Does native-resolution processing with global context improve retinal vessel segmentation and thin-vessel recovery?

**RQ2:** How are segmentation errors and uncertainty related to biomarker estimation errors, and can uncertainty help identify images for quality review?

### Research Focus

The study focuses on:

- Comparing segmentation configurations to examine the effect of native-resolution processing and global context.
- Examining thin-vessel false negatives and their relationship with biomarker estimation errors.
- Analyzing predictive and test-time augmentation (TTA)-based uncertainty signals.
- Evaluating uncertainty-guided image selection for quality review.
- Assessing the selected model on an external retinal dataset.

TTA variance and TTA Jensen–Shannon divergence (JSD) are treated as view-disagreement signals, not direct measures of epistemic uncertainty.

### Datasets

Two retinal fundus image datasets are used.

| Dataset | Purpose | Images |
|---|---|---:|
| FIVES | Model development, validation, and held-out testing | 798 |
| CHASE_DB1 | External evaluation | 28 |

The FIVES dataset contains 419 training images, 179 validation images, and 200 held-out test images.

CHASE_DB1 contains 28 images from 14 subjects. Its external test set contains 8 images from 4 subjects. The first manual annotation is used as the primary evaluation reference.

### Experimental Overview

The study follows these main stages:

1. Compare five segmentation configurations on the FIVES validation set.
2. Evaluate the selected Core-v1 model on the held-out FIVES test set.
3. Analyze thin-vessel false negatives and their relationship with VD, CD, and FD estimation errors.
4. Examine the relationships between uncertainty signals and biomarker estimation errors.
5. Evaluate uncertainty-guided image selection under fixed review budgets.
6. Assess the selected model on CHASE_DB1 without target-specific model tuning.

### Scope and Limitations

This study focuses on image-level segmentation performance, biomarker estimation errors, uncertainty analysis, and quality review.

It does not evaluate clinical diagnosis or clinical decision-making. Relationships between uncertainty and biomarker errors are interpreted as associations, not causal effects.

The CHASE_DB1 external test evaluation is limited to eight images from four subjects. Therefore, its results are reported descriptively and are not considered proof of robust generalization.

### README Guide

1. [Research Background](#1-research-background)
2. [Research Problem and Key Questions](#2-research-problem-and-key-questions)
3. [Literature Review](#3-literature-review)
4. [Research Gap](#4-research-gap)
5. [Objectives](#5-objectives)
6. [Study Design and Experimental Setup](#6-study-design-and-experimental-setup)
7. [Model Training and Model Selection](#7-model-training-and-model-selection)
8. [Results and Analysis](#8-results-and-analysis)
9. [Overall Findings](#9-overall-findings)
10. [Limitations](#10-limitations)
11. [Conclusion and Future Work](#11-conclusion-and-future-work)
12. [Reproducibility and Research Resources](#12-reproducibility-and-research-resources)
13. [References](#13-references)

# 1. Research Background

Retinal vessel segmentation identifies blood vessels in fundus images and provides the basis for analyzing retinal vascular structure. The resulting vessel maps can be used to estimate biomarkers such as Vessel Density (VD), Centerline Density (CD), and Fractal Dimension (FD).

However, thin vessels are often difficult to segment because of their small size and low contrast. Missing these vessels can affect the predicted vessel structure and introduce errors in biomarker estimation. Overall segmentation metrics may not fully capture these errors.

Image resolution and contextual information are also important in vessel segmentation. Native-resolution processing can preserve fine vessel details, while global context can provide information about the wider retinal structure.

Uncertainty signals provide additional information about prediction behaviour. They may help identify images where segmentation and biomarker estimates require closer review.

This study investigates native-resolution retinal vessel segmentation with global context, thin-vessel errors, biomarker estimation, and uncertainty-guided quality review.

# 2. Research Problem and Key Questions

## Research Problem

Retinal vessel segmentation models may miss thin vessels, even when their overall segmentation performance appears satisfactory. These errors can affect the estimation of retinal biomarkers such as Vessel Density (VD), Centerline Density (CD), and Fractal Dimension (FD).

Local image patches preserve fine vessel details but provide limited information about the wider retinal structure. This raises the question of whether adding aligned global context to native-resolution patches can improve segmentation.

Furthermore, segmentation performance alone does not indicate how uncertainty relates to biomarker estimation errors or whether it can help identify images that require quality review.

## Key Research Questions

**RQ1:** Does adding global context to native-resolution patches improve retinal vessel segmentation and thin-vessel recovery?

**RQ2:** How are segmentation errors and uncertainty related to biomarker estimation errors, and can uncertainty help identify images for quality review?

# 3. Literature Review

This section reviews 15 studies published between 2020 and 2026. The selected studies cover retinal vessel segmentation, thin-vessel recovery, contextual feature learning, vascular connectivity, biomarker reliability, and uncertainty-guided quality control. Five additional studies directly use the FIVES dataset.

## 3.1 Literature Matrix

| # | Paper | Model / Exact Method | Dataset and Image Count | Reported Numerical Results | Contribution | Limitation |
|---|---|---|---|---|---|---|
| **1** | [Pyramid-Net (2021)](https://doi.org/10.3389/fmed.2021.761050) | Pyramid-Net with Intra-layer Pyramid-scale Aggregation Blocks (IPABs), Pyramid Input Enhancement and Deep Pyramid Supervision. | DRIVE (40), STARE (20), CHASE-DB1 (28). | DRIVE: Sensitivity 82.38%, Accuracy 96.26%, AUC 98.32%. Thin-vessel accuracy: **91.64%**, vs. 86.91% for CE-Net. | Preserves multi-scale features for narrow-vessel detection and evaluates thin vessels separately. | Does not examine thin-vessel FN effects on VD, CD or FD errors. |
| **2** | [Multichannel DNN (2021)](https://doi.org/10.3389/fbioe.2021.697915) | Three U-Net learning branches for original, thin and thick vessels, followed by probability-map fusion. Width threshold: 2.2 pixels. | DRIVE (40), STARE (20), IOSTAR (30). | DRIVE: F1 82.47%, Sensitivity 81.40%. Thin-vessel Sensitivity: **76.77%**, Specificity 97.79%. | Uses vessel-width-specific supervision to improve thin- and thick-vessel segmentation. | Thin-vessel sensitivity was lower than the comparison method's 81.70%. No downstream biomarker-error analysis. |
| **3** | [Dynamic Deep Networks (2020)](https://doi.org/10.3389/fcomp.2020.00035) | Dynamic U-Net with targeted prediction using a second Mini-U-Net for ambiguous vessel regions. | DRIVE (40), STARE (20), CHASE-DB (28), AV-WIDE (30), VEVIO (16 frames and 16 mosaics); 150 images/frames/mosaics. | DRIVE: Precision 82.84%, Recall 82.35%, F1 82.59%, Accuracy 96.93%. | Uses dynamic class weighting and targeted refinement to detect faint and ambiguous vessels. | Two-stage processing requires an additional network. Remaining thin-vessel FN effects on biomarkers are not evaluated. |
| **4** | [Multi-Scale Residual Attention Network (2021)](https://doi.org/10.3390/sym13010024) | MRA-UNet with Residual Attention, Bottom Reconstruction and Spatial Activation modules. | DRIVE (40), CHASE (28), STARE (20). | On DRIVE, RA+BR improved F-measure by 2.1 percentage points over MUNet. Adding SA improved sensitivity by another 4.46 points. | Uses multi-scale attention and feature reconstruction to preserve small-vessel details. | Focuses on segmentation metrics, not biomarker reliability or uncertainty-guided review. |
| **5** | [Multi-teacher Knowledge Distillation (2025)](https://doi.org/10.1007/s13755-025-00356-4) | MTKD-UNet with separate original-, thin- and thick-vessel U-Net teachers and one student network. | DRIVE (40), CHASEDB1 (28), CHUAC (30), DCA1 (134); 232 images. | DRIVE: F1 80.74%; penalized variant 81.66%. CHASEDB1: F1 77.48%; penalized variant 79.95%. | Transfers vessel-specific knowledge from multiple teachers to a single student model. | Does not evaluate VD/CD/FD measurement errors or image-level uncertainty-based quality review. |
| **6** | [Retinal Fractal Dimension Repeatability (2022)](https://doi.org/10.3390/diagnostics12071769) | Automated SS-OCTA segmentation and box-counting fractal dimension measurement. Repeatability assessed using ICC. | 43 eyes from 43 healthy volunteers; 3 scans per participant. | Repeatability ICC: 0.722 (superficial) and 0.828 (deep). Reproducibility ICC: 0.651 and 0.363, respectively. | Shows that retinal FD measurements are affected by acquisition and observer variability. | Uses SS-OCTA rather than colour fundus images. Does not examine segmentation-induced FD errors. |
| **7** | [Ensuring a Connected Structure for Retinal Vessels (2023)](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Dulau_Ensuring_a_Connected_Structure_for_Retinal_Vessels_Deep-Learning_Segmentation_ICCVW_2023_paper.pdf) | Vessels Network Retrieval (VNR): Gap Filling and Rebranch or Remove post-processing. | DRIVE, DUALMODAL2019, HRF and LESAV; 137 images. | clDice-based predictions: mean connected components **174 → 1**, while Dice changed from 0.839 to 0.840. | Reduces vessel fragmentation while maintaining similar Dice performance. | Some reconnections may be anatomically incorrect, and genuine small components may be removed. |
| **8** | [MaskVSC (2025)](https://doi.org/10.1109/TMI.2025.3538336) | TopoMask, connectivity loss and Vision GNN post-processing for masked-vessel reconstruction. | Fundus: DRIVE (40), STARE (20), RITE (40), LES-AV (21). OCTA: ROSE-1 (117), OCTA-500 (500). 2PFM: 405 images. | DRIVE U-Net Dice 75.93% → **80.30%** with MaskVSC. clDice 78.06% → 84.57%. Betti-0 error 10.3 → 1.99. | Learns to reconstruct missing vessel segments and improves structural connectivity. | Primarily designed for 2D images; the authors identify limited disease diversity and 3D extension as future work. |
| **9** | [MVCN (2025)](https://doi.org/10.1016/j.compbiomed.2025.109982) | Multi-branch Vessel-shaped Convolution Network with MHEM, directional vessel convolutions and uncertainty-based vessel sub-labels. | DRIVE (40), CHASE_DB1 (28), STARE (20), HRF (45); 133 images. | AUC: 98.31–98.83%; Accuracy: 95.83–97.09% across four datasets. | Combines high-resolution features, vessel-shape information and epistemic uncertainty. | Does not evaluate per-image VD/CD/FD measurement errors or image-level review prioritization. |
| **10** | [SemTopNet (2026)](https://doi.org/10.3389/fcell.2026.1940436) | Transformer semantic-topology branch, centerline refinement and Monte Carlo Dropout uncertainty supervision. | OCTA: 120 CSVD patients and 40 healthy controls. Exact segmentation image count not separately reported. Biomarker cohort: 40 CSVD and 40 healthy participants. | Dice 0.8931, clDice 0.9027, Recall 0.9293, AUC 0.9567. | Combines uncertainty-aware segmentation, topology refinement and vascular biomarker analysis. | Uses OCTA rather than RGB fundus images. Does not link thin-vessel FN rates to per-image VD/CD/FD errors or fixed-budget review. |
| **11** | [Fadugba et al. — Segmentation Benchmark (2024)](https://arxiv.org/abs/2406.14994) | U-Net, FR-UNet, MA-Net, SA-UNet and W-Net; four loss functions: BCE, Dice, DiceBCE and clDice. | **FIVES (800):** 600 train, 200 test; resized to 512×512. | Best: FR-UNet + DiceBCE, Dice **90.37%**, MCC **89.80%**. U-Net + DiceBCE: Dice 90.15%. | Provides a controlled benchmark of five architectures and four loss functions on FIVES. | Image quality strongly affects segmentation results. Does not investigate biomarker measurement errors or uncertainty-guided review. |
| **12** | [Köhler et al. — Uncertainty-Guided Patch Review (2024)](https://proceedings.mlr.press/v250/kohler24a.html) | Five-seed FR-UNet ensemble, pixel-wise entropy and top-ranked non-overlapping 81×81 patches for manual review. | **FIVES (800):** 600 train, 120 validation, 200 test; resized to 512×512. | Initial test Dice: **0.887 ± 0.094**. Five-patch review increased median Dice by over 0.02. Average 3.2 patches/image reached a quality standard near 0.90. DSC estimator MAE: 0.02. | Uses uncertainty maps to select local segmentation regions for manual correction. | Assumes reviewed patches receive perfect correction. Focuses on Dice improvement rather than biomarker-specific image-level review. |
| **13** | [Vargas Quiros et al. — VascX Models (2025)](https://doi.org/10.1167/tvst.14.7.19) | Deep U-Net ensembles for vessel, artery-vein and optic-disc segmentation and fovea localization. | **FIVES (800)** included in model evaluation; development also uses more than 15 annotated datasets and Dutch study images. Feature agreement evaluated separately on 215 Rotterdam fundus images. | FIVES Dice: VascX 0.832, AutoMorph 0.824, LittleWNet 0.756. On Rotterdam, 14/24 features had significantly lower MAE than AutoMorph and 23/24 than LittleWNet. | Connects segmentation outputs with quantitative vascular-feature accuracy. | Feature MAE results come from Rotterdam, not FIVES. Does not examine our specific thin-FN versus VD/CD/FD error relationship. |
| **14** | [Wang et al. — LFU-Net (2026)](https://doi.org/10.1038/s41598-026-50642-8) | MobileNetV2, Frequency-Aware Encoder, hierarchical FreqTokenLoRA and Recursive Residual Attention Fusion. | **FIVES (800):** 600 train, independent 200 test. | Dice **85.00%**, IoU 73.90%, Sensitivity 81.90%, Faint Vessel Dice **77.50%**. AutoMorph Faint Vessel Dice: 66.50%. | Directly evaluates faint-vessel recovery using a separate width-based metric. | Faint-vessel evaluation uses a diameter-below-2-pixel definition. Does not evaluate downstream VD/CD/FD errors. |
| **15** | [Shu et al. — Morphology-Aware Reconstruction Distillation (2026)](https://doi.org/10.3389/fcell.2026.1825518) | MRD with Hierarchical Structure Fusion (HSF), Fuse Cross Stage Feature, Selective Fusion and Residual Gating. Uses decoder-focused knowledge distillation and Hierarchical Context Loss. | **FIVES (800)** at 2048×2048. Figure 8 describes 200 test cases; the exact train/test split is not specified in the dataset description. Also uses DRIVE and ROSE. | Dice **90.01%**, mIoU 82.84%, Hausdorff Distance 5.95, parameters 8.14M. MambaUNet Dice: 90.47%. | Preserves fine-vessel morphology and topology through lightweight decoder distillation. | Designed for U-shaped architectures. The authors identify cross-centre variability, 3D scaling and limited gains on simpler segmentation tasks as remaining issues. 

# 4. Research Gap

The reviewed studies show that retinal vessel segmentation, thin-vessel recovery, topology preservation, biomarker analysis and uncertainty-guided review have already received research attention. Recent FIVES-based work has also evaluated segmentation models, faint-vessel performance and uncertainty-based patch correction.

However, the reviewed FIVES studies do not establish the full relationship between **per-image thin-vessel false negatives and errors in Vessel Density (VD), Centerline Density (CD), and Fractal Dimension (FD)**. They also do not jointly evaluate whether predictive uncertainty and test-time augmentation disagreement can identify images with larger biomarker errors under fixed image-review budgets.

This study addresses these questions through a controlled segmentation comparison, thin-vessel error analysis, biomarker reliability assessment and uncertainty-guided image-level quality review using FIVES.
# 5. Objectives

### Objectives

1. Compare native-resolution segmentation with and without global context using a parameter-matched native-only control.
2. Measure thin-vessel false negatives and their relationship with VD, CD, and FD estimation errors.
3. Evaluate predictive entropy and six-view TTA disagreement for identifying high-error images under fixed review budgets.
4. Evaluate the selected model on the held-out FIVES test set and the external CHASE_DB1 dataset.

## 6. Study Design and Experimental Setup

### 6.1 Study Design

FIVES was used for model development and internal evaluation. Five segmentation configurations were compared on the validation set. The selected proposed model was then used for thin-vessel, biomarker, and uncertainty analyses. CHASE_DB1 was used for a limited external evaluation.

### 6.2 Dataset and Input Preparation

| Dataset | Images | Use |
|---|---:|---|
| FIVES | 798 | 419 training, 179 validation, 200 test |
| CHASE_DB1 | 8 selected images from 4 subjects | External evaluation |

- Original FIVES image size: 2048 × 2048 RGB.
- RGB normalization: [0, 1].
- Native patch size: 512 × 512; stride: 256; 49 patches per image.
- Total patches: 20,531 training and 8,771 validation.
- Proposed-model global view: 384 × 384.
- M0 input: resized full image at 512 × 512.

### 6.3 Segmentation Configurations

| Model | Configuration | Parameters | Threshold |
|---|---|---:|---:|
| M0 | Resized full-image U-Net | 117,361 | 0.65 |
| M1 | Native-resolution patch U-Net | 117,361 | 0.40 |
| M2 | Native patches with global context | 634,113 | 0.35 |
| Proposed Model | Multi-scale native and global context | 2,450,849 | 0.49 |
| Capacity-control | Parameter-matched native-only model | 2,449,585 | 0.47 |

The proposed model combines native features with coordinate-aligned global features through gated fusion. The capacity-control model has a similar parameter count but does not use a global-context branch.

![Proposed model architecture and inference workflow](assets/proposed_model_pipeline.png)

*Figure 1. Proposed model architecture and inference workflow.*

### 6.4 Training Setup

M0, M1, and M2 were trained for up to 20 epochs. The proposed model and capacity-control used a maximum of 70 epochs with early stopping.

| Setting | Proposed Model |
|---|---|
| Seed | 42 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Scheduler | CosineAnnealingLR (minimum LR: 0.00001) |
| Loss | 0.5 BCE + 0.5 soft Dice |
| Mixed precision | FP16 |
| Early stopping | 15 epochs without improvement |
| GPU | NVIDIA Tesla T4 |

- Brightness augmentation: alpha 0.9–1.1 and beta -0.05–0.05.
- The same sampled adjustment was applied to native and global views.
- Best proposed-model checkpoint: epoch 20; training stopped after epoch 35.

### 6.5 Prediction and Segmentation Evaluation

Patch probabilities were mapped to their original coordinates and combined by uniform overlap averaging. The fixed threshold was applied after full-image reconstruction. The proposed-model threshold (0.49) was selected on the validation set by maximizing pooled micro Dice.

Segmentation was evaluated using Dice, ROC-AUC, clDice, and thin-vessel recall. The validation set was used for configuration comparison; the held-out FIVES test set was reserved for final evaluation.

### 6.6 Thin-Vessel, Biomarker, and Uncertainty Analysis

- **Thin vessels:** Reference-mask pixels with an estimated vessel diameter of 4 pixels or less were classified as thin vessels. Diameter was propagated from the nearest skeleton using the Euclidean distance transform.
- **Biomarkers:** Vessel density (VD), centerline density (CD), and fractal dimension (FD) were calculated from reference and predicted masks. The FOV was defined as `max(R, G, B) > 7.65`. FD used box sizes of 2, 4, 8, 16, 32, 64, and 128 pixels.
- **Uncertainty signals:** Predictive entropy, entropy of the mean prediction, six-view TTA variance, and six-view TTA JSD were examined. The TTA views were identity, horizontal flip, vertical flip, and rotations of 90°, 180°, and 270°. Variance and JSD represent prediction disagreement across transformed views.
- **Image review:** Images were ranked by uncertainty. Review budgets of 9, 18, and 36 images were compared with random selection using 10,000 draws without replacement.

### 6.7 External Evaluation

The proposed model was evaluated on eight CHASE_DB1 images from four subjects. The first manual annotation was used as the reference mask. This provides a limited external evaluation.

