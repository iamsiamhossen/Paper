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

The following sections provide further details:

1. Research Background
2. Research Problem and Key Questions
3. Literature Review
4. Research Gap
5. Aim and Objectives
6. Study Design and Experimental Setup
7. Model Training and Model Selection
8. Results and Analysis
9. Overall Findings
10. Limitations
11. Conclusion and Future Work
12. Reproducibility and Research Resources
13. References
