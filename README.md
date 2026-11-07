# Project Overview

## Uncertainty-Guided Retinal Vessel Density Estimation and Reliability Analysis Using Deep Learning

### Introduction

Retinal vessel segmentation is an important task in retinal image analysis. It identifies blood vessels in fundus images and provides the basis for measuring vessel-related features.

However, overall segmentation performance does not fully describe how well a model preserves thin vessels. Missed thin vessels may affect measurements derived from the predicted vessel maps. These measurements include Vessel Density (VD), Centerline Density (CD), and Fractal Dimension (FD).

This study examines retinal vessel segmentation from both a segmentation and a measurement perspective. It investigates thin-vessel false negatives, their association with biomarker estimation errors, and the relationship between uncertainty signals and these errors.

### Research Focus

The study focuses on four connected areas:

- **Segmentation performance:** Comparison of different model configurations to select a suitable segmentation model.
- **Thin-vessel errors:** Measurement of thin-vessel false negatives and their association with errors in VD, CD, and FD.
- **Uncertainty analysis:** Evaluation of predictive entropy, TTA variance, TTA JSD, and entropy of the mean in relation to biomarker estimation errors.
- **Uncertainty-guided quality review:** Assessment of whether uncertainty-based image ranking can help identify images with higher biomarker errors under fixed review budgets.

TTA variance and TTA JSD are treated as view-disagreement signals. They are not interpreted as direct measures of epistemic uncertainty.

### Research Questions

The study is organized around five research questions:

| ID | Research Question |
|---|---|
| RQ1 | Which segmentation configuration provides the most suitable performance for the study? |
| RQ2 | How are thin-vessel false negatives associated with biomarker estimation errors? |
| RQ3 | How are predictive and TTA-based uncertainty signals associated with biomarker estimation errors? |
| RQ4 | Can uncertainty-guided image selection support quality review under fixed review budgets? |
| RQ5 | How does the selected model perform on an external retinal dataset? |

### Datasets

Two retinal fundus image datasets are used in this study.

| Dataset | Role | Images |
|---|---|---:|
| FIVES | Model development, validation, and held-out testing | 798 |
| CHASE_DB1 | External evaluation | 28 |

The FIVES dataset is divided into 419 training images, 179 validation images, and 200 held-out test images.

CHASE_DB1 is used for external evaluation. Its test set contains 8 images from 4 subjects. The first manual annotation is used as the primary reference for evaluation.

### Main Study Components

The experimental work follows a fixed sequence:

1. Compare five segmentation configurations using the FIVES validation set.
2. Evaluate the selected Core-v1 model on the held-out FIVES test set.
3. Examine the relationship between thin-vessel false negatives and biomarker estimation errors.
4. Analyze predictive and TTA-based uncertainty signals on the FIVES validation and test sets.
5. Evaluate uncertainty-guided image selection against random selection and assess a frozen review policy on the FIVES test set.
6. Perform external evaluation on CHASE_DB1 using the selected model.

The study uses saved experimental results and predefined evaluation protocols for its analysis. The external evaluation is descriptive because the CHASE_DB1 test set contains only eight images from four subjects.

### Scope of the Study

This research focuses on retinal vessel segmentation, image-level biomarker estimation errors, uncertainty analysis, and image-level quality review.

It does not evaluate clinical diagnosis or establish clinical decision-making performance. Correlations between uncertainty signals and biomarker errors are treated as associations, not evidence of causation.

The CHASE_DB1 results are reported as a limited external evaluation. They are not presented as proof of robust generalization.

### README Guide

The following sections describe the research in detail:

1. Research Background
2. Research Problem and Key Questions
3. Literature Review
4. Research Gap
5. Aim, Objectives, and Research Questions
6. Study Design and Experimental Setup
7. Model Training and Model Selection
8. Results and Analysis
9. Overall Findings
10. Limitations
11. Conclusion and Future Work
12. Reproducibility and Research Resources
13. References
