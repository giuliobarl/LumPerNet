# LumPerNet

This repository contains the code associated with the paper:

**Quantifying Perovskite Solar Cell Degradation via Machine Learning from Spatially Resolved Multimodal Luminescence Time Series**
Giulio Barletta, Simon Ternes, Saif Ali, Zohair Abbas, Chiara Ostendi, Marialucia D'Addio, Erica Magliano, Pietro Asinari, Eliodoro Chiavazzo, Aldo Di Carlo
[arXiv](https://arxiv.org/abs/2603.12857)

LumPerNet is a deep-learning pipeline for predicting the degradation state of perovskite solar cells from **spatially resolved luminescence imaging** and associated **JV measurements**.

The repository includes tools to:

- crop and organize cell-level ROIs from raw luminescence images,
- align imaging data with JV time series through a manifest,
- build per-cell multi-timepoint datasets,
- train a CNN-based regressor (`LumPerNet`) for $R_{PCE}$-related targets,
- compare against a simpler non-spatial baseline,
- inspect predicted degradation trajectories.

---

## Overview

The main goal of the project is to learn mappings from multimodal luminescence images to device-performance degradation targets, including:

- average PCE retention ($R_{PCE}$),
- $V_{oc}$ retention,
- $J_{sc}$ retention,
- $FF$ retention.

At dataset level, each sample corresponds to a **timepoint of a given cell**, while splitting is performed at the **cell level** to avoid timepoint leakage across train/validation/test folds.

---

## Repository structure

```text
LumPerNet/
├── GUI/
├── roi_cropping_pipeline.py
├── build_manifest_from_jv.py
├── build_dataset_from_manifest.py
├── dataset.py
├── models.py
├── cv_train_regressor.py
├── cv_train_baseline.py
├── inspect_trajectories.py
├── utils_data.py
├── utils_plot.py
├── setup.cfg
└── README.md
```

---

## Data and trained models

The datasets and trained models used to produce the results reported in the paper are available on Zenodo:

**Models and datasets for "Quantifying Perovskite Solar Cell Degradation via Machine Learning from Spatially Resolved Multimodal Luminescence Time Series"**
[Zenodo record](https://zenodo.org/records/18988274) DOI: `10.5281/zenodo.18988274`

If you use these resources, please cite the corresponding Zenodo record in addition to the associated publication.
