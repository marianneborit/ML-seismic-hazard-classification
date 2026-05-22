# Machine Learning-Based Seismic Hazard Classification and Spatial Clustering Analysis of Peruvian Earthquakes (2000–2025)

**Author:** Marianne Borit  
**Institution:** Universidad del Pacífico, Lima, Peru  
**Conference:** LACCI 2026 — Latin American Conference on Computational Intelligence

---

## Overview

This project applies machine learning to classify seismic hazard levels and identify spatial seismogenic zones in Peru using 25 years of USGS earthquake catalog data (2000–2025).

**Key results:**
- Random Forest achieved **86.7% accuracy** on three-class hazard classification (Low / Moderate / High)
- K-Means clustering identified **3 distinct seismogenic zones**, including a novel deep-focus cluster beneath the Peruvian Amazon (mean depth: 593 km)
- Regression baseline confirmed that direct magnitude prediction yields R² < 0.08, motivating the classification reformulation

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── seismic_hazard_peru.py       # Full pipeline as Python script
├── notebook.ipynb               # Same pipeline as Jupyter notebook
└── figures/                     # Output figures (generated on run)
    ├── Fig1_exploratory.png
    ├── Fig2_classification.png
    ├── Fig3_feature_importance.png
    ├── Fig4_clustering.png
    ├── Fig5_cluster_analysis.png
    └── Fig6_temporal.png
```

---

## Data

Data is fetched automatically from the **USGS Earthquake Catalog API** — no manual download required.

- **Source:** https://earthquake.usgs.gov/earthquakes/search/
- **Region:** Peru (lat: -18.5° to -0.5°, lon: -81.5° to -68.5°)
- **Period:** January 2000 – December 2025
- **Minimum magnitude:** Mw 3.0
- **Total events:** 6,951

---

## Methodology

### Feature Engineering
15 features derived from catalog attributes:
- Geographic coordinates (latitude, longitude)
- Focal depth and depth category (shallow / intermediate / deep)
- Euclidean distance to Nazca subduction reference point (-12.0°, -77.5°)
- Latitudinal zone (North / Central / South Peru)
- Zone seismic density
- Polynomial interaction terms (lat×lon, depth², dist_nazca²)
- Zone-level magnitude mean and standard deviation

### Hazard Classification
| Class | Magnitude Range |
|-------|----------------|
| Low | Mw 3.0 – 4.0 |
| Moderate | Mw 4.0 – 5.0 |
| High | Mw ≥ 5.0 |

Class imbalance (81.5% Moderate) addressed with **SMOTE**.

### Models Evaluated
- Random Forest (200 estimators)
- XGBoost (200 estimators, lr=0.05)
- Gradient Boosting (200 estimators)
- Ridge Regression, SVR (regression baseline)

---

## Results

### Classification
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Gradient Boosting | 68.2% | 0.68 |
| XGBoost | 74.9% | 0.74 |
| **Random Forest** | **86.7%** | **0.87** |

### Regression Baseline
| Model | RMSE | R² |
|-------|------|----|
| SVR | 0.4889 | -0.0018 |
| Ridge Regression | 0.4842 | 0.0173 |
| Gradient Boosting | 0.4711 | 0.0699 |
| Random Forest | 0.4697 | 0.0755 |

### Seismic Clusters
| Cluster | Zone | Mean Depth (km) | Mean Magnitude |
|---------|------|-----------------|----------------|
| 0 | Southern Peru | 76.9 | 4.47 |
| 1 | Northern Peru | 75.4 | 4.52 |
| 2 | Deep Amazon | 593.3 | 4.90 |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/seismic-hazard-peru.git
cd seismic-hazard-peru
pip install -r requirements.txt
```

---

## Usage

### Run as Python script
```bash
python seismic_hazard_peru.py
```

### Run as Jupyter notebook
```bash
jupyter notebook notebook.ipynb
```

### Run in Google Colab (no installation needed)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload `notebook.ipynb` to Google Colab and run all cells.

---

## Requirements

See `requirements.txt`. Main dependencies:
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- imbalanced-learn

---

## Reproducibility

All experiments use `random_state=42`. Data is fetched live from USGS API — results may vary slightly if the catalog is updated.

---

## AI Disclosure

The author used AI assistance (Claude, Anthropic) for code generation and writing support during the preparation of this manuscript.

---

## Status

This paper has been submitted to **LACCI 2026** (Latin American Conference on Computational Intelligence) and is currently under review. Citation will be added upon acceptance.

---

## License

MIT License — free to use, modify, and distribute with attribution.
