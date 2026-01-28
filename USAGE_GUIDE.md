# Species Distribution Modeling Codebase - Usage Guide

## Overview
This guide provides step-by-step instructions for setting up and using the Species Distribution Modeling (SDM) codebase. It covers installation, running main workflows, and tips for common tasks such as species/genus modeling, bias correction, and generating probability heatmaps.

---

## 1. Setup Instructions

### Prerequisites
- Python 3.7+
- pip (Python package manager)
- Earth Engine Python API (for feature extraction)
- Required Python packages (see below)

### Install Required Packages
Run the following command in your project directory:
```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install the main dependencies:
```bash
pip install numpy pandas scikit-learn scipy tqdm matplotlib shap joblib folium shapely geopy earthengine-api
```

### Earth Engine Authentication
To use the Earth Engine API for feature extraction:
```bash
earthengine authenticate
```
Follow the instructions in your browser to complete authentication.

---

## 2. Directory Structure

- `Modules/` : Contains main Python modules (custom loss trainers, feature extractor, etc.)
- `data/` : Input data (presence/absence CSVs, ecoregion polygons)
- `outputs/` : Output results (models, heatmaps, feature importance, etc.)
- `generate_probability_heatmap_species.py` : Script to generate probability heatmaps

---

## 3. Main Workflows

### A. Species-Level Modeling
Run comprehensive modeling for a species:
```python
from Modules.custom_loss_trainers import comprehensive_species_with_precomputed_features
results = comprehensive_species_with_precomputed_features("Artemisia nilagirica")
```

#### With Bias Correction
```python
results = comprehensive_species_with_precomputed_features("Artemisia nilagirica", bias_correction=True)
```

### B. Genus-Level Modeling
```python
from Modules.custom_loss_trainers import comprehensive_genus_with_precomputed_features
results = comprehensive_genus_with_precomputed_features("Artemisia")
```

#### With Bias Correction
```python
results = comprehensive_genus_with_precomputed_features("Artemisia", bias_correction=True)
```

### C. Ecoregion-Level Modeling
Species:
```python
from Modules.custom_loss_trainers import ecoregion_level_species_with_precomputed_features
results = ecoregion_level_species_with_precomputed_features("Artemisia nilagirica", bias_correction=True)
```
Genus:
```python
from Modules.custom_loss_trainers import ecoregion_level_genus_with_precomputed_features
results = ecoregion_level_genus_with_precomputed_features("Artemisia", bias_correction=True)
```

---

## 4. Generating Probability Heatmaps

Use the provided script to generate a probability heatmap for a species or genus:

1. Edit `generate_probability_heatmap_species.py`:
   - Set `SPECIES` or `GENUS` and `USE_GENUS` at the top of the script.
   - Ensure the corresponding model and feature files exist in `outputs/` and `data/`.

2. Run the script:
```bash
python generate_probability_heatmap_species.py
```

- The script will output PNG and NPY files in the `outputs/` directory.
- Elevation heatmaps are also generated if elevation data is available.

---

## 5. Feature Importance Analysis

To analyze feature importance for multiple species:
```python
from Modules.custom_loss_trainers import perform_feature_importance_for_all_species
species_list = ["Artemisia", "Artocarpus heterophyllus", "Azadirachta indica"]
importance_results = perform_feature_importance_for_all_species(species_list)
```

---

## 6. Tips & Troubleshooting

- **Earth Engine Errors**: If you see authentication errors, rerun `earthengine authenticate`.
- **Missing Data**: Ensure all required CSVs and model files are present in the correct directories.
- **Outputs**: All results, models, and plots are saved in the `outputs/` directory.
- **Bias Correction**: Enable by passing `bias_correction=True` to modeling functions.
- **Documentation**: See `Modules/custom_loss_trainers_documentation.md` for detailed API documentation.

---

## 7. Example Directory Layout
```
project_root/
├── Modules/
│   ├── custom_loss_trainers.py
│   └── features_extractor.py
├── data/
│   ├── presence_points_with_features.csv
│   ├── absence_points_all_india.csv
│   └── eco_regions_polygon/
├── outputs/
│   ├── best_model_Artemisia.joblib
│   ├── Artemisia_probability_heatmap.npy
│   └── Artemisia_probability_heatmap.png
├── generate_probability_heatmap_species.py
├── requirements.txt
└── USAGE_GUIDE.md
```

---

## 8. Getting Help
- For code-level questions, see the docstrings in `Modules/custom_loss_trainers.py` or the generated documentation files.
- For troubleshooting, check the error messages and ensure all dependencies are installed.
- For further assistance, contact the codebase maintainer or open an issue in your project tracker.