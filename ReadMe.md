# **Species Distribution Modeling (SDM) Pipeline**

This pipeline helps in generating species distribution models (SDM) by utilizing presence points, pseudo-absence points, and various models. It provides flexibility to customize inputs and outputs for efficient SDM processing.

---

## **Modules Overview**
### **1. `Generate_Prob.py`**
- Contains functions to generate **probability distributions** from a model.
- Outputs the probability map as a `.tif` file.

### **2. `LULC_filter.py`**
- Implements **Land Use Land Cover (LULC) filtering**.
- Supports adding additional filters in the future.

### **3. `features_extractor.py`**
- Provides functions to **extract features** at specific points.
- Easily extensible for new features.

### **4. `models.py`**
- Contains various **models for training** and prediction.

### **5. `presence_dataloader.py`**
- Loads **presence points** after applying all necessary preprocessing steps.

### **6. `pseudo_absence_generator.py`**
- Generates **pseudo-absence points** within tree-covered regions.

### **7. `utility.py`**
- **Main Pipeline Function:** `fetch_gbif_and_extract_features()` - Fetches GBIF data and extracts WorldClim features
- **Forest Filtering:** `filter_points_by_forest_lulc()` - Filters presence points to forest regions using Dynamic World LULC
- **Index Calculations:** `calculate_concentration_index()`, `calculate_endemicity_index()` - Ecoregion distribution metrics
- **Spatial Utilities:** Polygon grid sampling, feature vector extraction, similarity calculations

---

## **Input Requirements**
Users must provide the following inputs in the `Inputs` folder:

1. **Polygon**: A file in **WKT (Well-Known Text)** format representing the region of interest.
2. **Genus Name**: The genus for which the SDM will be generated.
3. **Reliability Threshold**: A threshold value of **0.03** for **pseudo-absence generation** (filters out points too similar to presence locations).

---

## **Outputs**
1. **Model Evaluation Parameters**: Printed to the console.
2. **Probability Distribution File**: A `.tif` file named `Probability_Distribution.tif` saved in the `Outputs` folder.
   - The resolution can be adjusted in the `Generate_Prob.py` module.

---

## **Pipeline Workflow**
- The pipeline automates the process:
  1. **Fetch presence points** from GBIF using its API (reads genus name from `Inputs/genus_name.txt`)
  2. **Filter by forest regions** (optional) using Dynamic World LULC dataset
  3. **Extract environmental features** from WorldClim and other datasets
  4. **Normalize features** using min-max normalization (scaled to [0, 1])
  5. **Save results** to CSV file for further analysis
  6. **Calculate indices** (concentration, endemicity) using utility functions
  7. Optionally, users can download and place **presence points** manually in the `data` folder for faster processing
  8. Generates or uses provided **pseudo-absence points** (can also be added manually in the `data` folder)

**See `PIPELINE_DOCUMENTATION.md` for complete pipeline details.**

---

## **Notes**
- While the pipeline supports fetching data automatically, it is recommended to manually download presence points and pseudo-absence points for better speed and control.
- Flexibility is provided for both presence and pseudo-absence points:
  - Add them manually to the `data` folder.
  - Let the code generate them automatically.

## **How to use**
- Clone this repo in your project folder
- You will need to create a virtual environment in Python in your project folder
- Then install all the required libraries, using pip install -r requirements.txt 
- Then you will need to initialize a project on GEE, as we are using GEE API.
- In the main function write the name of your project in ee.initialize(project={your project name})
- Thanks