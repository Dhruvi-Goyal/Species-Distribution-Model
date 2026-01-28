# Species Distribution Modeling Pipeline - Complete Documentation

## Overview
This codebase implements a comprehensive Species Distribution Modeling (SDM) pipeline that:
1. Fetches species occurrence data from GBIF (Global Biodiversity Information Facility)
2. Filters presence points by forest regions using Dynamic World LULC
3. Extracts environmental features from WorldClim and other datasets
4. Trains machine learning models for species distribution prediction
5. Generates probability distribution maps

---

## Complete Pipeline Workflow

### Phase 1: Data Collection & Preprocessing

#### Step 1: Fetch GBIF Data and Extract Features
**Module:** `Modules/utility.py` → `fetch_gbif_and_extract_features()`

**Process:**
1. **Read Inputs:**
   - Reads genus name from `Inputs/genus_name.txt`
   - Reads polygon boundary from `Inputs/polygon.wkt`

2. **Fetch GBIF Data:**
   - Uses `Modules/presence_dataloader.py` → `Presence_dataloader` class
   - Queries GBIF API for all species under the specified genus
   - Filters by polygon boundary
   - Returns unique (longitude, latitude) points

3. **Filter by Forest Regions (Optional):**
   - Uses Dynamic World Land Use Land Cover (LULC) dataset
   - Retains only points with LULC label = 1 (trees/forest)
   - Function: `filter_points_by_forest_lulc()` in `utility.py`

4. **Extract Environmental Features:**
   - Uses `Modules/features_extractor.py` → `Feature_Extractor` class
   - Extracts features from WorldClim and other datasets:
     - **19 WorldClim bioclimatic variables** (bio01-bio19):
       - Temperature: annual mean, diurnal range, seasonality, extremes, quarterly means
       - Precipitation: annual, wettest/driest months/quarters, seasonality
     - **Additional features:**
       - `aridity_index`
       - `topsoil_ph`, `subsoil_ph`
       - `topsoil_texture`, `subsoil_texture`
       - `elevation` (from SRTM)

5. **Normalize Features:**
   - All features are normalized using **min-max normalization**
   - Formula: `normalized_value = (value - min_val) / (max_val - min_val)`
   - Normalization range computed from Malabar ecoregion (currently hardcoded)
   - Result: All values scaled to [0, 1] range

6. **Save Results:**
   - Saves to CSV file (default: `data/presence_points_with_features.csv`)
   - Format matches existing CSV structure with columns:
     - `longitude`, `latitude`
     - All 19 bioclimatic variables (normalized)
     - All additional features (normalized)

**Usage Example:**
```python
from Modules import utility
import ee

ee.Initialize()
output_file = utility.fetch_gbif_and_extract_features(
    ee=ee,
    max_points=2000,
    filter_forest=True,  # Filter to forest regions only
    output_file="data/presence_points_with_features.csv"
)
```

---

### Phase 2: Index Calculations

#### Step 2: Calculate Concentration and Endemicity Indices
**Module:** `Modules/utility.py` → `calculate_concentration_index()`, `calculate_endemicity_index()`

**Process:**
1. **Read presence points CSV** (from Step 1 or existing file)
2. **Load ecoregion polygons** from `data/eco_regions_polygon/` directory
3. **Calculate indices:**
   - **Concentration Index:** Number of ecoregions with presence / 48
   - **Endemicity Index:** Entropy-based measure of distribution concentration

**Usage Example:**
```python
from Modules import utility

# Using the file from Step 1
concentration_index, num_ecoregions, ecoregions_list = utility.calculate_concentration_index(
    presence_points_csv="data/presence_points_with_features.csv",
    ecoregion_wkt_directory="data/eco_regions_polygon"
)

endemicity_index = utility.calculate_endemicity_index(
    presence_points_csv="data/presence_points_with_features.csv",
    ecoregion_wkt_directory="data/eco_regions_polygon"
)
```

---

### Phase 3: Model Training (Optional - Advanced)

#### Step 3: Train Species Distribution Models
**Module:** `Modules/models.py` → `Models` class

**Process:**
1. Load presence and absence data
2. Extract or use pre-computed features
3. Train various ML models (Random Forest, XGBoost, etc.)
4. Evaluate models using multiple metrics
5. Generate probability distribution maps

---

## Module Descriptions

### Core Modules

1. **`presence_dataloader.py`**
   - Fetches species occurrence data from GBIF API
   - Handles pagination and rate limiting
   - Queries all species under a given genus
   - Returns unique coordinate points

2. **`features_extractor.py`**
   - Extracts environmental features from Earth Engine datasets
   - Sources: WorldClim, SRTM, soil datasets
   - Performs min-max normalization
   - Normalizes all features to [0, 1] range

3. **`utility.py`**
   - **Main Pipeline Functions:**
     - `fetch_gbif_and_extract_features()` - Complete data fetching and feature extraction
     - `filter_points_by_forest_lulc()` - Forest filtering using Dynamic World LULC
   - **Index Calculations:**
     - `calculate_concentration_index()` - Ecoregion distribution measure
     - `calculate_endemicity_index()` - Distribution entropy measure
   - **Utility Functions:**
     - `divide_polygon_to_grids()` - Spatial sampling
     - `representative_feature_vector_for_polygon()` - Polygon feature vectors

4. **`models.py`**
   - Machine learning models for SDM
   - Multiple algorithms and loss functions
   - Model evaluation and comparison

5. **`LULC_filter.py`**
   - Land Use Land Cover filtering
   - Uses Dynamic World dataset

6. **`pseudo_absence_generator.py`**
   - Generates pseudo-absence points
   - Uses reliability-based filtering

7. **`Generate_Prob.py`**
   - Generates probability distribution maps
   - Outputs GeoTIFF files

---

## Input Files Required

Place these files in the `Inputs/` directory:

1. **`genus_name.txt`**
   - Contains the genus name (e.g., "Mangifera", "Dalbergia")
   - One genus name per line

2. **`polygon.wkt`**
   - WKT (Well-Known Text) format polygon
   - Defines the geographic boundary for data collection
   - Can be MULTIPOLYGON format (will be converted to POLYGON)

3. **`reliability_threshold.txt`** (Optional)
   - Threshold for pseudo-absence generation
   - Default: 0.03

---

## Feature Normalization Details

### Normalization Process:
1. **Min-Max Values:** Computed from "Malabar Coast moist forests" ecoregion
2. **Formula:** `normalized = (value - min) / (max - min)`
3. **Result:** All features scaled to [0, 1] range
4. **Special Cases:**
   - If `max - min == 0`: normalized value = 0
   - Missing values (NaN) are preserved as NaN

### Feature List (27 total):
**WorldClim Bioclimatic Variables (19):**
- `annual_mean_temperature`
- `mean_diurnal_range`
- `isothermality`
- `temperature_seasonality`
- `max_temperature_warmest_month`
- `min_temperature_coldest_month`
- `temperature_annual_range`
- `mean_temperature_wettest_quarter`
- `mean_temperature_driest_quarter`
- `mean_temperature_warmest_quarter`
- `mean_temperature_coldest_quarter`
- `annual_precipitation`
- `precipitation_wettest_month`
- `precipitation_driest_month`
- `precipitation_seasonality`
- `precipitation_wettest_quarter`
- `precipitation_driest_quarter`
- `precipitation_warmest_quarter`
- `precipitation_coldest_quarter`

**Additional Features (7):**
- `aridity_index`
- `topsoil_ph`
- `subsoil_ph`
- `topsoil_texture`
- `subsoil_texture`
- `elevation`

**Coordinates (2):**
- `longitude`
- `latitude`

---

## Complete Example Workflow

```python
from Modules import utility
import ee

# Initialize Earth Engine
ee.Initialize(project='your-project-name')

# Step 1: Fetch GBIF data and extract features
print("Step 1: Fetching GBIF data and extracting features...")
output_file = utility.fetch_gbif_and_extract_features(
    ee=ee,
    max_points=2000,
    filter_forest=True,  # Filter to forest regions
    output_file="data/presence_points_with_features.csv"
)

# Step 2: Calculate indices
print("\nStep 2: Calculating concentration and endemicity indices...")
concentration_index, num_ecoregions, ecoregions_list = utility.calculate_concentration_index(
    presence_points_csv=output_file,
    ecoregion_wkt_directory="data/eco_regions_polygon"
)

endemicity_index = utility.calculate_endemicity_index(
    presence_points_csv=output_file,
    ecoregion_wkt_directory="data/eco_regions_polygon"
)

# Print results
print(f"\n{'='*60}")
print("RESULTS:")
print(f"{'='*60}")
print(f"Concentration Index: {concentration_index:.4f}")
print(f"Number of Ecoregions with Presence: {num_ecoregions}")
print(f"Endemicity Index: {endemicity_index:.4f}")
print(f"\nEcoregions with presence: {ecoregions_list}")
```

---

## Key Features

### Data Source Integration:
- ✅ **GBIF API** - Fetches species occurrence data
- ✅ **WorldClim** - Bioclimatic variables
- ✅ **Dynamic World** - Land Use Land Cover data
- ✅ **SRTM** - Elevation data
- ✅ **Soil Datasets** - pH and texture data

### Processing Features:
- ✅ **Multi-species support** - Fetches all species under a genus
- ✅ **Forest filtering** - Retains only forest regions using LULC
- ✅ **Feature normalization** - Min-max scaling to [0, 1]
- ✅ **Rate limiting** - Handles GBIF API rate limits
- ✅ **Error handling** - Robust error handling and retries

### Output:
- ✅ **CSV files** - Presence points with features
- ✅ **Index calculations** - Concentration and endemicity metrics
- ✅ **Compatible format** - Matches existing CSV structure

---

## Notes

1. **Normalization Region:** Currently uses Malabar ecoregion for min/max calculation. This may need adjustment for different study areas.

2. **Forest Filtering:** Optional but recommended for tree species. Set `filter_forest=False` if you want all habitat types.

3. **GBIF Rate Limits:** The code includes exponential backoff for rate limiting. Large datasets may take time.

4. **Earth Engine:** Requires authentication and initialization. Run `earthengine authenticate` first.

---

## Troubleshooting

### No GBIF Records Found:
- Check genus name spelling
- Verify polygon covers areas where species occur
- Check GBIF website directly: https://www.gbif.org

### Normalization Issues:
- Values outside [0, 1] may occur if your region exceeds Malabar range
- Consider computing min/max from your polygon region instead

### Polygon Format:
- Supports MULTIPOLYGON (converted to POLYGON)
- GBIF may have issues with complex geometries


