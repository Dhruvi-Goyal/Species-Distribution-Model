# Custom Loss Trainers Module Documentation

## Overview

The `custom_loss_trainers` module provides comprehensive functionality for Species Distribution Modeling (SDM) using custom loss functions. It includes implementations of Random Forest, Logistic Regression, and Weighted Logistic Regression classifiers with Dice, Focal, and Tversky loss variants.

## Table of Contents

1. [Loss Functions](#loss-functions)
2. [Custom Classifiers](#custom-classifiers)
3. [Evaluation Functions](#evaluation-functions)
4. [Comprehensive Modeling Functions](#comprehensive-modeling-functions)
5. [Ecoregion Analysis Functions](#ecoregion-analysis-functions)
6. [Feature Importance Functions](#feature-importance-functions)
7. [Utility Functions](#utility-functions)

## Loss Functions

### `dice_loss(y_true, y_pred, smooth=1.0)`

**Purpose**: Implements Dice loss for binary classification.

**Parameters**:
- `y_true`: Ground truth labels (numpy array)
- `y_pred`: Predicted probabilities (numpy array)
- `smooth`: Smoothing factor to prevent division by zero (default: 1.0)

**Returns**: Dice loss value (float)

**Formula**: `1 - (2 * intersection + smooth) / (union + smooth)`

### `focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)`

**Purpose**: Implements Focal loss to handle class imbalance.

**Parameters**:
- `y_true`: Ground truth labels (numpy array)
- `y_pred`: Predicted probabilities (numpy array)
- `alpha`: Weight for the rare class (default: 0.25)
- `gamma`: Focusing parameter (default: 2.0)

**Returns**: Focal loss value (float)

### `tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1.0)`

**Purpose**: Implements Tversky loss, a generalization of Dice loss.

**Parameters**:
- `y_true`: Ground truth labels (numpy array)
- `y_pred`: Predicted probabilities (numpy array)
- `alpha`: Weight for false positives (default: 0.3)
- `beta`: Weight for false negatives (default: 0.7)
- `smooth`: Smoothing factor (default: 1.0)

**Returns**: Tversky loss value (float)

## Custom Classifiers

### `CustomLossRandomForest`

**Purpose**: Random Forest classifier with custom loss functions.

**Initialization**:
```python
clf = CustomLossRandomForest(n_estimators=100, loss_type='dice', loss_params=None)
```

**Parameters**:
- `n_estimators`: Number of trees in the forest (default: 100)
- `loss_type`: Loss function type ('dice', 'focal', 'tversky') (default: 'dice')
- `loss_params`: Dictionary of loss function parameters (default: None)

**Methods**:
- `fit(X, y, sample_weights=None, max_iter=100, restarts=10)`: Train the model
- `predict_proba(X)`: Predict class probabilities
- `predict(X)`: Predict class labels

### `CustomLossLogisticRegression`

**Purpose**: Logistic Regression with custom loss functions and L2 regularization.

**Initialization**:
```python
clf = CustomLossLogisticRegression(loss_type='dice', loss_params=None, max_iter=1000, lr=0.01, l2=1e-3)
```

**Parameters**:
- `loss_type`: Loss function type ('dice', 'focal', 'tversky') (default: 'dice')
- `loss_params`: Dictionary of loss function parameters (default: None)
- `max_iter`: Maximum iterations for optimization (default: 1000)
- `lr`: Learning rate (default: 0.01)
- `l2`: L2 regularization strength (default: 1e-3)

**Methods**:
- `fit(X, y, sample_weights=None)`: Train the model
- `predict_proba(X)`: Predict class probabilities
- `predict(X)`: Predict class labels

### `CustomLossWeightedLogisticRegression`

**Purpose**: Weighted Logistic Regression with custom loss functions.

**Initialization**:
```python
clf = CustomLossWeightedLogisticRegression(loss_type='dice', loss_params=None, max_iter=1000, lr=0.01, l2=1e-3)
```

**Parameters**: Same as `CustomLossLogisticRegression`

**Methods**: Same as `CustomLossLogisticRegression`

## Evaluation Functions

### `evaluate_summary(X, y, all_df=None, min_points=10, optimize_for='tpr', species_name=None, genus_name=None)`

**Purpose**: Comprehensive evaluation of models with multiple loss functions and metrics.

**Parameters**:
- `X`: Feature matrix
- `y`: Target labels
- `all_df`: Additional data for analysis (optional)
- `min_points`: Minimum points required for evaluation (default: 10)
- `optimize_for`: Metric to optimize for ('tpr', 'f1', 'auc') (default: 'tpr')
- `species_name`: Name of the species (optional)
- `genus_name`: Name of the genus (optional)

**Returns**: Dictionary containing evaluation results

### `evaluate_best_tversky(X, y, context="", optimize_for='f1')`

**Purpose**: Find the best Tversky loss parameters for given data.

**Parameters**:
- `X`: Feature matrix
- `y`: Target labels
- `context`: Context string for logging (default: "")
- `optimize_for`: Metric to optimize for (default: 'f1')

**Returns**: Dictionary with best parameters and results

### `check_data_quality(X, y, species_name="")`

**Purpose**: Analyze data quality and provide insights.

**Parameters**:
- `X`: Feature matrix
- `y`: Target labels
- `species_name`: Name of the species (default: "")

**Returns**: Dictionary with data quality metrics

### `calculate_feature_based_reliability(absence_point_features, presence_features_df, threshold=0.2, power_transform=None)`

**Purpose**: Calculate reliability scores for absence points based on feature similarity to presence points.

**Parameters**:
- `absence_point_features`: Features of absence points
- `presence_features_df`: Features of presence points
- `threshold`: Similarity threshold (default: 0.2)
- `power_transform`: Power transformation to apply (optional)

**Returns**: Reliability scores for absence points

## Comprehensive Modeling Functions

### `comprehensive_species_with_precomputed_features(species, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', reliability_threshold=0.07, bias_correction=False)`

**Purpose**: Comprehensive species-level modeling with precomputed features.

**Parameters**:
- `species`: Species name
- `features_csv_path`: Path to features CSV file (default: "data/presence_points_with_features.csv")
- `optimize_for`: Metric to optimize for (default: 'tpr')
- `reliability_threshold`: Threshold for reliability filtering (default: 0.07)
- `bias_correction`: Whether to apply bias correction (default: False)

**Returns**: Dictionary with modeling results

**Features**:
- Automatic absence point selection
- Multiple model types (Random Forest, Logistic Regression, Weighted Logistic Regression)
- Multiple loss functions (Dice, Focal, Tversky)
- Bias correction option for ecoregion-based weighting
- Comprehensive evaluation metrics

### `comprehensive_genus_with_precomputed_features(genus, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', reliability_threshold=0.07, bias_correction=False)`

**Purpose**: Comprehensive genus-level modeling with precomputed features.

**Parameters**: Same as `comprehensive_species_with_precomputed_features` but for genus

**Returns**: Dictionary with modeling results

### `comprehensive_species_modeling_with_ecoregion_testing(species, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', test_ecoregions=True, bias_correction=False)`

**Purpose**: Species modeling with additional ecoregion testing.

**Parameters**:
- `species`: Species name
- `features_csv_path`: Path to features CSV file
- `optimize_for`: Metric to optimize for (default: 'tpr')
- `test_ecoregions`: Whether to test on ecoregions (default: True)
- `bias_correction`: Whether to apply bias correction (default: False)

**Returns**: Dictionary with modeling and ecoregion testing results

### `comprehensive_genus_modeling_with_ecoregion_testing(genus, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', test_ecoregions=True, bias_correction=False)`

**Purpose**: Genus modeling with additional ecoregion testing.

**Parameters**: Same as `comprehensive_species_modeling_with_ecoregion_testing` but for genus

**Returns**: Dictionary with modeling and ecoregion testing results

## Ecoregion Analysis Functions

### `test_custom_model_on_all_ecoregions(trained_model, species_name, output_file=None, num_workers=16)`

**Purpose**: Test a trained model on all ecoregions in parallel.

**Parameters**:
- `trained_model`: Trained model object
- `species_name`: Name of the species
- `output_file`: Output file path (optional)
- `num_workers`: Number of parallel workers (default: 16)

**Returns**: Dictionary with ecoregion testing results

### `process_single_ecoregion_custom_model(filename, polygon_dir, trained_model, feature_extractor)`

**Purpose**: Process a single ecoregion with a custom model.

**Parameters**:
- `filename`: Ecoregion polygon filename
- `polygon_dir`: Directory containing polygon files
- `trained_model`: Trained model object
- `feature_extractor`: Feature extractor object

**Returns**: Average probability for the ecoregion

### `ecoregion_level_species_with_precomputed_features(species, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', bias_correction=False)`

**Purpose**: Ecoregion-level species modeling.

**Parameters**:
- `species`: Species name
- `features_csv_path`: Path to features CSV file
- `optimize_for`: Metric to optimize for (default: 'tpr')
- `bias_correction`: Whether to apply bias correction (default: False)

**Returns**: Dictionary with ecoregion-level modeling results

### `ecoregion_level_genus_with_precomputed_features(genus, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', bias_correction=False)`

**Purpose**: Ecoregion-level genus modeling.

**Parameters**: Same as `ecoregion_level_species_with_precomputed_features` but for genus

**Returns**: Dictionary with ecoregion-level modeling results

## Feature Importance Functions

### `perform_feature_importance_for_all_species(species_list, features_csv_path="data/presence_points_with_features.csv")`

**Purpose**: Perform feature importance analysis for multiple species.

**Parameters**:
- `species_list`: List of species names
- `features_csv_path`: Path to features CSV file

**Returns**: Dictionary with feature importance results for all species

### `perform_feature_importance_for_all_genera(genus_list, features_csv_path="data/presence_points_with_features.csv")`

**Purpose**: Perform feature importance analysis for multiple genera.

**Parameters**:
- `genus_list`: List of genus names
- `features_csv_path`: Path to features CSV file

**Returns**: Dictionary with feature importance results for all genera

### `perform_feature_importance_for_ecoregion_species(species_list, features_csv_path="data/presence_points_with_features.csv")`

**Purpose**: Perform feature importance analysis for ecoregion-level species.

**Parameters**:
- `species_list`: List of species names
- `features_csv_path`: Path to features CSV file

**Returns**: Dictionary with ecoregion-level feature importance results

### `perform_feature_importance_for_ecoregion_genera(genus_list, features_csv_path="data/presence_points_with_features.csv")`

**Purpose**: Perform feature importance analysis for ecoregion-level genera.

**Parameters**:
- `genus_list`: List of genus names
- `features_csv_path`: Path to features CSV file

**Returns**: Dictionary with ecoregion-level feature importance results

## Correlation Analysis Functions

### `compute_rank_correlation(prob_file, similarity_file, similarity_col, output_csv)`

**Purpose**: Compute rank correlation between probabilities and similarity measures.

**Parameters**:
- `prob_file`: File containing probability predictions
- `similarity_file`: File containing similarity measures
- `similarity_col`: Column name for similarity measure
- `output_csv`: Output CSV file path

**Returns**: None (saves results to CSV)

### `analyze_ecoregion_correlations(species_name, model_type="CustomLossRandomForest")`

**Purpose**: Analyze correlations between ecoregion probabilities and environmental similarity.

**Parameters**:
- `species_name`: Name of the species
- `model_type`: Type of model used (default: "CustomLossRandomForest")

**Returns**: Dictionary with correlation analysis results

### `comprehensive_analysis_with_correlations(species, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', test_ecoregions=True, analyze_correlations=True)`

**Purpose**: Comprehensive analysis including correlation analysis.

**Parameters**:
- `species`: Species name
- `features_csv_path`: Path to features CSV file
- `optimize_for`: Metric to optimize for (default: 'tpr')
- `test_ecoregions`: Whether to test on ecoregions (default: True)
- `analyze_correlations`: Whether to analyze correlations (default: True)

**Returns**: Dictionary with comprehensive analysis results

### `comprehensive_genus_analysis_with_correlations(genus, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', test_ecoregions=True, analyze_correlations=True)`

**Purpose**: Comprehensive genus analysis including correlation analysis.

**Parameters**: Same as `comprehensive_analysis_with_correlations` but for genus

**Returns**: Dictionary with comprehensive analysis results

### `batch_correlation_analysis(species_list=None, genus_list=None)`

**Purpose**: Perform batch correlation analysis for multiple species/genera.

**Parameters**:
- `species_list`: List of species names (optional)
- `genus_list`: List of genus names (optional)

**Returns**: Dictionary with batch correlation analysis results

## Utility Functions

### `visualize_presence_absence_points(presence_df, absence_df, species_name, output_file=None)`

**Purpose**: Create interactive map visualization of presence and absence points.

**Parameters**:
- `presence_df`: DataFrame containing presence points
- `absence_df`: DataFrame containing absence points
- `species_name`: Name of the species
- `output_file`: Output HTML file path (optional)

**Returns**: None (saves interactive map to HTML file)

### `get_coordinate_columns(df)`

**Purpose**: Get coordinate column names from a DataFrame.

**Parameters**:
- `df`: DataFrame to analyze

**Returns**: Tuple of (longitude_col, latitude_col)

## Bias Correction

The module includes optional bias correction functionality that:

1. **Computes Raw Weights**: For each ecoregion with occurrence count `c`, calculates `w_raw = 1/(c+1)`
2. **Normalizes Weights**: Scales raw weights to range [0.5, 1.5] using the formula:
   ```
   w = 0.5 + (w_raw - w_min)/(w_max - w_min)
   ```
3. **Applies Weights**: Uses normalized weights in model training to reduce sampling bias

Bias correction is available as an optional parameter (`bias_correction=False`) in:
- `comprehensive_species_with_precomputed_features`
- `comprehensive_genus_with_precomputed_features`
- `ecoregion_level_species_with_precomputed_features`
- `ecoregion_level_genus_with_precomputed_features`

## Usage Examples

### Basic Species Modeling
```python
from Modules.custom_loss_trainers import comprehensive_species_with_precomputed_features

# Model a species with default settings
results = comprehensive_species_with_precomputed_features("Artemisia")

# Model with bias correction
results = comprehensive_species_with_precomputed_features("Artemisia", bias_correction=True)
```

### Ecoregion-Level Modeling
```python
from Modules.custom_loss_trainers import ecoregion_level_species_with_precomputed_features

# Ecoregion-level modeling
results = ecoregion_level_species_with_precomputed_features("Artemisia", bias_correction=True)
```

### Feature Importance Analysis
```python
from Modules.custom_loss_trainers import perform_feature_importance_for_all_species

# Analyze feature importance for multiple species
species_list = ["Artemisia", "Artocarpus heterophyllus", "Azadirachta indica"]
importance_results = perform_feature_importance_for_all_species(species_list)
```

### Custom Loss Classifier
```python
from Modules.custom_loss_trainers import CustomLossRandomForest

# Create and train a custom loss classifier
clf = CustomLossRandomForest(loss_type='tversky', loss_params={'alpha': 0.3, 'beta': 0.7})
clf.fit(X_train, y_train, sample_weights=sample_weights)
predictions = clf.predict_proba(X_test)
```

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning algorithms
- `scipy`: Scientific computing
- `tqdm`: Progress bars
- `matplotlib`: Plotting
- `shap`: SHAP values for feature importance
- `joblib`: Model persistence
- `folium`: Interactive maps
- `shapely`: Geometric operations
- `geopy`: Geographic calculations

## File Structure

The module expects the following file structure:
```
data/
├── presence_points_with_features.csv
├── absence_points_all_india.csv
├── eco_regions_polygon/
│   ├── ecoregion1.wkt
│   ├── ecoregion2.wkt
│   └── ...
└── testing_SDM/
    └── absence_points_*.csv
```

## Output Structure

The module generates outputs in the following structure:
```
outputs/
├── extracted_features/
├── feature_distributions/
├── metrics_plots/
└── testing_SDM_out/
    ├── species_name/
    │   ├── shap_importance_scores.csv
    │   ├── shap_importance.png
    │   └── shap_summary.png
    └── ...
``` 