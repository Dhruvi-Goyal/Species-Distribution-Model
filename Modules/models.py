# Import necessary libraries for data manipulation, machine learning, and geospatial analysis
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.linear_model import LogisticRegression
from .features_extractor import Feature_Extractor
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import os
from .custom_losses import CustomNeuralNetwork, FocalLoss, DiceLoss, TverskyLoss, TverskyScorer, FocalScorer, DiceScorer, optimize_threshold_for_tpr
from . import features_extractor
from .feature_sensitivity_analysis import FeatureSensitivityAnalyzer
import ee
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Define the environmental feature columns used for species distribution modeling
# These include bioclimatic variables, soil properties, and elevation
feature_cols = [
    'annual_mean_temperature', 'mean_diurnal_range', 'isothermality',
    'temperature_seasonality', 'max_temperature_warmest_month', 'min_temperatur_coldest_month',
    'temperature_annual_range', 'mean_temperature_wettest_quarter', 'mean_temperature_driest_quarter',
    'mean_temperature_warmest_quarter', 'mean_temperature_coldest_quarter', 'annual_precipitation',
    'precipitation_wettest_month', 'precipitation_driest_month', 'precipitation_seasonality',
    'precipitation_wettest_quarter', 'precipitation_driest_quarter', 'precipitation_warmest_quarter',
    'precipitation_coldest_quarter', 'aridity_index', 'topsoil_ph', 'subsoil_ph', 'topsoil_texture',
    'subsoil_texture', 'elevation'
]



def visualize_presence_absence_points(presence_df, absence_df, species_name, output_file=None):
    """
    Visualize presence and absence points on an interactive map using folium.

    Parameters:
    -----------
    presence_df : DataFrame
        Presence points with longitude and latitude columns
    absence_df : DataFrame
        Absence points with longitude and latitude columns
    species_name : str
        Name of the species for the map title
    output_file : str, optional
        Output HTML file path (if None, auto-generates based on species name)

    Returns:
    --------
    str
        Path to the generated HTML file
    """
    try:
        import folium
        from folium import plugins
    except ImportError:
        print("folium not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "folium"])
        import folium
        from folium import plugins

    print(f"\nCreating interactive map for {species_name}...")

    # Auto-generate output file name if not provided
    if output_file is None:
        output_file = f'outputs/testing_SDM_out/{species_name.replace(" ", "_")}_presence_absence_map.html'

    # Ensure output directory exists

    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create directory if there is a directory path
        os.makedirs(output_dir, exist_ok=True)

    # Determine coordinate columns for presence and absence
    def get_coord_cols(df):
        if 'decimalLongitude' in df.columns and 'decimalLatitude' in df.columns:
            return ['decimalLongitude', 'decimalLatitude']
        elif 'longitude' in df.columns and 'latitude' in df.columns:
            return ['longitude', 'latitude']
        else:
            raise ValueError(f"DataFrame is missing required coordinate columns. Columns found: {list(df.columns)}")

    try:
        pres_coord_cols = get_coord_cols(presence_df)
        abs_coord_cols = get_coord_cols(absence_df)
    except Exception as e:
        print(f"Error: {e}")
        return None

    # Calculate center of the map
    all_lats = []
    all_lons = []

    if len(presence_df) > 0:
        try:
            lats = pd.to_numeric(presence_df[pres_coord_cols[1]], errors='coerce')
            lons = pd.to_numeric(presence_df[pres_coord_cols[0]], errors='coerce')
            # Only add valid numeric coordinates
            valid_mask = lats.notna() & lons.notna()
            all_lats.extend(lats[valid_mask].tolist())
            all_lons.extend(lons[valid_mask].tolist())
        except Exception as e:
            print(f"Error processing presence coordinates: {e}")

    if len(absence_df) > 0:
        try:
            lats = pd.to_numeric(absence_df[abs_coord_cols[1]], errors='coerce')
            lons = pd.to_numeric(absence_df[abs_coord_cols[0]], errors='coerce')
            # Only add valid numeric coordinates
            valid_mask = lats.notna() & lons.notna()
            all_lats.extend(lats[valid_mask].tolist())
            all_lons.extend(lons[valid_mask].tolist())
        except Exception as e:
            print(f"Error processing absence coordinates: {e}")

    if not all_lats or not all_lons:
        print("No valid coordinates found for mapping")
        return None

    # Ensure all coordinates are floats
    all_lats = [float(lat) for lat in all_lats if lat is not None]
    all_lons = [float(lon) for lon in all_lons if lon is not None]

    if not all_lats or not all_lons:
        print("No valid numeric coordinates found for mapping")
        return None

    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    # Create the map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )

    # Blue: ALL actual presence points from all_presence_point.csv
    presence_group = folium.FeatureGroup(name=f'Actual Presence (Blue, n={len(presence_df)})', overlay=True)
    for _, row in presence_df.iterrows():
        folium.CircleMarker(
            location=[row[pres_coord_cols[1]], row[pres_coord_cols[0]]],
            radius=2, color='blue', fill=True, fillColor='blue', fillOpacity=0.7, weight=1
        ).add_to(presence_group)
    presence_group.add_to(m)

    # Yellow: predicted presence (not actual)
    predicted_group = folium.FeatureGroup(name=f'Predicted Presence (Yellow, n={len(absence_df)})', overlay=True)
    for _, row in absence_df.iterrows():
        folium.CircleMarker(
            location=[row[abs_coord_cols[1]], row[abs_coord_cols[0]]],
            radius=2, color='yellow', fill=True, fillColor='yellow', fillOpacity=0.7, weight=1
        ).add_to(predicted_group)
    predicted_group.add_to(m)

    # Red: training absences only
    absence_group = folium.FeatureGroup(name=f'Training Absences (Red, n={len(absence_df)})', overlay=True)

    # Determine coordinate columns for training absence
    lon_col, lat_col = get_coordinate_columns(absence_df)
    if lon_col is None or lat_col is None:
        return None

    for _, row in absence_df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=1.5, color='red', fill=True, fillColor='red', fillOpacity=0.6, weight=1
        ).add_to(absence_group)
    absence_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_file)
    print(f"Interactive map saved to: {output_file}")
    return output_file


def get_coordinate_columns(df):
    """Helper function to determine coordinate column names in a DataFrame."""
    if 'longitude' in df.columns and 'latitude' in df.columns:
        return 'longitude', 'latitude'
    elif 'decimalLongitude' in df.columns and 'decimalLatitude' in df.columns:
        return 'decimalLongitude', 'decimalLatitude'
    else:
        print(f"Warning: Could not find coordinate columns in DataFrame")
        print(f"Available columns: {list(df.columns)}")
        return None, None



def calculate_feature_based_reliability(absence_point_features, presence_features_df, threshold=0.08, power_transform=None):
    """
    Calculate reliability score for an absence point based on feature similarity to presence points.
    Uses the same Gaussian kernel approach as in custom_loss_trainers.py.

    Args:
        absence_point_features: Feature vector of the absence point
        presence_features_df: DataFrame with presence point features
        threshold: Reliability threshold (0.2 by default, can be set to 0.03 or higher)
        power_transform: Power transformation exponent (if None, uses threshold value)

    Returns:
        float: Reliability score (0-1), higher means more reliable absence
    """
    # Calculate Euclidean distances between the absence point and all presence points


    distances = cdist([absence_point_features], presence_features_df, metric='euclidean')



    # Convert distances to similarities using Gaussian kernel (same as custom_loss_trainers.py)
    # The divisor (2 * number of features) normalizes for the feature space dimensionality
    # Compute similarities safely
    # presence_features_df.shape[1] = number of features (used for scaling)
    similarities = np.exp(-distances**2 / (2 * presence_features_df.shape[1]))

    # with open("similarities_output.txt", "a") as f:
    #     f.write("\n==============================\n")
    #     f.write("--- Absence Point Features ---\n")
    #     np.savetxt(f, [absence_point_features], fmt="%.6f", delimiter=",")

    #     f.write("\n--- Presence Features (matrix) ---\n")
    #     np.savetxt(f, presence_features_df, fmt="%.6f", delimiter=",")

    #     f.write("\n--- Similarities ---\n")
    #     np.savetxt(f, similarities, fmt="%.6f", delimiter=",")


    # Calculate mean similarity across all presence points
    mean_similarity = np.nanmean(similarities)

    # Reliability is inverse of similarity: less similar = more reliable as pseudo-absence
    reliability = 1 - mean_similarity

    # Apply power transformation using threshold value (or specified power_transform)
    if power_transform is None:
        power_transform = threshold  # Use threshold as power transform
    reliability = reliability ** power_transform

    return reliability

class Models:
    """
    A comprehensive class for species distribution modeling with various machine learning algorithms
    and advanced weighting schemes to handle class imbalance and spatial bias.
    """

    def __init__(self):
        """Initialize the Models class"""
        return

    def load_data(self, presence_path='data/presence.csv', absence_path='data/pseudo_absence.csv'):
        """
        Load and preprocess species presence/absence data with two types of sample weighting:
        1. Reliability-based weights for data quality
        2. Bias-correction weights based on ecoregion sampling density

        Parameters:
        -----------
        presence_path : str
            Path to CSV file containing species presence records
        absence_path : str
            Path to CSV file containing pseudo-absence records

        Returns:
        --------
        tuple
            X (features), y (labels), coords (coordinates), feature_cols (column names),
            reliability_weights, bias_weights
        """

        # ------------------------------------
        # Load presence and absence data
        # ------------------------------------
        presence_df = pd.read_csv(presence_path)
        absence_df = pd.read_csv(absence_path)
        # print('len',len(presence_df))
        # print('len',len(absence_df))

        # ------------------------------------
        # 1. Reliability-based Weights
        # ------------------------------------
        # For presence samples, set weight = 1 (assume high reliability)
        reliability_presence = np.ones(len(presence_df))

        # For absence samples, use the 'reliability' column if available
        if 'reliability' in absence_df.columns:
            absence_df['reliability'] = absence_df['reliability'].fillna(1)

        # Extract reliability values and normalize them to [0,1] range
        reliability = absence_df['reliability'].values
        min_rel = np.min(reliability)
        max_rel = np.max(reliability)
        if max_rel != min_rel:
            reliability_absence = (reliability - min_rel) / (max_rel - min_rel)
        else:
            reliability_absence = np.ones(len(reliability))

        # Apply a mild power transformation to reduce extremes in reliability weights
        reliability_absence = np.array([w**(0.2) for w in reliability_absence])

        # Combine reliability weights for presence and absence data
        reliability_weights = np.hstack([reliability_presence, reliability_absence])

        # ------------------------------------
        # 2. Bias-correction Weights
        # ------------------------------------
        # Read eco-region counts file to understand sampling density per ecoregion
        # This helps correct for geographic sampling bias
        counts_file = "outputs/testing_SDM_out/species_ecoregion_count_1.csv"
        if os.path.exists(counts_file):
            region_counts_df = pd.read_csv(counts_file)
            # print('region counts df length', len(region_counts_df))

            # Compute raw weight: inverse relationship with count (fewer samples = higher weight)
            region_counts_df['raw_weight'] = 1 / (region_counts_df['count'] + 1)

            # Normalize raw weights to a subtle range [0.5, 1.5] to avoid extreme values
            min_w = region_counts_df['raw_weight'].min()
            max_w = region_counts_df['raw_weight'].max()
            region_counts_df['eco_weight'] = 0.5 + region_counts_df['raw_weight']

            # Create mapping dictionary: eco_region -> eco_weight
            eco_weight_dict = region_counts_df.set_index('ecoregion')['eco_weight'].to_dict()
            # print("Eco-region weight mapping:", eco_weight_dict)
        else:
            print(f"Warning: {counts_file} not found. Defaulting eco weights to 1.")
            eco_weight_dict = {}

        # To assign bias weights, we need to determine which eco-region each point falls into
        # Since CSV files don't contain eco-region info, we perform spatial join with polygons
        ecoregion_folder = "data/eco_regions_polygon"

        # Convert lat/lon coordinates to Point geometries for spatial operations
        presence_df["geometry"] = presence_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        absence_df["geometry"] = absence_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)

        # Create GeoDataFrames with proper coordinate reference system (WGS84)
        presence_gdf = gpd.GeoDataFrame(presence_df, geometry="geometry", crs="EPSG:4326")
        absence_gdf = gpd.GeoDataFrame(absence_df, geometry="geometry", crs="EPSG:4326")

        # Combine presence and absence data while preserving original ordering
        combined_gdf = pd.concat([presence_gdf, absence_gdf], ignore_index=True)
        # print('len',len(presence_df))
        # print('len',len(absence_df))
        # print('len',len(combined_gdf))

        # Function to load eco-region polygons from WKT (Well-Known Text) files
        def load_ecoregions(folder):
            """Load ecoregion polygons from WKT files in the specified folder"""
            ecoregions = []
            for file in os.listdir(folder):
                if file.endswith(".wkt"):
                    with open(os.path.join(folder, file), "r") as f:
                        wkt_text = f.read().strip()
                        poly = wkt.loads(wkt_text)  # Parse WKT to geometry
                        ecoregions.append({"ecoregion": file.replace(".wkt", ""), "geometry": poly})
            return gpd.GeoDataFrame(ecoregions, geometry="geometry", crs="EPSG:4326")

        # Load eco-region polygons from WKT files
        ecoregion_gdf = load_ecoregions(ecoregion_folder)

        # Perform spatial join to assign each point to its corresponding eco-region
        # Uses "within" predicate to find which polygon contains each point
        combined_with_ecoregion = gpd.sjoin(combined_gdf, ecoregion_gdf, how="left", predicate="within")

        # Define function to retrieve bias weight for a given eco-region
        def get_bias_weight(eco):
            """Return bias weight for ecoregion, default to 1 if not found or NaN"""
            if pd.isna(eco):
                return 1
            else:
                return eco_weight_dict.get(eco, 1)

        # Apply the mapping to get bias weights for each data point
        bias_weights = combined_with_ecoregion['ecoregion'].apply(get_bias_weight).values

        # Save bias weights with coordinates to CSV for inspection and debugging
        coords_bias = np.column_stack((combined_with_ecoregion.geometry.x, combined_with_ecoregion.geometry.y))
        bias_df = pd.DataFrame(coords_bias, columns=["longitude", "latitude"])
        bias_df["bias_weight"] = bias_weights
        output_bias_file = "outputs/bias_weights.csv"
        os.makedirs(os.path.dirname(output_bias_file), exist_ok=True)
        # bias_df.to_csv(output_bias_file, index=False)
        # print(f"Bias weights saved to {output_bias_file}")

        # ------------------------------------
        # 3. Feature Extraction & Combination
        # ------------------------------------
        # Extract environmental features from the original CSV data
        presence_features = presence_df[feature_cols].values
        absence_features = absence_df[feature_cols].values

        # Combine features and create binary labels (1=presence, 0=absence)
        X = np.vstack([presence_features, absence_features])
        y = np.hstack([np.ones(len(presence_features)), np.zeros(len(absence_features))])

        # Extract coordinates from the combined GeoDataFrame (maintaining order)
        coords = np.column_stack((combined_with_ecoregion.geometry.x, combined_with_ecoregion.geometry.y))

        # ------------------------------------
        # Filter out points with NaN values BEFORE calculating weights
        # ------------------------------------
        # Find rows without NaN values in features
        valid_mask = ~np.isnan(X).any(axis=1)

        if not valid_mask.all():
            print(f"   Filtering out {np.sum(~valid_mask)} points with NaN values before weight calculation")
            print(f"   Keeping {np.sum(valid_mask)} valid points for modeling")

        # Apply mask to all arrays
        X = X[valid_mask]
        y = y[valid_mask]
        coords = coords[valid_mask]
        reliability_weights = reliability_weights[valid_mask]
        bias_weights = bias_weights[valid_mask]

        # ------------------------------------
        # Shuffle the data along with both sets of weights
        # ------------------------------------
        # Randomly shuffle all arrays together to ensure proper randomization
        X, y, coords, reliability_weights, bias_weights = shuffle(
            X, y, coords, reliability_weights, bias_weights, random_state=42
        )

        # Final safety check: ensure all weights are valid numbers
        if np.isnan(reliability_weights).any() or np.isinf(reliability_weights).any():
            print(f"   Warning: Found invalid values in reliability weights after filtering. Replacing with 1.0")
            reliability_weights = np.nan_to_num(reliability_weights, nan=1.0, posinf=1.0, neginf=1.0)

        if np.isnan(bias_weights).any() or np.isinf(bias_weights).any():
            print(f"   Warning: Found invalid values in bias weights after filtering. Replacing with 1.0")
            bias_weights = np.nan_to_num(bias_weights, nan=1.0, posinf=1.0, neginf=1.0)

        # Return all processed data components
        return X, y, coords, feature_cols, reliability_weights, bias_weights


    def RandomForest(self, X, y, sample_weights=None):
        """
        Train a Random Forest classifier with optional sample weighting.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training

        Returns:
        --------
        tuple
            Trained classifier, test features, test labels, predictions, probabilities
        """
        # Split data and track indices to properly subset sample weights
        indices = np.arange(len(y))
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X, y, indices, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize Random Forest classifier with 100 trees
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train with sample weights if provided
        if sample_weights is not None:
            # Ensure sample_weights is a 1D array
            sample_weights = np.ravel(sample_weights)
            # Subset the weights using the training indices
            weights_train = sample_weights[indices_train]
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)

        # Generate predictions and class probabilities
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class

        return clf, X_test, y_test, y_pred, y_proba

    def logistic_regression_L2(self, X, y):
        """
        Train a Logistic Regression model with L2 regularization.
        Includes data cleaning to handle infinite/NaN values.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels

        Returns:
        --------
        tuple
            Trained classifier, test features, test labels, predictions, probabilities
        """
        # Convert to numpy arrays and filter out rows with infinite/NaN values
        X = np.array(X, dtype=float)
        mask = np.isfinite(X).all(axis=1)  # Keep only finite values
        X, y = X[mask], y[mask]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Additional filtering for training and test sets
        train_mask = np.isfinite(X_train).all(axis=1)
        test_mask = np.isfinite(X_test).all(axis=1)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]

        # Initialize and train Logistic Regression with L2 penalty
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
        clf.fit(X_train, y_train)

        # Generate predictions and probabilities
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        return clf, X_test, y_test, y_pred, y_proba

    def train_and_evaluate_model_logistic_weighted(self, X, y, sample_weights=None):
        """
        Train a weighted Logistic Regression model with comprehensive data preprocessing.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training

        Returns:
        --------
        tuple
            Trained classifier, test features, test labels, predictions, probabilities
        """
        # Convert to numpy arrays and ensure proper data types
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # Remove rows with NaN values to ensure clean training data
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = np.array(sample_weights, dtype=float)
            sample_weights = sample_weights[mask]

        # Split data while preserving sample weights alignment
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize Logistic Regression model
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)

        # Train with or without sample weights
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)

        # Generate predictions and class probabilities
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        return clf, X_test, y_test, y_pred, y_proba

    def evaluate_model(self, clf: RandomForestClassifier, X_test, y_test, sample_weights=None, dataset_name='Test'):
        """
        Comprehensive model evaluation with multiple metrics.

        Parameters:
        -----------
        clf : sklearn classifier
            Trained classifier to evaluate
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        sample_weights : array-like, optional
            Sample weights (currently unused but kept for compatibility)
        dataset_name : str
            Name for the dataset being evaluated (for display purposes)

        Returns:
        --------
        dict or None
            Dictionary containing evaluation metrics, or None if error occurs
        """
        try:
            # Generate predictions using the trained classifier
            y_pred = clf.predict(X_test)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

        # Calculate comprehensive evaluation metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # Display evaluation results
        print(f"\n{dataset_name} Set Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])

        return metrics

    def train_with_focal_loss(self, X, y, sample_weights=None, alpha=0.25, gamma=2.0):
        """
        Train a neural network model using focal loss to improve handling of class imbalance.
        Focal loss down-weights easy examples and focuses on hard examples.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        alpha : float, default=0.25
            Weighting factor for rare class
        gamma : float, default=2.0
            Focusing parameter (higher gamma = more focus on hard examples)

        Returns:
        --------
        CustomNeuralNetwork
            Trained neural network model with focal loss
        """
        nn_model = CustomNeuralNetwork(loss_fn='focal', alpha=alpha, gamma=gamma)
        nn_model.fit(X, y, sample_weights=sample_weights)
        return nn_model

    def train_with_dice_loss(self, X, y, sample_weights=None, smooth=1.0):
        """
        Train a neural network model using dice loss to focus on true positives.
        Dice loss measures overlap between predicted and actual positive regions.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        smooth : float, default=1.0
            Smoothing factor to avoid division by zero

        Returns:
        --------
        CustomNeuralNetwork
            Trained neural network model with dice loss
        """
        nn_model = CustomNeuralNetwork(loss_fn='dice', smooth=smooth)
        nn_model.fit(X, y, sample_weights=sample_weights)
        return nn_model

    def train_with_tversky_loss(self, X, y, sample_weights=None, alpha=0.3, beta=0.7, smooth=1.0):
        """
        Train a neural network model using Tversky loss to handle class imbalance
        with explicit control over false positives and false negatives.

        Tversky loss is a generalization of Dice loss that allows asymmetric
        weighting of false positives and false negatives.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        alpha : float, default=0.3
            Penalty weight for false positives (lower = less penalty)
        beta : float, default=0.7
            Penalty weight for false negatives (higher = more penalty)
        smooth : float, default=1.0
            Smoothing factor to avoid division by zero

        Returns:
        --------
        CustomNeuralNetwork
            Trained neural network model with Tversky loss
        """
        nn_model = CustomNeuralNetwork(
            loss_fn='tversky',
            alpha=alpha,
            beta=beta,
            smooth=smooth
        )
        nn_model.fit(X, y, sample_weights=sample_weights)
        return nn_model

    def optimize_for_tpr(self, X, y, sample_weights=None, threshold_range=(0.1, 0.9), steps=20):
        """
        Optimize decision threshold to maximize true positive rate while maintaining
        reasonable accuracy. This is particularly useful for species distribution
        modeling where detecting presence is more important than avoiding false positives.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        threshold_range : tuple, default=(0.1, 0.9)
            Range of thresholds to test
        steps : int, default=20
            Number of threshold values to test

        Returns:
        --------
        tuple
            Trained classifier and dictionary of optimization metrics
        """
        # Split data for training and threshold optimization
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        if sample_weights is not None:
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train, y_train)

        # Get class probabilities for threshold optimization
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Initialize optimization variables
        best_threshold = 0.5
        best_tpr = 0
        best_accuracy = 0

        # Test different thresholds to find optimal balance
        for threshold in np.linspace(threshold_range[0], threshold_range[1], steps):
            y_pred = (y_proba >= threshold).astype(int)
            tpr = recall_score(y_test, y_pred)  # True Positive Rate
            accuracy = accuracy_score(y_test, y_pred)

            # Update best threshold if TPR improves and accuracy remains acceptable
            if tpr > best_tpr and accuracy > 0.5:
                best_tpr = tpr
                best_threshold = threshold
                best_accuracy = accuracy

        # Store optimal threshold for later use
        self.optimal_threshold = best_threshold

        # Return model and optimization results
        metrics = {
            'optimal_threshold': best_threshold,
            'true_positive_rate': best_tpr,
            'accuracy': best_accuracy
        }

        return clf, metrics

    def evaluate_model_with_tpr(self, clf, X_test, y_test, sample_weights=None, dataset_name='Test'):
        """
        Evaluate model with focus on true positive rate and true negative rate.
        Uses optimal threshold if available from previous optimization.

        Parameters:
        -----------
        clf : sklearn classifier
            Trained classifier to evaluate
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        sample_weights : array-like, optional
            Sample weights (currently unused)
        dataset_name : str
            Name for the dataset being evaluated

        Returns:
        --------
        dict or None
            Dictionary containing TPR-focused evaluation metrics
        """
        try:
            # Get class probabilities
            y_proba = clf.predict_proba(X_test)[:, 1]

            # Use optimal threshold if available, otherwise use default prediction
            if hasattr(self, 'optimal_threshold'):
                y_pred = (y_proba >= self.optimal_threshold).astype(int)
            else:
                y_pred = clf.predict(X_test)

            # Calculate confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            # Calculate key metrics for species distribution modeling
            accuracy = accuracy_score(y_test, y_pred)
            tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity)
            tnr = tn / (tn + fp)  # True Negative Rate (Specificity)

            # Compile comprehensive metrics
            metrics = {
                'accuracy': accuracy,
                'true_positive_rate': tpr,
                'true_negative_rate': tnr,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }

            # Display detailed evaluation results
            print(f"\n{dataset_name} Set Evaluation:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"True Positive Rate: {metrics['true_positive_rate']:.4f}")
            print(f"True Negative Rate: {metrics['true_negative_rate']:.4f}")
            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
            print("\nClassification Report:")
            print(metrics['classification_report'])

            return metrics

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None

    def train_with_tversky_scoring(self, X, y, sample_weights=None, alpha=0.3, beta=0.7, model_type='rf'):
        """
        Train model with Tversky scoring to optimize threshold using Tversky score.
        """
        # Convert to numpy arrays and handle NaN values
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        # Remove rows with NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = np.array(sample_weights, dtype=float)
            sample_weights = sample_weights[mask]
        # Split data while maintaining weight alignment
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        # Initialize classifier based on model type
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
        # Train model with optional sample weights
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        # Get class probabilities for threshold optimization
        y_proba = clf.predict_proba(X_test)[:, 1]
        # Optimize threshold using Tversky scorer
        tversky_scorer = TverskyScorer(alpha=alpha, beta=beta)
        thresholds = np.linspace(0.1, 0.9, 20)
        best_threshold = 0.5
        best_score = 0
        for threshold in thresholds:
            score = tversky_scorer(y_test, y_proba, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        self.optimal_threshold = best_threshold
        return clf

    def train_with_focal_scoring(self, X, y, sample_weights=None, alpha=0.25, gamma=2.0, model_type='rf'):
        """
        Train model with Focal scoring to handle class imbalance through threshold optimization.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        alpha : float, default=0.25
            Focal loss alpha parameter (class weighting)
        gamma : float, default=2.0
            Focal loss gamma parameter (focusing parameter)
        model_type : str, default='rf'
            Type of model to train ('rf' for Random Forest, 'logistic' for Logistic Regression)

        Returns:
        --------
        sklearn classifier
            Trained classifier with optimized threshold stored
        """
        # Convert to numpy arrays and handle NaN values
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # Remove rows with NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = np.array(sample_weights, dtype=float)
            sample_weights = sample_weights[mask]

        # Split data while maintaining weight alignment
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize classifier based on model type
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)

        # Train model with optional sample weights
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)

        # Get class probabilities for threshold optimization
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Optimize threshold using Focal scorer
        focal_scorer = FocalScorer(alpha=alpha, gamma=gamma)
        thresholds = np.linspace(0.1, 0.9, 20)
        best_threshold = 0.5
        best_score = 0

        # Find threshold that maximizes Focal score
        for threshold in thresholds:
            score = focal_scorer(y_test, y_proba, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold

        # Store optimal threshold for later use
        self.optimal_threshold = best_threshold

        return clf

    def train_with_dice_scoring(self, X, y, sample_weights=None, smooth=1.0, model_type='rf'):
        """
        Train model with Dice scoring to optimize overlap between predicted and actual positives.
        Dice score is particularly useful for imbalanced datasets in species distribution modeling.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        smooth : float, default=1.0
            Smoothing factor to avoid division by zero
        model_type : str, default='rf'
            Type of model to train ('rf' for Random Forest, 'logistic' for Logistic Regression)

        Returns:
        --------
        sklearn classifier
            Trained classifier with optimized threshold stored
        """
        # Convert to numpy arrays and handle NaN values
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # Remove rows with NaN values to ensure clean training data
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = np.array(sample_weights, dtype=float)
            sample_weights = sample_weights[mask]

        # Split data while maintaining weight alignment
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize classifier based on model type
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)

        # Train model with optional sample weights
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)

        # Get class probabilities for threshold optimization
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Optimize threshold using Dice scorer
        dice_scorer = DiceScorer(smooth=smooth)
        thresholds = np.linspace(0.1, 0.9, 20)
        best_threshold = 0.5
        best_score = 0

        # Find threshold that maximizes Dice score
        for threshold in thresholds:
            score = dice_scorer(y_test, y_proba, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold

        # Store optimal threshold for later use
        self.optimal_threshold = best_threshold

        return clf

    def comprehensive_species_modeling(self, species_name, presence_path=None, absence_path=None):
        """
        Comprehensive species distribution modeling pipeline that:
        1. Loads presence points for a specific species from all_presence_point.csv
        2. Extracts environmental features
        3. Uses as pseudo-absence points those from other species with a different 'order' and not already presence points for the target species
        4. Evaluates all combinations of models and loss functions
        5. Saves extracted features for reuse in feature importance analysis

        Args:
            species_name (str): Name of the species to model (must match species column in CSV)

        Returns:
            dict: Dictionary containing all evaluation results
        """
        import os
        from .features_extractor import Feature_Extractor
        # from .pseudo_absence_generator import PseudoAbsences  # Not needed for this approach

        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE MODELING FOR SPECIES: {species_name}")
        print(f"{'='*60}")

        # Initialize results dictionary
        results = {
            'species_name': species_name,
            'model_evaluations': {},
            'presence_count': 0,
            'absence_count': 0,
            'feature_count': 0
        }

        try:
            # Step 1: Load presence points for the specific species
            presence_csv_path = "data/testing_SDM/all_presence_point.csv"
            print(f"\n1. Loading presence points for {species_name} from: {presence_csv_path}")

            # Load the complete dataset
            all_species_df = pd.read_csv(presence_csv_path)

            # Verify required columns exist
            required_columns = ['species', 'order', 'decimalLatitude', 'decimalLongitude']
            missing_columns = [col for col in required_columns if col not in all_species_df.columns]
            if missing_columns:
                raise ValueError(f"CSV must contain columns: {missing_columns}")

            # Filter data for the specific species
            presence_df = all_species_df[all_species_df['species'] == species_name].copy()
            if len(presence_df) == 0:
                raise ValueError(f"No presence points found for species: {species_name}")

            # Get the order of the target species
            target_order = presence_df['order'].iloc[0]

            # Rename columns to match expected format and select only lat/lon
            presence_df = presence_df.rename(columns={
                'decimalLatitude': 'latitude',
                'decimalLongitude': 'longitude'
            })
            presence_df = presence_df[['longitude', 'latitude']]

            results['presence_count'] = len(presence_df)
            print(f"   Loaded {len(presence_df)} presence points for {species_name}")

            # Step 2: Extract environmental features for presence points
            print(f"\n2. Extracting environmental features for presence points...")
            try:
                import ee
                ee.Authenticate()
                ee.Initialize(project='ee-mtpictd')
            except:
                print("   Warning: Earth Engine not initialized. Please initialize EE first.")
                return results
            feature_extractor = Feature_Extractor(ee)
            presence_with_features = feature_extractor.add_features(presence_df)
            results['feature_count'] = len(presence_with_features.columns) - 2  # Exclude lat/lon
            print(f"   Extracted {results['feature_count']} environmental features for presence points")

            # Step 3: Select pseudo-absence points from all_species_df
            print(f"\n3. Selecting pseudo-absence points from all_presence_point.csv...")
            # Get all points with a different order and not already presence points for this species
            presence_coords = set(zip(presence_df['longitude'], presence_df['latitude']))
            pseudo_absence_df = all_species_df[(all_species_df['order'] != target_order)].copy()
            pseudo_absence_df = pseudo_absence_df.rename(columns={
                'decimalLatitude': 'latitude',
                'decimalLongitude': 'longitude'
            })
            pseudo_absence_df = pseudo_absence_df[['longitude', 'latitude']]
            pseudo_absence_df = pseudo_absence_df[~pseudo_absence_df.apply(lambda row: (row['longitude'], row['latitude']) in presence_coords, axis=1)]
            # Sample the same number of pseudo-absences as presences (or all if fewer available)
            if len(pseudo_absence_df) > len(presence_df):
                pseudo_absence_df = pseudo_absence_df.sample(n=len(presence_df), random_state=42)
            results['absence_count'] = len(pseudo_absence_df)
            print(f"   Selected {len(pseudo_absence_df)} pseudo-absence points")

            # Step 4: Extract environmental features for pseudo-absence points
            print(f"\n4. Extracting environmental features for pseudo-absence points...")
            absence_with_features = feature_extractor.add_features(pseudo_absence_df)
            print(f"   Extracted features for pseudo-absence points")

            # Step 5: Save extracted features for reuse
            print(f"\n5. Saving extracted features for reuse...")
            features_output_dir = "outputs/extracted_features"
            os.makedirs(features_output_dir, exist_ok=True)

            # Save presence and absence features separately
            presence_features_file = os.path.join(features_output_dir, f"{species_name.replace(' ', '_').lower()}_presence_features.csv")
            absence_features_file = os.path.join(features_output_dir, f"{species_name.replace(' ', '_').lower()}_absence_features.csv")

            presence_with_features.to_csv(presence_features_file, index=False)
            absence_with_features.to_csv(absence_features_file, index=False)
            print(f"   Saved presence features to: {presence_features_file}")
            print(f"   Saved absence features to: {absence_features_file}")

            # Step 6: Prepare data for modeling
            print(f"\n6. Preparing data for modeling...")
            feature_cols = [col for col in presence_with_features.columns if col not in ['longitude', 'latitude']]
            presence_features = presence_with_features[feature_cols].values
            absence_features = absence_with_features[feature_cols].values
            X = np.vstack([presence_features, absence_features])
            y = np.hstack([np.ones(len(presence_features)), np.zeros(len(absence_features))])

            # Clean data by removing rows with NaN values
            print(f"   Original data shape: {X.shape}")
            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
            y = y[mask]
            print(f"   After removing NaN values: {X.shape}")

            # Apply bias correction with full ecoregion-based weighting
            feature_cols = [col for col in presence_with_features.columns if col not in ['longitude', 'latitude']]
            X, y, reliability_weights, bias_weights, combined_weights = self.apply_bias_correction(
                presence_with_features, absence_with_features, feature_cols
            )

            # Clean data by removing rows with NaN values after bias correction
            print(f"   After bias correction data shape: {X.shape}")
            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
            y = y[mask]
            combined_weights = combined_weights[mask]
            print(f"   After removing NaN values: {X.shape}")
            print(f"   Prepared {len(X)} total samples after cleaning")

            # Step 7: Define model and loss function combinations
            print(f"\n7. Evaluating model combinations...")
            model_configs = [
                ('Random_Forest', 'rf'),
                ('Logistic_Regression', 'logistic'),
                ('Weighted_Logistic_Regression', 'logistic_weighted')
            ]
            loss_configs = [
                ('Original_Loss', 'original', {}),
                ('Dice_Loss', 'dice', {'smooth': 1.0}),
                ('Focal_Loss', 'focal', {'alpha': 0.25, 'gamma': 2.0}),
                ('Tversky_Loss', 'tversky', {'alpha': 0.3, 'beta': 0.7})
            ]

            # Initialize results table for CSV output
            results_table = []
            best_accuracy = -1
            best_combo = None
            best_row = None

            # --- Random Forest and Logistic Regression ---
            for model_name, model_type in model_configs:
                for loss_name, loss_type, loss_params in loss_configs:
                    combination_name = f"{model_name}_{loss_name}"
                    print(f"\n   Evaluating: {combination_name}")
                    try:
                        if loss_type == 'original':
                            if model_type == 'rf':
                                clf, X_test, y_test, y_pred, y_proba = self.RandomForest(X, y, sample_weights=combined_weights)
                            elif model_type == 'logistic':
                                clf, X_test, y_test, y_pred, y_proba = self.logistic_regression_L2(X, y)
                            else:
                                clf, X_test, y_test, y_pred, y_proba = self.train_and_evaluate_model_logistic_weighted(X, y, sample_weights=combined_weights)
                            metrics = self.evaluate_model(clf, X_test, y_test, dataset_name=combination_name)
                            optimal_threshold = 0.5
                            # Calculate TPR and TNR
                            cm = metrics['confusion_matrix']
                            if cm.shape == (2, 2):
                                tn, fp, fn, tp = cm.ravel()
                                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                            else:
                                tpr = tnr = None
                            # Print TPR and TNR
                            print(f"      TPR: {tpr:.4f}" if tpr is not None else "      TPR: N/A")
                            print(f"      TNR: {tnr:.4f}" if tnr is not None else "      TNR: N/A")
                        else:
                            # For non-original loss, use the original model but optimize threshold with custom scorer
                            if model_type == 'rf':
                                clf, X_test, y_test, y_pred, y_proba = self.RandomForest(X, y, sample_weights=combined_weights)
                            elif model_type == 'logistic':
                                clf, X_test, y_test, y_pred, y_proba = self.logistic_regression_L2(X, y)
                            else:
                                clf, X_test, y_test, y_pred, y_proba = self.train_and_evaluate_model_logistic_weighted(X, y, sample_weights=combined_weights)
                            # Choose scorer
                            if loss_type == 'dice':
                                scorer = DiceScorer(smooth=loss_params.get('smooth', 1.0))
                            elif loss_type == 'focal':
                                scorer = FocalScorer(alpha=loss_params.get('alpha', 0.25), gamma=loss_params.get('gamma', 2.0))
                            elif loss_type == 'tversky':
                                scorer = TverskyScorer(alpha=loss_params.get('alpha', 0.3), beta=loss_params.get('beta', 0.7))
                            else:
                                scorer = None
                            # Optimize threshold
                            thresholds = np.linspace(0.1, 0.9, 20)
                            best_score = -np.inf
                            optimal_threshold = 0.5
                            for threshold in thresholds:
                                score = scorer(y_test, y_proba, threshold)
                                if score > best_score:
                                    best_score = score
                                    optimal_threshold = threshold
                            # Apply optimal threshold
                            y_pred = (y_proba >= optimal_threshold).astype(int)
                            cm = confusion_matrix(y_test, y_pred)
                            if cm.shape == (2, 2):
                                tn, fp, fn, tp = cm.ravel()
                                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                            else:
                                tpr = tnr = None
                            metrics = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'confusion_matrix': cm,
                                'classification_report': classification_report(y_test, y_pred, output_dict=True)
                            }
                            # Print TPR and TNR
                            print(f"      TPR: {tpr:.4f}" if tpr is not None else "      TPR: N/A")
                            print(f"      TNR: {tnr:.4f}" if tnr is not None else "      TNR: N/A")

                        # Store results in dictionary
                        results['model_evaluations'][combination_name] = {
                            'accuracy': metrics['accuracy'] if metrics else None,
                            'confusion_matrix': metrics['confusion_matrix'].tolist() if metrics else None,
                            'classification_report': metrics['classification_report'] if metrics else None,
                            'optimal_threshold': optimal_threshold,
                            'tpr': tpr,
                            'tnr': tnr
                        }

                        # Prepare row for CSV
                        row = {
                            'model_loss': combination_name,
                            'accuracy': metrics['accuracy'],
                            'optimal_threshold': optimal_threshold,
                            'tpr': tpr,
                            'tnr': tnr
                        }
                        # Add precision, recall, f1, support for each class
                        for label in ['0', '1', 'macro avg', 'weighted avg']:
                            if label in metrics['classification_report']:
                                row[f'{label}_precision'] = metrics['classification_report'][label]['precision']
                                row[f'{label}_recall'] = metrics['classification_report'][label]['recall']
                                row[f'{label}_f1-score'] = metrics['classification_report'][label]['f1-score']
                                row[f'{label}_support'] = metrics['classification_report'][label]['support']
                        results_table.append(row)

                        # Track best
                        if metrics['accuracy'] > best_accuracy:
                            best_accuracy = metrics['accuracy']
                            best_combo = combination_name
                            best_row = row

                        print(f"      Accuracy: {metrics['accuracy']:.4f}" if metrics else "      Evaluation failed")

                    except Exception as e:
                        print(f"      Error in {combination_name}: {str(e)}")
                        results['model_evaluations'][combination_name] = {
                            'error': str(e)
                        }

            # Save results as CSV
            results_df = pd.DataFrame(results_table)
            output_csv = f"outputs/{species_name}_comprehensive_results.csv"
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            results_df.to_csv(output_csv, index=False)
            print(f"\nResults table saved to: {output_csv}")

            # Print summary table
            print(f"\n{'='*80}")
            print(f"SUMMARY TABLE FOR {species_name}")
            print(f"{'='*80}")
            print(f"{'Model/Loss':<35} {'Accuracy':<10} {'F1-Score':<10} {'TPR':<8} {'TNR':<8} {'Threshold':<10}")
            print(f"{'-'*80}")
            for _, row in results_df.iterrows():
                f1_score = row.get('weighted avg_f1-score', 'N/A')
                f1_display = f"{f1_score:.4f}" if isinstance(f1_score, (int, float)) else "N/A"
                print(f"{row['model_loss']:<35} {row['accuracy']:<10.4f} {f1_display:<10} {row['tpr']:<8.4f} {row['tnr']:<8.4f} {row['optimal_threshold']:<10.4f}")

            # Print best model/loss
            print(f"\nBest performing model+loss: {best_combo} (Accuracy: {best_accuracy:.4f})")
            if best_row:
                print("Best model/loss detailed metrics:")
                for k, v in best_row.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")

            print(f"\n{'='*60}")
            print(f"SUMMARY FOR {species_name}")
            print(f"{'='*60}")
            print(f"Presence points: {results['presence_count']}")
            print(f"Absence points: {results['absence_count']}")
            print(f"Environmental features: {results['feature_count']}")
            print(f"Model combinations evaluated: {len(results['model_evaluations'])}")

            # Save JSON results
            output_file = f"outputs/{species_name}_modeling_results.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            import json
            json_results = {}
            for key, value in results.items():
                if key == 'model_evaluations':
                    json_results[key] = {}
                    for model_name, eval_results in value.items():
                        json_results[key][model_name] = {}
                        for metric_name, metric_value in eval_results.items():
                            if isinstance(metric_value, np.ndarray):
                                json_results[key][model_name][metric_name] = metric_value.tolist()
                            else:
                                json_results[key][model_name][metric_name] = metric_value
                else:
                    json_results[key] = value
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\nResults saved to: {output_file}")

        except Exception as e:
            print(f"Error in comprehensive modeling: {str(e)}")
            results['error'] = str(e)
        return results

    def train_on_max_ecoregion(self, species_name):
        """
        For a given species, find the ecoregion with the most presence points, filter both presence and pseudo-absence points to that ecoregion, and run the modeling pipeline on the filtered data.
        """
        import os
        from shapely.wkt import loads as load_wkt
        from shapely.geometry import Point
        from .features_extractor import Feature_Extractor
        print(f"\n{'='*60}")
        print(f"ECOREGION-LEVEL MODELING FOR SPECIES: {species_name}")
        print(f"{'='*60}")
        # Load all presence points for the species
        presence_csv_path = "data/testing_SDM/all_presence_point.csv"
        all_species_df = pd.read_csv(presence_csv_path)
        required_columns = ['species', 'order', 'decimalLatitude', 'decimalLongitude']
        missing_columns = [col for col in required_columns if col not in all_species_df.columns]
        if missing_columns:
            raise ValueError(f"CSV must contain columns: {missing_columns}")
        presence_df = all_species_df[all_species_df['species'] == species_name].copy()
        if len(presence_df) == 0:
            raise ValueError(f"No presence points found for species: {species_name}")
        # Load all ecoregion polygons
        eco_dir = "data/eco_regions_polygon"
        eco_polygons = {}
        for fname in os.listdir(eco_dir):
            if fname.endswith('.wkt'):
                eco_name = fname.replace('.wkt', '')
                with open(os.path.join(eco_dir, fname), 'r') as f:
                    eco_polygons[eco_name] = load_wkt(f.read().strip())
        # Find which ecoregion has the most presence points
        presence_points = [Point(lon, lat) for lon, lat in zip(presence_df['decimalLongitude'], presence_df['decimalLatitude'])]
        eco_counts = {eco: 0 for eco in eco_polygons}
        eco_assignments = []
        for pt in presence_points:
            found = False
            for eco, poly in eco_polygons.items():
                if poly.contains(pt):
                    eco_counts[eco] += 1
                    eco_assignments.append(eco)
                    found = True
                    break
            if not found:
                eco_assignments.append(None)
        # Get the ecoregion with the maximum count
        max_eco = max(eco_counts, key=eco_counts.get)
        print(f"   Ecoregion with max presence points: {max_eco} ({eco_counts[max_eco]} points)")
        # Filter presence points to only those in the max ecoregion
        presence_df = presence_df[[eco == max_eco for eco in eco_assignments]].copy()
        presence_df = presence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
        presence_df = presence_df[['longitude', 'latitude']]
        # Prepare pseudo-absence points: use points from other species, different order, not in presence, and in the same ecoregion
        target_order = all_species_df[all_species_df['species'] == species_name]['order'].iloc[0]
        presence_coords = set(zip(presence_df['longitude'], presence_df['latitude']))
        pseudo_absence_df = all_species_df[(all_species_df['order'] != target_order)].copy()
        pseudo_absence_df = pseudo_absence_df.rename(columns={
            'decimalLatitude': 'latitude',
            'decimalLongitude': 'longitude'
        })
        pseudo_absence_df = pseudo_absence_df[['longitude', 'latitude']]
        # Only keep pseudo-absence points in the max ecoregion and not in presence
        initial_absence_count = len(pseudo_absence_df)
        pseudo_absence_points = [Point(lon, lat) for lon, lat in zip(pseudo_absence_df['longitude'], pseudo_absence_df['latitude'])]
        pseudo_absence_df = pseudo_absence_df[[eco_polygons[max_eco].contains(pt) and (pt.x, pt.y) not in presence_coords for pt in pseudo_absence_points]]
        filtered_absence_count = len(pseudo_absence_df)
        removed_duplicates = initial_absence_count - filtered_absence_count

        # Sample the same number of pseudo-absences as presences (or all if fewer available)
        if len(pseudo_absence_df) > len(presence_df):
            pseudo_absence_df = pseudo_absence_df.sample(n=len(presence_df), random_state=42)

        print(f"   Filtered to {len(presence_df)} presence and {len(pseudo_absence_df)} pseudo-absence points in {max_eco}")
        print(f"🔍 ABSENCE POINT FILTERING: Removed {removed_duplicates} duplicate points that matched presence coordinates")
        print(f"   Initial absence points: {initial_absence_count}")
        print(f"   After filtering: {filtered_absence_count}")
        print(f"   Duplicates removed: {removed_duplicates}")
        # Feature extraction
        try:
            import ee
            ee.Authenticate()
            ee.Initialize(project='ee-mtpictd')
        except:
            print("   Warning: Earth Engine not initialized. Please initialize EE first.")
            return
        feature_extractor = Feature_Extractor(ee)
        presence_with_features = feature_extractor.add_features(presence_df)
        absence_with_features = feature_extractor.add_features(pseudo_absence_df)
        # Prepare data for modeling (same as in comprehensive_species_modeling)
        feature_cols = [col for col in presence_with_features.columns if col not in ['longitude', 'latitude']]
        presence_features = presence_with_features[feature_cols].values
        absence_features = absence_with_features[feature_cols].values
        X = np.vstack([presence_features, absence_features])
        y = np.hstack([np.ones(len(presence_features)), np.zeros(len(absence_features))])
        # Clean data by removing rows with NaN values
        print(f"   Original data shape: {X.shape}")
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        print(f"   After removing NaN values: {X.shape}")

        # Apply bias correction with full ecoregion-based weighting
        # This now automatically filters out NaN values
        feature_cols = [col for col in presence_with_features.columns if col not in ['longitude', 'latitude']]
        X, y, reliability_weights, bias_weights, combined_weights = self.apply_bias_correction(
            presence_with_features, absence_with_features, feature_cols
        )

        # Data is already cleaned of NaN values in apply_bias_correction
        print(f"   After bias correction and NaN filtering: {X.shape}")
        print(f"   Prepared {len(X)} total samples for modeling")

        # Modeling loop (same as in comprehensive_species_modeling)
        model_configs = [
            ('Random_Forest', 'rf'),
            ('Logistic_Regression', 'logistic'),
            ('Weighted_Logistic_Regression', 'logistic_weighted')
        ]
        loss_configs = [
            ('Original_Loss', 'original', {}),
            ('Dice_Loss', 'dice', {'smooth': 1.0}),
            ('Focal_Loss', 'focal', {'alpha': 0.25, 'gamma': 2.0}),
            ('Tversky_Loss', 'tversky', {'alpha': 0.3, 'beta': 0.7})
        ]
        results_table = []
        best_accuracy = -1
        best_combo = None
        best_row = None

        try:
            for model_name, model_type in model_configs:
                for loss_name, loss_type, loss_params in loss_configs:
                    combination_name = f"{model_name}_{loss_name}"
                    print(f"\n   Evaluating: {combination_name}")
                    try:
                        if loss_type == 'original':
                            if model_type == 'rf':
                                clf, X_test, y_test, y_pred, y_proba = self.RandomForest(X, y, sample_weights=combined_weights)
                            elif model_type == 'logistic':
                                clf, X_test, y_test, y_pred, y_proba = self.logistic_regression_L2(X, y)
                            else:
                                clf, X_test, y_test, y_pred, y_proba = self.train_and_evaluate_model_logistic_weighted(X, y, sample_weights=combined_weights)
                            metrics = self.evaluate_model(clf, X_test, y_test, dataset_name=combination_name)
                            optimal_threshold = 0.5
                            # Calculate TPR and TNR
                            cm = metrics['confusion_matrix']
                            if cm.shape == (2, 2):
                                tn, fp, fn, tp = cm.ravel()
                                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                            else:
                                tpr = tnr = None
                            # Print classification report
                            print(metrics['classification_report'])
                        else:
                            if model_type == 'rf':
                                clf, X_test, y_test, y_pred, y_proba = self.RandomForest(X, y, sample_weights=combined_weights)
                            elif model_type == 'logistic':
                                clf, X_test, y_test, y_pred, y_proba = self.logistic_regression_L2(X, y)
                            else:
                                clf, X_test, y_test, y_pred, y_proba = self.train_and_evaluate_model_logistic_weighted(X, y, sample_weights=combined_weights)
                            if loss_type == 'dice':
                                from .custom_losses import DiceScorer
                                scorer = DiceScorer(smooth=loss_params.get('smooth', 1.0))
                            elif loss_type == 'focal':
                                from .custom_losses import FocalScorer
                                scorer = FocalScorer(alpha=loss_params.get('alpha', 0.25), gamma=loss_params.get('gamma', 2.0))
                            elif loss_type == 'tversky':
                                from .custom_losses import TverskyScorer
                                scorer = TverskyScorer(alpha=loss_params.get('alpha', 0.3), beta=loss_params.get('beta', 0.7))
                            else:
                                scorer = None
                            thresholds = np.linspace(0.1, 0.9, 20)
                            best_score = -np.inf
                            optimal_threshold = 0.5
                            for threshold in thresholds:
                                score = scorer(y_test, y_proba, threshold)
                                if score > best_score:
                                    best_score = score
                                    optimal_threshold = threshold
                            # Calculate metrics with optimal threshold
                            y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                            accuracy = accuracy_score(y_test, y_pred_optimal)
                            precision = precision_score(y_test, y_pred_optimal, zero_division=0)
                            recall = recall_score(y_test, y_pred_optimal, zero_division=0)
                            f1 = f1_score(y_test, y_pred_optimal, zero_division=0)
                            cm = confusion_matrix(y_test, y_pred_optimal)
                            if cm.shape == (2, 2):
                                tn, fp, fn, tp = cm.ravel()
                                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                            else:
                                tpr = tnr = None
                            metrics = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'confusion_matrix': cm
                            }
                            print(f"      Optimal threshold: {optimal_threshold:.4f}")
                            print(f"      Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

                        # Store results
                        row = {
                            'model_loss': combination_name,
                            'accuracy': metrics['accuracy'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1'],
                            'tpr': tpr if tpr is not None else 0.0,
                            'tnr': tnr if tnr is not None else 0.0,
                            'optimal_threshold': optimal_threshold
                        }
                        results_table.append(row)

                        # Update best model
                        if metrics['accuracy'] > best_accuracy:
                            best_accuracy = metrics['accuracy']
                            best_combo = combination_name
                            best_row = row

                    except Exception as e:
                        print(f"      Error in {combination_name}: {str(e)}")
                        results_table.append({
                            'model_loss': combination_name,
                            'error': str(e)
                        })

            # Save results as CSV
            results_df = pd.DataFrame(results_table)
            output_csv = f"outputs/{species_name}_ecoregion_results.csv"
            results_df.to_csv(output_csv, index=False)
            print(f"\nResults table saved to: {output_csv}")
            # Print best model/loss
            print(f"\nBest performing model+loss: {best_combo} (Accuracy: {best_accuracy:.4f})")
            if best_row:
                print("Best model/loss metrics:")
                for k, v in best_row.items():
                    print(f"  {k}: {v}")
            print(f"\n{'='*60}")
            print(f"SUMMARY FOR {species_name} in {max_eco}")
            print(f"{'='*60}")
            print(f"Presence points: {len(presence_df)}")
            print(f"Absence points: {len(pseudo_absence_df)}")
            print(f"Model combinations evaluated: {len(model_configs) * len(loss_configs)}")

            # Save extracted features for later use in feature importance analysis
            output_dir = "outputs/extracted_features"
            os.makedirs(output_dir, exist_ok=True)
            species_safe = species_name.replace(' ', '_').lower()
            presence_path = os.path.join(output_dir, f"{species_safe}_ecoregion_presence_features.csv")
            absence_path = os.path.join(output_dir, f"{species_safe}_ecoregion_absence_features.csv")
            presence_with_features.to_csv(presence_path, index=False)
            absence_with_features.to_csv(absence_path, index=False)
            print(f"Features saved to: {presence_path} and {absence_path}")

        except Exception as e:
            print(f"Error in ecoregion modeling for {species_name}: {str(e)}")
            return

    def evaluate_all_models(self, X, y, weights):
        """
        Evaluate all available models and return results.
        """
        results = {}

        # Random Forest
        try:
            clf, accuracy, precision, recall, f1 = self.RandomForest(X, y, sample_weights=weights)
            results['Random_Forest_original'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            print(f"Error in Random Forest: {e}")
            results['Random_Forest_original'] = {'error': str(e)}

        # Logistic Regression
        try:
            clf, accuracy, precision, recall, f1 = self.logistic_regression_L2(X, y)
            results['Logistic_Regression_original'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            print(f"Error in Logistic Regression: {e}")
            results['Logistic_Regression_original'] = {'error': str(e)}

        # Weighted Logistic Regression
        try:
            clf, accuracy, precision, recall, f1 = self.train_and_evaluate_model_logistic_weighted(X, y, sample_weights=weights)
            results['Weighted_Logistic_Regression_original'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            print(f"Error in Weighted Logistic Regression: {e}")
            results['Weighted_Logistic_Regression_original'] = {'error': str(e)}

        # Tversky Loss
        try:
            clf = self.train_with_tversky_scoring(X, y, sample_weights=weights, model_type='rf')
            # Evaluate the model
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights, test_size=0.2, random_state=42, stratify=y
            )
            y_pred = clf.predict(X_test)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            results['Random_Forest_Tversky'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            print(f"Error in Tversky Loss: {e}")
            results['Random_Forest_Tversky'] = {'error': str(e)}

        # Focal Loss
        try:
            clf = self.train_with_focal_scoring(X, y, sample_weights=weights, model_type='rf')
            # Evaluate the model
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights, test_size=0.2, random_state=42, stratify=y
            )
            y_pred = clf.predict(X_test)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            results['Random_Forest_Focal'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            print(f"Error in Focal Loss: {e}")
            results['Random_Forest_Focal'] = {'error': str(e)}

        # Dice Loss
        try:
            clf = self.train_with_dice_scoring(X, y, sample_weights=weights, model_type='rf')
            # Evaluate the model
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights, test_size=0.2, random_state=42, stratify=y
            )
            y_pred = clf.predict(X_test)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            results['Random_Forest_Dice'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            print(f"Error in Dice Loss: {e}")
            results['Random_Forest_Dice'] = {'error': str(e)}

        return results



    def comprehensive_genus_modeling(self, genus_name, model_type, loss_func, is_bias, is_endemic, presence_path=None, absence_path=None):
        """
        Comprehensive genus-level distribution modeling pipeline with configurable parameters.
        
        Args:
            genus_name: Name of the genus to model
            model_type: Type of model ('Random Forest', 'Logistic Regression', etc.)
            loss_func: Loss function to use ('Dice', 'Tversky', 'Focal', etc.)
            is_bias: Whether to apply bias correction (True/False)
            is_endemic: Whether the genus is endemic (True/False) - affects alpha/beta and optimization metric
            presence_path: Path to CSV file with presence points and features for this genus
            absence_path: Optional path to absence data
        """
        import os
        genus = genus_name
        features_csv_path = "data/presence_points_with_features.csv"
        reliability_threshold = 0.07
        
        # Set optimization metric based on endemic status
        if is_endemic:
            optimize_for = 'tnr'  # Endemic species: optimize for True Negative Rate
            alpha = 0.3
            beta = 0.7
            print(f"Endemic genus detected: Optimizing for TNR with alpha={alpha}, beta={beta}")
        else:
            optimize_for = 'tpr'  # Non-endemic species: optimize for True Positive Rate
            alpha = 0.7
            beta = 0.3
            print(f"Non-endemic genus detected: Optimizing for TPR with alpha={alpha}, beta={beta}")
        
        print(f"\nStarting genus modeling: {genus}")
        print(f"  Model Type: {model_type}")
        print(f"  Loss Function: {loss_func}")
        print(f"  Bias Correction: {is_bias}")
        print(f"  Endemic Status: {is_endemic}")
        print(f"  Optimization Metric: {optimize_for}")
        
        # Load absence points from features CSV
        try:
            features_df = pd.read_csv(features_csv_path, low_memory=False)
        except FileNotFoundError:
            print(f"Error: Features CSV not found at {features_csv_path}")
            return None
        all_absence_df = features_df.copy()

        # Load presence points with features from the provided presence_path
        if presence_path is None:
            presence_path = f"presence_points_{genus_name}.csv"
        
        try:
            pres_with_features = pd.read_csv(presence_path, low_memory=False)
            print(f"Loaded presence points with features from: {presence_path}")
        except FileNotFoundError:
            print(f"Error: Presence file not found at {presence_path}")
            return None
        
        print(f"Found {len(pres_with_features)} presence points for {genus}")

        # Define feature columns (exclude coordinate columns)
        exclude_cols = ['longitude', 'latitude', 'decimalLongitude', 'decimalLatitude']
        feature_cols = [col for col in pres_with_features.columns if col not in exclude_cols]

        # Remove any points from absence that are the same as presence points
        # Extract coordinates from presence data
        if 'longitude' in pres_with_features.columns and 'latitude' in pres_with_features.columns:
            presence_coords = set(zip(pres_with_features['longitude'], pres_with_features['latitude']))
            coord_cols_pres = ['longitude', 'latitude']
        elif 'decimalLongitude' in pres_with_features.columns and 'decimalLatitude' in pres_with_features.columns:
            presence_coords = set(zip(pres_with_features['decimalLongitude'], pres_with_features['decimalLatitude']))
            coord_cols_pres = ['decimalLongitude', 'decimalLatitude']
        else:
            print("Error: Could not find coordinate columns in presence data")
            return None
        
        # Extract coordinates from absence data
        if 'longitude' in all_absence_df.columns and 'latitude' in all_absence_df.columns:
            coord_cols_abs = ['longitude', 'latitude']
        elif 'decimalLongitude' in all_absence_df.columns and 'decimalLatitude' in all_absence_df.columns:
            coord_cols_abs = ['decimalLongitude', 'decimalLatitude']
        else:
            print("Error: Could not find coordinate columns in absence data")
            return None
        
        initial_absence_count = len(all_absence_df)
        all_absence_df = all_absence_df[~all_absence_df.apply(
            lambda r: (r[coord_cols_abs[0]], r[coord_cols_abs[1]]) in presence_coords, axis=1
        )]
        filtered_absence_count = len(all_absence_df)
        removed_duplicates = initial_absence_count - filtered_absence_count

        print(f"Found {len(all_absence_df)} potential absence points (all available)")
        print(f"🔍 ABSENCE POINT FILTERING: Removed {removed_duplicates} duplicate points that matched presence coordinates")
        print(f"   Initial absence points: {initial_absence_count}")
        print(f"   After filtering: {filtered_absence_count}")
        print(f"   Duplicates removed: {removed_duplicates}")

        # Remove rows with NaN values in features
        pres_clean = pres_with_features[~pres_with_features[feature_cols].isna().any(axis=1)].copy()
        all_absence_clean = all_absence_df[~all_absence_df[feature_cols].isna().any(axis=1)].copy()

        # Ensure all feature columns are numeric
        for col in feature_cols:
            if col in pres_clean.columns:
                pres_clean.loc[:, col] = pd.to_numeric(pres_clean[col], errors='coerce')
            if col in all_absence_clean.columns:
                all_absence_clean.loc[:, col] = pd.to_numeric(all_absence_clean[col], errors='coerce')

        # Remove any rows that still have NaN values after conversion
        pres_clean = pres_clean.dropna(subset=feature_cols)
        all_absence_clean = all_absence_clean.dropna(subset=feature_cols)

        print(f"Valid presence points: {len(pres_clean)}")
        print(f"Valid potential absence points: {len(all_absence_clean)}")

        # Calculate reliability scores for absence selection
        cache_file = f"outputs/absence_reliability_{genus.replace(' ', '_')}.csv"
        
        print("Calculating reliability scores...")
        reliability_scores = []
        for idx, row in all_absence_clean.iterrows():
            absence_features = row[feature_cols].values.astype(float)
            reliability = calculate_feature_based_reliability(absence_features, pres_clean[feature_cols].values.astype(float))
            reliability_scores.append(reliability)
        all_absence_clean['reliability_score'] = reliability_scores
        reliable_absences = all_absence_clean[all_absence_clean['reliability_score'] > reliability_threshold]
        
        if len(reliable_absences) == 0:
            threshold_50_percentile = all_absence_clean['reliability_score'].quantile(0.5)
            reliable_absences = all_absence_clean[all_absence_clean['reliability_score'] >= threshold_50_percentile]
        
        num_presence = len(pres_clean)
        target_absence = num_presence
        
        if len(reliable_absences) >= target_absence:
            absence_selected = reliable_absences.sample(n=target_absence, random_state=42)
        else:
            absence_selected = reliable_absences
        
        if len(absence_selected) == 0:
            absence_selected = all_absence_clean.sample(n=min(target_absence, len(all_absence_clean)), random_state=42)
        
        # Save to cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        absence_selected.to_csv(cache_file, index=False)
        print(f"Saved absence selection and reliability scores to cache: {cache_file}")

        # Analyze feature separation between presence and absence points
        print("\n" + "="*60)
        print("FEATURE SEPARATION ANALYSIS")
        print("="*60)

        separation_metrics = []
        for col in feature_cols:
            if col in pres_clean.columns and col in absence_selected.columns:
                pres_vals = pres_clean[col].dropna()
                abs_vals = absence_selected[col].dropna()

                if len(pres_vals) > 0 and len(abs_vals) > 0:
                    # Calculate basic statistics
                    pres_mean, pres_std = pres_vals.mean(), pres_vals.std()
                    abs_mean, abs_std = abs_vals.mean(), abs_vals.std()

                    # Calculate overlap using histogram intersection
                    hist_pres, _ = np.histogram(pres_vals, bins=20, density=True)
                    hist_abs, _ = np.histogram(abs_vals, bins=20, density=True)
                    overlap = np.minimum(hist_pres, hist_abs).sum() / 20

                    # Calculate separation score (1 - overlap)
                    separation_score = 1 - overlap

                    # Calculate effect size (Cohen d)
                    pooled_std = np.sqrt(((len(pres_vals) - 1) * pres_std**2 + (len(abs_vals) - 1) * abs_std**2) / (len(pres_vals) + len(abs_vals) - 2))
                    if pooled_std > 0:
                        cohens_d = abs(pres_mean - abs_mean) / pooled_std
                    else:
                        cohens_d = 0

                    separation_metrics.append({
                        'feature': col,
                        'presence_mean': pres_mean,
                        'absence_mean': abs_mean,
                        'presence_std': pres_std,
                        'absence_std': abs_std,
                        'overlap': overlap,
                        'separation_score': separation_score,
                        'cohens_d': cohens_d
                    })

        # Sort by separation score (highest first)
        separation_metrics.sort(key=lambda x: x['separation_score'], reverse=True)

        # Print summary
        print(f"{'Feature':<25} {'Separation':<12} {'Overlap':<10} {'Cohen d':<10} {'Pres Mean':<12} {'Abs Mean':<12}")
        print("-" * 90)
        for metric in separation_metrics:
            print(f"{metric['feature']:<25} {metric['separation_score']:<12.3f} {metric['overlap']:<10.3f} "
                f"{metric['cohens_d']:<10.3f} {metric['presence_mean']:<12.3f} {metric['absence_mean']:<12.3f}")

        # Highlight best separating features
        high_separation = [m for m in separation_metrics if m['separation_score'] > 0.7]
        if high_separation:
            print(f"\n🔍 HIGH SEPARATION FEATURES (separation > 0.7):")
            for metric in high_separation:
                print(f"   • {metric['feature']}: separation={metric['separation_score']:.3f}, Cohen d={metric['cohens_d']:.3f}")

        # Highlight features with high overlap (potential issues)
        high_overlap = [m for m in separation_metrics if m['overlap'] > 0.8]
        if high_overlap:
            print(f"\n⚠️  HIGH OVERLAP FEATURES (overlap > 0.8):")
            for metric in high_overlap:
                print(f"   • {metric['feature']}: overlap={metric['overlap']:.3f}, separation={metric['separation_score']:.3f}")

        print("="*60)

        print(f"Final dataset: presence={len(pres_clean)}, absence={len(absence_selected)}")

        # Create interactive map visualization
        try:
            map_file = visualize_presence_absence_points(pres_clean, absence_selected, genus)
            if map_file:
                print(f"✓ Interactive map created: {map_file}")
        except Exception as e:
            print(f"Warning: Could not create map visualization: {e}")

        # Save feature distribution plots
        import matplotlib.pyplot as plt
        plot_dir = 'outputs/feature_distributions'
        os.makedirs(plot_dir, exist_ok=True)
        for col in feature_cols:
            plt.figure(figsize=(6,4))
            plt.hist(pres_clean[col].dropna(), bins=20, alpha=0.5, label='Presence', color='blue', density=True)
            plt.hist(absence_selected[col].dropna(), bins=20, alpha=0.5, label='Absence', color='red', density=True)
            plt.title(f'Feature: {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{col}_hist_{genus}.png'))
            plt.close()
        print(f'Feature distribution plots saved to {plot_dir}')

        # Standardize coordinate column names
        pres_clean_std = pres_clean.copy()
        absence_selected_std = absence_selected.copy()

        # Ensure presence data has 'longitude' and 'latitude' columns
        if 'longitude' not in pres_clean_std.columns and 'decimalLongitude' in pres_clean_std.columns:
            pres_clean_std = pres_clean_std.rename(columns={'decimalLongitude': 'longitude', 'decimalLatitude': 'latitude'})

        # Ensure absence data has 'longitude' and 'latitude' columns
        if 'longitude' not in absence_selected_std.columns and 'decimalLongitude' in absence_selected_std.columns:
            absence_selected_std = absence_selected_std.rename(columns={'decimalLongitude': 'longitude', 'decimalLatitude': 'latitude'})

        all_df = pd.concat([pres_clean_std, absence_selected_std], ignore_index=True)

        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE MODELING FOR GENUS: {genus_name}")
        print(f"{'='*60}")

        results = {
            'genus_name': genus_name,
            'model_evaluations': {},
            'presence_count': len(pres_clean),
            'absence_count': len(absence_selected),
            'feature_count': len(feature_cols)
        }

        try:
            output_dir = "outputs/extracted_features"
            os.makedirs(output_dir, exist_ok=True)
            genus_safe = genus_name.replace(' ', '_').lower()
            presence_save_path = os.path.join(output_dir, f"{genus_safe}_presence_features.csv")
            absence_save_path = os.path.join(output_dir, f"{genus_safe}_absence_features.csv")
            pres_clean.to_csv(presence_save_path, index=False)
            absence_selected.to_csv(absence_save_path, index=False)

            # Apply bias correction with feature-based reliability weighting
            X, y, reliability_weights, bias_weights, combined_weights = self.apply_bias_correction(
                pres_clean_std, absence_selected_std, feature_cols
            )

            # Choose sample weights based on bias correction parameter
            if is_bias:
                sample_weights = bias_weights
                print(f"Using bias correction weights")
            else:
                sample_weights = reliability_weights
                print(f"Using reliability weights only (no bias correction)")

            print(f"   After bias correction and NaN filtering: {X.shape}")
            print(f"   Prepared {len(X)} total samples for modeling")

            # Configure model and loss function based on input parameters
            print(f"\n   Training with: {model_type} + {loss_func}")
            
            results_table = []
            
            try:
                # Train the specified model
                if model_type == 'Random Forest':
                    clf, X_test, y_test, y_pred, y_proba = self.RandomForest(X, y, sample_weights=sample_weights)
                elif model_type == 'Logistic Regression':
                    clf, X_test, y_test, y_pred, y_proba = self.logistic_regression_L2(X, y)
                elif model_type == 'Weighted Logistic Regression':
                    clf, X_test, y_test, y_pred, y_proba = self.train_and_evaluate_model_logistic_weighted(X, y, sample_weights=sample_weights)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Apply the specified loss function for threshold optimization
                if loss_func == 'Dice':
                    from .custom_losses import DiceScorer
                    scorer = DiceScorer(smooth=1.0)
                elif loss_func == 'Focal':
                    from .custom_losses import FocalScorer
                    scorer = FocalScorer(alpha=0.25, gamma=2.0)
                elif loss_func == 'Tversky':
                    from .custom_losses import TverskyScorer
                    scorer = TverskyScorer(alpha=alpha, beta=beta)  # Use endemic-based alpha/beta
                else:
                    scorer = None  # Original loss (no threshold optimization)
                
                # Threshold optimization based on loss function and optimization metric
                if scorer is not None:
                    thresholds = np.linspace(0.1, 0.9, 20)
                    best_score = -np.inf
                    optimal_threshold = 0.5
                    
                    print(f"   Optimizing threshold for {optimize_for} with {loss_func} loss...")
                    for threshold in thresholds:
                        score = scorer(y_test, y_proba, threshold)
                        if score > best_score:
                            best_score = score
                            optimal_threshold = threshold
                    
                    # Calculate metrics with optimal threshold
                    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                    
                    accuracy = accuracy_score(y_test, y_pred_optimal)
                    precision = precision_score(y_test, y_pred_optimal, zero_division=0)
                    recall = recall_score(y_test, y_pred_optimal, zero_division=0)
                    f1 = f1_score(y_test, y_pred_optimal, zero_division=0)
                    cm = confusion_matrix(y_test, y_pred_optimal)
                    
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    else:
                        tpr = tnr = None
                    
                    metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'confusion_matrix': cm
                    }
                    
                    print(f"      Optimal threshold: {optimal_threshold:.4f}")
                    print(f"      Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    print(f"      TPR: {tpr:.4f}, TNR: {tnr:.4f}")
                else:
                    # No threshold optimization (original loss)
                    metrics = self.evaluate_model(clf, X_test, y_test, dataset_name=f"{model_type}_{loss_func}")
                    optimal_threshold = 0.5
                    
                    # Calculate TPR and TNR
                    cm = metrics['confusion_matrix']
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    else:
                        tpr = tnr = None
                    
                    print(metrics['classification_report'])
                
                # Store results
                combination_name = f"{model_type}_{loss_func}"
                row = {
                    'model_loss': combination_name,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'tpr': tpr if tpr is not None else 0.0,
                    'tnr': tnr if tnr is not None else 0.0,
                    'optimal_threshold': optimal_threshold,
                    'optimization_metric': optimize_for
                }
                results_table.append(row)
                
            except Exception as e:
                print(f"      Error in {model_type}_{loss_func}: {str(e)}")
                results['model_evaluations'][f"{model_type}_{loss_func}"] = {
                    'error': str(e)
                }

            # Save results as CSV
            results_df = pd.DataFrame(results_table)
            output_csv = f"outputs/{genus_name}_comprehensive_results.csv"
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            results_df.to_csv(output_csv, index=False)
            print(f"\nResults table saved to: {output_csv}")

            # Print summary table
            print(f"\n{'='*80}")
            print(f"SUMMARY TABLE FOR {genus_name}")
            print(f"{'='*80}")
            print(f"{'Model/Loss':<35} {'Accuracy':<10} {'F1-Score':<10} {'TPR':<8} {'TNR':<8} {'Threshold':<10} {'Opt For':<8}")
            print(f"{'-'*80}")
            for _, row in results_df.iterrows():
                f1_score_val = row.get('f1', 'N/A')
                f1_display = f"{f1_score_val:.4f}" if isinstance(f1_score_val, (int, float)) else "N/A"
                print(f"{row['model_loss']:<35} {row['accuracy']:<10.4f} {f1_display:<10} {row['tpr']:<8.4f} {row['tnr']:<8.4f} {row['optimal_threshold']:<10.4f} {row['optimization_metric']:<8}")

            # Create prediction map with probabilities
            # if len(results_table) > 0:
            #     print(f"\nCreating prediction map for {model_type}_{loss_func}")
            #     try:
            #         # Retrain on full dataset
            #         if model_type == 'Random Forest':
            #             from sklearn.ensemble import RandomForestClassifier
            #             clf = RandomForestClassifier(n_estimators=100, random_state=42)
            #             clf.fit(X, y, sample_weight=sample_weights)
            #         elif model_type == 'Logistic Regression':
            #             from sklearn.linear_model import LogisticRegression
            #             clf = LogisticRegression(random_state=42, max_iter=1000)
            #             clf.fit(X, y, sample_weight=sample_weights)
            #         else:
            #             clf = LogisticRegression(random_state=42, max_iter=1000)
            #             clf.fit(X, y, sample_weight=sample_weights)
                    
                    # Create prediction map
                #     prediction_map_file = self.create_prediction_map_for_genus(clf, feature_cols, genus_name, 
                #                                                             pres_clean_std, absence_selected_std)
                #     if prediction_map_file:
                #         print(f"✓ Prediction map created: {prediction_map_file}")
                # except Exception as e:
                #     print(f"Warning: Could not create prediction map: {e}")

            print(f"\nModeling complete for genus: {genus_name}")
            return results

        except Exception as e:
            print(f"Error in genus modeling for {genus_name}: {str(e)}")
            results['error'] = str(e)
            return results

    def perform_feature_importance_for_all_species(self, species_list):
        """
        For each species, find the best India-level model, retrain, and perform feature importance analysis.
        Prints and visualizes feature importance for each species.
        Now also reports on missing data after feature extraction.
        Uses saved features from comprehensive_species_modeling if available.
        Also generates SHAP summary plots for model interpretability.
        """
        # Import SHAP at the beginning
        try:
            import shap
        except ImportError:
            print("SHAP library not found. Installing SHAP...")
            import subprocess
            subprocess.check_call(["pip", "install", "shap"])
            import shap

        for species_name in species_list:
            print(f"\n{'='*80}")
            print(f"Feature Importance Analysis for {species_name}")
            print(f"{'='*80}")
            # 1. Load comprehensive results CSV
            csv_path = f"outputs/{species_name}_comprehensive_results.csv"
            if not os.path.exists(csv_path):
                print(f"  Skipping {species_name}: No comprehensive results CSV found.")
                continue
            df = pd.read_csv(csv_path)
            # 2. Find best model/loss (highest accuracy)
            best_row = df.loc[df['accuracy'].idxmax()]
            best_combo = best_row['model_loss']
            print(f"  Best model/loss: {best_combo} (Accuracy: {best_row['accuracy']:.4f})")
            # 3. Parse model/loss
            if 'Random_Forest' in best_combo:
                model_type = 'rf'
            elif 'Logistic_Regression' in best_combo:
                model_type = 'logistic'
            elif 'Weighted_Logistic_Regression' in best_combo:
                model_type = 'logistic_weighted'
            else:
                print(f"  Unknown model type for {species_name}, skipping.")
                continue
            if 'Tversky' in best_combo:
                loss_type = 'tversky'
            elif 'Focal' in best_combo:
                loss_type = 'focal'
            elif 'Dice' in best_combo:
                loss_type = 'dice'
            else:
                loss_type = 'original'
            # 4. Load or extract features
            features_output_dir = "outputs/extracted_features"
            presence_features_file = os.path.join(features_output_dir, f"{species_name.replace(' ', '_').lower()}_presence_features.csv")
            absence_features_file = os.path.join(features_output_dir, f"{species_name.replace(' ', '_').lower()}_absence_features.csv")

            if os.path.exists(presence_features_file) and os.path.exists(absence_features_file):
                print(f"  Loading saved features from files...")
                presence_with_features = pd.read_csv(presence_features_file)
                absence_with_features = pd.read_csv(absence_features_file)
                print(f"  Loaded {len(presence_with_features)} presence and {len(absence_with_features)} absence feature records")
            else:
                print(f"  Saved features not found. Extracting features (this may take a while)...")
                # Prepare data (India-level) - same as in comprehensive_species_modeling
                all_points = pd.read_csv("data/testing_SDM/all_presence_point.csv")
                target_order = all_points[all_points['species'] == species_name]['order'].iloc[0]
                presence_df = all_points[all_points['species'] == species_name].copy()
                presence_df = presence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
                presence_df = presence_df[['longitude', 'latitude']]
                absence_df = all_points[all_points['order'] != target_order].copy()
                absence_df = absence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
                absence_df = absence_df[['longitude', 'latitude']]
                # Feature extraction
                features_extractor_obj = features_extractor.Feature_Extractor(ee)
                presence_with_features = features_extractor_obj.add_features(presence_df)
                absence_with_features = features_extractor_obj.add_features(absence_df)
                print(f"  Extracted features for {len(presence_with_features)} presence and {len(absence_with_features)} absence points")

            feature_cols = [col for col in presence_with_features.columns if col not in ['longitude', 'latitude']]
            # Concatenate and check for missing data
            combined_df = pd.concat([presence_with_features, absence_with_features], ignore_index=True)
            n_total = len(combined_df)
            n_missing = combined_df[feature_cols].isnull().any(axis=1).sum()
            percent_missing = 100 * n_missing / n_total if n_total > 0 else 0
            print(f"  Total points: {n_total}")
            print(f"  Points with missing features: {n_missing} ({percent_missing:.1f}%)")
            if percent_missing > 20:
                print(f"  WARNING: More than 20% of points have missing data. Feature importance results may be unreliable.")
            # Filter out points with missing data
            valid_mask = ~combined_df[feature_cols].isnull().any(axis=1)
            filtered_df = combined_df.loc[valid_mask].reset_index(drop=True)
            if len(filtered_df) == 0:
                print(f"  No valid points left after filtering missing data. Skipping {species_name}.")
                continue
            X = filtered_df[feature_cols].values
            y = np.hstack([
                np.ones(len(presence_with_features)),
                np.zeros(len(absence_with_features))
            ])[valid_mask.values]
            sample_weights = np.ones(len(y))
            # 5. Retrain best model
            modelss = Models()
            if loss_type == 'original':
                if model_type == 'rf':
                    clf, _, _, _, _ = modelss.RandomForest(X, y, sample_weights=sample_weights)
                elif model_type == 'logistic':
                    clf, _, _, _, _ = modelss.logistic_regression_L2(X, y)
                else:
                    clf, _, _, _, _ = modelss.train_and_evaluate_model_logistic_weighted(X, y, sample_weights=sample_weights)
            elif loss_type == 'tversky':
                clf = modelss.train_with_tversky_scoring(X, y, sample_weights=sample_weights, model_type=model_type)
            elif loss_type == 'focal':
                clf = modelss.train_with_focal_scoring(X, y, sample_weights=sample_weights, model_type=model_type)
            elif loss_type == 'dice':
                clf = modelss.train_with_dice_scoring(X, y, sample_weights=sample_weights, model_type=model_type)
            else:
                print(f"  Unknown loss type for {species_name}, skipping.")
                continue

            # 6. SHAP Analysis
            print(f"  Generating SHAP summary plots...")

            # Create species-specific output directory
            species_safe_name = species_name.replace(' ', '_').lower()
            species_output_dir = f"outputs/testing_SDM_out/{species_safe_name}"
            os.makedirs(species_output_dir, exist_ok=True)

            # Initialize SHAP importance as empty dict in case SHAP fails
            shap_importance = {}
            shap_importance_sorted = []

            try:
                # Create SHAP explainer based on model type
                if model_type == 'rf':
                    # For Random Forest, use TreeExplainer
                    explainer = shap.TreeExplainer(clf)
                else:
                    # For logistic regression and other models, use KernelExplainer as fallback
                    # This is more robust than LinearExplainer
                    explainer = shap.KernelExplainer(clf.predict_proba, X[:100])  # Use small sample for background

                # Calculate SHAP values (use a sample if dataset is too large)
                if len(X) > 1000:
                    print(f"    Using sample of 1000 points for SHAP analysis (dataset has {len(X)} points)")
                    sample_indices = np.random.choice(len(X), 1000, replace=False)
                    X_sample = X[sample_indices]
                else:
                    X_sample = X

                # Get SHAP values with proper error handling
                try:
                    shap_values = explainer.shap_values(X_sample)

                    # Handle different SHAP values formats
                    if isinstance(shap_values, list):
                        # For binary classification, shap_values is a list [negative_class, positive_class]
                        if len(shap_values) == 2:
                            shap_values = shap_values[1]  # Use positive class SHAP values
                        else:
                            shap_values = shap_values[0]  # Use first class if more than 2
                    elif isinstance(shap_values, np.ndarray):
                        # If it's already a numpy array, use as is
                        if len(shap_values.shape) == 3:
                            # If 3D array, take positive class
                            shap_values = shap_values[:, :, 1]
                        elif len(shap_values.shape) == 2:
                            # If 2D array, use as is
                            shap_values = shap_values
                        else:
                            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
                    else:
                        raise ValueError(f"Unexpected SHAP values type: {type(shap_values)}")

                    # Ensure shap_values is 2D
                    if len(shap_values.shape) == 1:
                        shap_values = shap_values.reshape(1, -1)

                    print(f"    SHAP values shape: {shap_values.shape}")

                except Exception as shap_error:
                    print(f"    Error calculating SHAP values: {str(shap_error)}")
                    raise shap_error

                # Create SHAP summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
                plt.title(f'SHAP Summary Plot - {species_name}', fontsize=16, fontweight='bold')

                # Save SHAP summary plot in species folder
                shap_summary_path = os.path.join(species_output_dir, "shap_summary.png")
                plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    SHAP summary plot saved to: {shap_summary_path}")

                # Create SHAP bar plot (feature importance)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type="bar", show=False)
                plt.title(f'SHAP Feature Importance - {species_name}', fontsize=16, fontweight='bold')

                # Save SHAP bar plot in species folder
                shap_bar_path = os.path.join(species_output_dir, "shap_importance.png")
                plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    SHAP feature importance plot saved to: {shap_bar_path}")

                # Calculate and print SHAP-based feature importance
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                shap_importance = dict(zip(feature_cols, mean_abs_shap))
                shap_importance_sorted = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)

                print(f"\nSHAP-based Feature Importance:")
                for feature, importance in shap_importance_sorted:
                    print(f"  {feature}: {importance:.4f}")

                # Save SHAP importance scores to CSV
                shap_importance_df = pd.DataFrame(shap_importance_sorted, columns=['Feature', 'SHAP_Importance'])
                shap_csv_path = os.path.join(species_output_dir, "shap_importance_scores.csv")
                shap_importance_df.to_csv(shap_csv_path, index=False)
                print(f"    SHAP importance scores saved to: {shap_csv_path}")

            except Exception as e:
                print(f"    Error in SHAP analysis: {str(e)}")
                print(f"    Continuing with feature sensitivity analysis...")
                # Set default values for SHAP importance
                shap_importance = {feature: 0.0 for feature in feature_cols}
                shap_importance_sorted = [(feature, 0.0) for feature in feature_cols]

            # 7. Feature sensitivity analysis (existing code)
            feature_ranges = {
                'annual_mean_temperature': (-185, 293),
                'mean_diurnal_range': (49, 163),
                'isothermality': (19, 69),
                'temperature_seasonality': (431, 11303),
                'max_temperature_warmest_month': (-51, 434),
                'min_temperature_coldest_month': (-369, 246),
                'temperature_annual_range': (74, 425),
                'mean_temperature_wettest_quarter': (-143, 339),
                'mean_temperature_driest_quarter': (-275, 309),
                'mean_temperature_warmest_quarter': (-97, 351),
                'mean_temperature_coldest_quarter': (-300, 275),
                'annual_precipitation': (51, 11401),
                'precipitation_wettest_month': (7, 2949),
                'precipitation_driest_month': (0, 81),
                'precipitation_seasonality': (27, 172),
                'precipitation_wettest_quarter': (18, 8019),
                'precipitation_driest_quarter': (0, 282),
                'precipitation_warmest_quarter': (10, 6090),
                'precipitation_coldest_quarter': (0, 5162),
                'aridity_index': (403, 65535),
                'topsoil_ph': (0, 8.3),
                'subsoil_ph': (0, 8.3),
                'topsoil_texture': (0, 3),
                'subsoil_texture': (0, 13),
                'elevation': (-54, 7548)
            }
            analyzer = FeatureSensitivityAnalyzer(clf, feature_cols, feature_ranges)
            try:
                base_point, base_prob = analyzer.find_high_probability_point(X, threshold=0.9)
                print(f"  Found point with probability: {base_prob:.4f}")
            except ValueError as e:
                print(f"  Warning: {e}. Using point with highest probability instead.")
                probs = clf.predict_proba(X)[:, 1]
                best_idx = np.argmax(probs)
                base_point = X[best_idx]
                base_prob = probs[best_idx]
                print(f"  Using point with probability: {base_prob:.4f}")
            results = analyzer.analyze_all_features(base_point, X)

            # Save feature sensitivity plots in species folder
            plot_path = os.path.join(species_output_dir, "feature_sensitivity.png")
            analyzer.plot_feature_sensitivity(results, save_path=plot_path)
            importance_scores = analyzer.get_feature_importance(results)
            print("\nFeature Sensitivity-based Importance Scores:")
            for feature, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {score:.4f}")
            print(f"  Feature sensitivity plots saved to: {plot_path}")

            # Save sensitivity-based importance scores to CSV
            sensitivity_importance_sorted = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            sensitivity_importance_df = pd.DataFrame(sensitivity_importance_sorted, columns=['Feature', 'Sensitivity_Importance'])
            sensitivity_csv_path = os.path.join(species_output_dir, "sensitivity_importance_scores.csv")
            sensitivity_importance_df.to_csv(sensitivity_csv_path, index=False)
            print(f"    Sensitivity importance scores saved to: {sensitivity_csv_path}")

            # 8. Permutation importance (scikit-learn)
            from sklearn.inspection import permutation_importance
            print(f"\n  Calculating permutation importance...")
            try:
                perm_result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, scoring='accuracy')
                perm_importance = dict(zip(feature_cols, perm_result.importances_mean))
                perm_importance_sorted = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
                print("\nPermutation Importance Scores:")
                for feature, score in perm_importance_sorted:
                    print(f"  {feature}: {score:.4f}")
                # Save to CSV
                perm_df = pd.DataFrame(perm_importance_sorted, columns=['Feature', 'Permutation_Importance'])
                perm_csv_path = os.path.join(species_output_dir, "permutation_importance_scores.csv")
                perm_df.to_csv(perm_csv_path, index=False)
                print(f"    Permutation importance scores saved to: {perm_csv_path}")
            except Exception as e:
                print(f"    Error in permutation importance: {str(e)}")
                perm_importance = {feature: 0.0 for feature in feature_cols}
                perm_importance_sorted = [(feature, 0.0) for feature in feature_cols]

            # 9. Create combined importance comparison (now with permutation importance)
            combined_importance = {}
            for feature in feature_cols:
                combined_importance[feature] = {
                    'SHAP_Importance': shap_importance.get(feature, 0.0),
                    'Sensitivity_Importance': importance_scores.get(feature, 0.0),
                    'Permutation_Importance': perm_importance.get(feature, 0.0)
                }
            combined_df = pd.DataFrame(combined_importance).T.reset_index()
            combined_df.columns = ['Feature', 'SHAP_Importance', 'Sensitivity_Importance', 'Permutation_Importance']
            # Sort by SHAP importance, but handle case where all SHAP values might be 0
            if combined_df['SHAP_Importance'].sum() > 0:
                combined_df = combined_df.sort_values('SHAP_Importance', ascending=False)
            elif combined_df['Permutation_Importance'].sum() > 0:
                combined_df = combined_df.sort_values('Permutation_Importance', ascending=False)
            else:
                combined_df = combined_df.sort_values('Sensitivity_Importance', ascending=False)
            combined_csv_path = os.path.join(species_output_dir, "combined_importance_scores.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"    Combined importance scores saved to: {combined_csv_path}")
            print(f"\nAll results for {species_name} saved to: {species_output_dir}")

    def perform_feature_importance_for_all_genera(self, genus_list):
        """
        For each genus, find the best model (India-level or ecoregion-level), retrain, and perform feature importance analysis.
        Prints and visualizes feature importance for each genus.
        Now also reports on missing data after feature extraction.
        Uses saved features from comprehensive_genus_modeling or train_on_max_ecoregion_for_genus if available.
        Also generates SHAP summary plots for model interpretability.
        """
        # Import SHAP at the beginning
        try:
            import shap
        except ImportError:
            print("SHAP library not found. Installing SHAP...")
            import subprocess
            subprocess.check_call(["pip", "install", "shap"])
            import shap

        import os

        for genus_name in genus_list:
            print(f"\n{'='*80}")
            print(f"Feature Importance Analysis for {genus_name}")
            print(f"{'='*80}")

            # 1. Check for both India-level and ecoregion-level results
            comprehensive_csv_path = f"outputs/{genus_name}_comprehensive_results.csv"
            ecoregion_csv_path = f"outputs/{genus_name}_ecoregion_results.csv"

            if os.path.exists(comprehensive_csv_path):
                csv_path = comprehensive_csv_path
                analysis_type = "India-level"
                print(f"  Using {analysis_type} results from: {csv_path}")
            elif os.path.exists(ecoregion_csv_path):
                csv_path = ecoregion_csv_path
                analysis_type = "ecoregion-level"
                print(f"  Using {analysis_type} results from: {csv_path}")
            else:
                print(f"  Skipping {genus_name}: No comprehensive or ecoregion results CSV found.")
                continue

            df = pd.read_csv(csv_path)
            # 2. Find best model/loss (highest accuracy)
            best_row = df.loc[df['accuracy'].idxmax()]
            best_combo = best_row['model_loss']
            print(f"  Best model/loss: {best_combo} (Accuracy: {best_row['accuracy']:.4f})")
            # 3. Parse model/loss
            if 'Random_Forest' in best_combo:
                model_type = 'rf'
            elif 'Logistic_Regression' in best_combo:
                model_type = 'logistic'
            elif 'Weighted_Logistic_Regression' in best_combo:
                model_type = 'logistic_weighted'
            else:
                print(f"  Unknown model type for {genus_name}, skipping.")
                continue
            if 'Tversky' in best_combo:
                loss_type = 'tversky'
            elif 'Focal' in best_combo:
                loss_type = 'focal'
            elif 'Dice' in best_combo:
                loss_type = 'dice'
            else:
                loss_type = 'original'
            # 4. Load or extract features
            features_output_dir = "outputs/extracted_features"
            genus_safe = genus_name.replace(' ', '_').lower()

            # Check for both types of feature files
            presence_features_file = os.path.join(features_output_dir, f"{genus_safe}_presence_features.csv")
            absence_features_file = os.path.join(features_output_dir, f"{genus_safe}_absence_features.csv")
            ecoregion_presence_features_file = os.path.join(features_output_dir, f"{genus_safe}_ecoregion_presence_features.csv")
            ecoregion_absence_features_file = os.path.join(features_output_dir, f"{genus_safe}_ecoregion_absence_features.csv")

            if analysis_type == "India-level" and os.path.exists(presence_features_file) and os.path.exists(absence_features_file):
                print(f"  Loading saved India-level features from files...")
                presence_with_features = pd.read_csv(presence_features_file)
                absence_with_features = pd.read_csv(absence_features_file)
                print(f"  Loaded {len(presence_with_features)} presence and {len(absence_with_features)} absence feature records")
            elif analysis_type == "ecoregion-level" and os.path.exists(ecoregion_presence_features_file) and os.path.exists(ecoregion_absence_features_file):
                print(f"  Loading saved ecoregion-level features from files...")
                presence_with_features = pd.read_csv(ecoregion_presence_features_file)
                absence_with_features = pd.read_csv(ecoregion_absence_features_file)
                print(f"  Loaded {len(presence_with_features)} presence and {len(absence_with_features)} absence feature records")
            else:
                print(f"  Saved features not found. Extracting features (this may take a while)...")
                # Prepare data based on analysis type
                all_points = pd.read_csv("data/testing_SDM/all_presence_point.csv")

                if analysis_type == "India-level":
                    # India-level data preparation (same as in comprehensive_genus_modeling)
                    presence_df = all_points[all_points['genus'] == genus_name].copy()
                    presence_df = presence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
                    presence_df = presence_df[['longitude', 'latitude']]
                    presence_coords = set(zip(presence_df['longitude'], presence_df['latitude']))
                    absence_df = all_points[all_points['genus'] != genus_name].copy()
                    absence_df = absence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
                    absence_df = absence_df[['longitude', 'latitude']]
                    absence_df = absence_df[~absence_df.apply(lambda r: (r['longitude'], r['latitude']) in presence_coords, axis=1)]
                    if len(absence_df) > len(presence_df):
                        absence_df = absence_df.sample(n=len(presence_df), random_state=42)
                else:
                    # Ecoregion-level data preparation (same as in train_on_max_ecoregion_for_genus)
                    from shapely.wkt import loads as load_wkt
                    from shapely.geometry import Point
                    import os

                    presence_df = all_points[all_points['genus'] == genus_name].copy()
                    if len(presence_df) == 0:
                        print(f"  No presence points found for genus: {genus_name}, skipping.")
                        continue

                    # Load all ecoregion polygons
                    eco_dir = "data/eco_regions_polygon"
                    eco_polygons = {}
                    for fname in os.listdir(eco_dir):
                        if fname.endswith('.wkt'):
                            eco_name = fname.replace('.wkt', '')
                            with open(os.path.join(eco_dir, fname), 'r') as f:
                                eco_polygons[eco_name] = load_wkt(f.read().strip())

                    # Find which ecoregion has the most presence points
                    presence_points = [Point(lon, lat) for lon, lat in zip(presence_df['decimalLongitude'], presence_df['decimalLatitude'])]
                    eco_counts = {eco: 0 for eco in eco_polygons}
                    eco_assignments = []
                    for pt in presence_points:
                        found = False
                        for eco, poly in eco_polygons.items():
                            if poly.contains(pt):
                                eco_counts[eco] += 1
                                eco_assignments.append(eco)
                                found = True
                                break
                        if not found:
                            eco_assignments.append(None)

                    # Get the ecoregion with the maximum count
                    max_eco = max(eco_counts, key=eco_counts.get)
                    print(f"  Using ecoregion: {max_eco} ({eco_counts[max_eco]} points)")

                    # Filter presence points to only those in the max ecoregion
                    presence_df = presence_df[[eco == max_eco for eco in eco_assignments]].copy()
                    presence_df = presence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
                    presence_df = presence_df[['longitude', 'latitude']]

                    # Prepare pseudo-absence points: use points from other genera, not in presence, and in the same ecoregion
                    presence_coords = set(zip(presence_df['longitude'], presence_df['latitude']))
                    absence_df = all_points[(all_points['genus'] != genus_name)].copy()
                    absence_df = absence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
                    absence_df = absence_df[['longitude', 'latitude']]

                    # Only keep pseudo-absence points in the max ecoregion and not in presence
                    initial_absence_count = len(absence_df)
                    pseudo_absence_points = [Point(lon, lat) for lon, lat in zip(absence_df['longitude'], absence_df['latitude'])]
                    absence_df = absence_df[[eco_polygons[max_eco].contains(pt) and (pt.x, pt.y) not in presence_coords for pt in pseudo_absence_points]]
                    filtered_absence_count = len(absence_df)
                    removed_duplicates = initial_absence_count - filtered_absence_count

                    # Sample the same number of pseudo-absences as presences (or all if fewer available)
                    if len(absence_df) > len(presence_df):
                        absence_df = absence_df.sample(n=len(presence_df), random_state=42)

                # Feature extraction
                features_extractor_obj = features_extractor.Feature_Extractor(ee)
                presence_with_features = features_extractor_obj.add_features(presence_df)
                absence_with_features = features_extractor_obj.add_features(absence_df)
                print(f"  Extracted features for {len(presence_with_features)} presence and {len(absence_with_features)} absence points")

            feature_cols = [col for col in presence_with_features.columns if col not in ['longitude', 'latitude']]
            # Concatenate and check for missing data
            combined_df = pd.concat([presence_with_features, absence_with_features], ignore_index=True)
            n_total = len(combined_df)
            n_missing = combined_df[feature_cols].isnull().any(axis=1).sum()
            percent_missing = 100 * n_missing / n_total if n_total > 0 else 0
            print(f"  Total points: {n_total}")
            print(f"  Points with missing features: {n_missing} ({percent_missing:.1f}%)")
            if percent_missing > 20:
                print(f"  WARNING: More than 20% of points have missing data. Feature importance results may be unreliable.")
            # Filter out points with missing data
            valid_mask = ~combined_df[feature_cols].isnull().any(axis=1)
            filtered_df = combined_df.loc[valid_mask].reset_index(drop=True)
            if len(filtered_df) == 0:
                print(f"  No valid points left after filtering missing data. Skipping {genus_name}.")
                continue
            X = filtered_df[feature_cols].values
            y = np.hstack([
                np.ones(len(presence_with_features)),
                np.zeros(len(absence_with_features))
            ])[valid_mask.values]
            sample_weights = np.ones(len(y))
            # 5. Retrain best model
            modelss = Models()
            if loss_type == 'original':
                if model_type == 'rf':
                    clf, _, _, _, _ = modelss.RandomForest(X, y, sample_weights=sample_weights)
                elif model_type == 'logistic':
                    clf, _, _, _, _ = modelss.logistic_regression_L2(X, y)
                else:
                    clf, _, _, _, _ = modelss.train_and_evaluate_model_logistic_weighted(X, y, sample_weights=sample_weights)
            elif loss_type == 'tversky':
                clf = modelss.train_with_tversky_scoring(X, y, sample_weights=sample_weights, model_type=model_type)
            elif loss_type == 'focal':
                clf = modelss.train_with_focal_scoring(X, y, sample_weights=sample_weights, model_type=model_type)
            elif loss_type == 'dice':
                clf = modelss.train_with_dice_scoring(X, y, sample_weights=sample_weights, model_type=model_type)
            else:
                print(f"  Unknown loss type for {genus_name}, skipping.")
                continue

            # 6. SHAP Analysis
            print(f"  Generating SHAP summary plots...")

            # Create genus-specific output directory with analysis type
            genus_safe_name = genus_name.replace(' ', '_').lower()
            genus_output_dir = f"outputs/testing_SDM_out/{genus_safe_name}_{analysis_type.replace('-', '_').lower()}"
            os.makedirs(genus_output_dir, exist_ok=True)

            # Initialize SHAP importance as empty dict in case SHAP fails
            shap_importance = {}
            shap_importance_sorted = []

            try:
                # Create SHAP explainer based on model type
                if model_type == 'rf':
                    # For Random Forest, use TreeExplainer
                    explainer = shap.TreeExplainer(clf)
                else:
                    # For logistic regression and other models, use KernelExplainer as fallback
                    # This is more robust than LinearExplainer
                    explainer = shap.KernelExplainer(clf.predict_proba, X[:100])  # Use small sample for background

                # Calculate SHAP values (use a sample if dataset is too large)
                if len(X) > 1000:
                    print(f"    Using sample of 1000 points for SHAP analysis (dataset has {len(X)} points)")
                    sample_indices = np.random.choice(len(X), 1000, replace=False)
                    X_sample = X[sample_indices]
                else:
                    X_sample = X

                # Get SHAP values with proper error handling
                try:
                    shap_values = explainer.shap_values(X_sample)

                    # Handle different SHAP values formats
                    if isinstance(shap_values, list):
                        # For binary classification, shap_values is a list [negative_class, positive_class]
                        if len(shap_values) == 2:
                            shap_values = shap_values[1]  # Use positive class SHAP values
                        else:
                            shap_values = shap_values[0]  # Use first class if more than 2
                    elif isinstance(shap_values, np.ndarray):
                        # If it's already a numpy array, use as is
                        if len(shap_values.shape) == 3:
                            # If 3D array, take positive class
                            shap_values = shap_values[:, :, 1]
                        elif len(shap_values.shape) == 2:
                            # If 2D array, use as is
                            shap_values = shap_values
                        else:
                            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
                    else:
                        raise ValueError(f"Unexpected SHAP values type: {type(shap_values)}")

                    # Ensure shap_values is 2D
                    if len(shap_values.shape) == 1:
                        shap_values = shap_values.reshape(1, -1)

                    print(f"    SHAP values shape: {shap_values.shape}")

                except Exception as shap_error:
                    print(f"    Error calculating SHAP values: {str(shap_error)}")
                    raise shap_error

                # Create SHAP summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
                plt.title(f'SHAP Summary Plot - {genus_name} ({analysis_type})', fontsize=16, fontweight='bold')

                # Save SHAP summary plot in genus folder
                shap_summary_path = os.path.join(genus_output_dir, "shap_summary.png")
                plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    SHAP summary plot saved to: {shap_summary_path}")

                # Create SHAP bar plot (feature importance)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type="bar", show=False)
                plt.title(f'SHAP Feature Importance - {genus_name} ({analysis_type})', fontsize=16, fontweight='bold')

                # Save SHAP bar plot in genus folder
                shap_bar_path = os.path.join(genus_output_dir, "shap_importance.png")
                plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    SHAP feature importance plot saved to: {shap_bar_path}")

                # Calculate and print SHAP-based feature importance
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                shap_importance = dict(zip(feature_cols, mean_abs_shap))
                shap_importance_sorted = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)

                print(f"\nSHAP-based Feature Importance:")
                for feature, importance in shap_importance_sorted:
                    print(f"  {feature}: {importance:.4f}")

                # Save SHAP importance scores to CSV
                shap_importance_df = pd.DataFrame(shap_importance_sorted, columns=['Feature', 'SHAP_Importance'])
                shap_csv_path = os.path.join(genus_output_dir, "shap_importance_scores.csv")
                shap_importance_df.to_csv(shap_csv_path, index=False)
                print(f"    SHAP importance scores saved to: {shap_csv_path}")

            except Exception as e:
                print(f"    Error in SHAP analysis: {str(e)}")
                print(f"    Continuing with feature sensitivity analysis...")
                # Set default values for SHAP importance
                shap_importance = {feature: 0.0 for feature in feature_cols}
                shap_importance_sorted = [(feature, 0.0) for feature in feature_cols]

            # 7. Feature sensitivity analysis (existing code)
            feature_ranges = {
                'annual_mean_temperature': (-185, 293),
                'mean_diurnal_range': (49, 163),
                'isothermality': (19, 69),
                'temperature_seasonality': (431, 11303),
                'max_temperature_warmest_month': (-51, 434),
                'min_temperature_coldest_month': (-369, 246),
                'temperature_annual_range': (74, 425),
                'mean_temperature_wettest_quarter': (-143, 339),
                'mean_temperature_driest_quarter': (-275, 309),
                'mean_temperature_warmest_quarter': (-97, 351),
                'mean_temperature_coldest_quarter': (-300, 275),
                'annual_precipitation': (51, 11401),
                'precipitation_wettest_month': (7, 2949),
                'precipitation_driest_month': (0, 81),
                'precipitation_seasonality': (27, 172),
                'precipitation_wettest_quarter': (18, 8019),
                'precipitation_driest_quarter': (0, 282),
                'precipitation_warmest_quarter': (10, 6090),
                'precipitation_coldest_quarter': (0, 5162),
                'aridity_index': (403, 65535),
                'topsoil_ph': (0, 8.3),
                'subsoil_ph': (0, 8.3),
                'topsoil_texture': (0, 3),
                'subsoil_texture': (0, 13),
                'elevation': (-54, 7548)
            }
            analyzer = FeatureSensitivityAnalyzer(clf, feature_cols, feature_ranges)
            try:
                base_point, base_prob = analyzer.find_high_probability_point(X, threshold=0.9)
                print(f"  Found point with probability: {base_prob:.4f}")
            except ValueError as e:
                print(f"  Warning: {e}. Using point with highest probability instead.")
                probs = clf.predict_proba(X)[:, 1]
                best_idx = np.argmax(probs)
                base_point = X[best_idx]
                base_prob = probs[best_idx]
                print(f"  Using point with probability: {base_prob:.4f}")
            results = analyzer.analyze_all_features(base_point, X)

            # Save feature sensitivity plots in genus folder
            plot_path = os.path.join(genus_output_dir, "feature_sensitivity.png")
            analyzer.plot_feature_sensitivity(results, save_path=plot_path)
            importance_scores = analyzer.get_feature_importance(results)
            print("\nFeature Sensitivity-based Importance Scores:")
            for feature, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {score:.4f}")
            print(f"  Feature sensitivity plots saved to: {plot_path}")

            # Save sensitivity-based importance scores to CSV
            sensitivity_importance_sorted = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            sensitivity_importance_df = pd.DataFrame(sensitivity_importance_sorted, columns=['Feature', 'Sensitivity_Importance'])
            sensitivity_csv_path = os.path.join(genus_output_dir, "sensitivity_importance_scores.csv")
            sensitivity_importance_df.to_csv(sensitivity_csv_path, index=False)
            print(f"    Sensitivity importance scores saved to: {sensitivity_csv_path}")

            # 8. Permutation importance (scikit-learn)
            from sklearn.inspection import permutation_importance
            print(f"\n  Calculating permutation importance...")
            try:
                perm_result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, scoring='accuracy')
                perm_importance = dict(zip(feature_cols, perm_result.importances_mean))
                perm_importance_sorted = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
                print("\nPermutation Importance Scores:")
                for feature, score in perm_importance_sorted:
                    print(f"  {feature}: {score:.4f}")
                # Save to CSV
                perm_df = pd.DataFrame(perm_importance_sorted, columns=['Feature', 'Permutation_Importance'])
                perm_csv_path = os.path.join(genus_output_dir, "permutation_importance_scores.csv")
                perm_df.to_csv(perm_csv_path, index=False)
                print(f"    Permutation importance scores saved to: {perm_csv_path}")
            except Exception as e:
                print(f"    Error in permutation importance: {str(e)}")
                perm_importance = {feature: 0.0 for feature in feature_cols}
                perm_importance_sorted = [(feature, 0.0) for feature in feature_cols]

            # 9. Create combined importance comparison (now with permutation importance)
            combined_importance = {}
            for feature in feature_cols:
                combined_importance[feature] = {
                    'SHAP_Importance': shap_importance.get(feature, 0.0),
                    'Sensitivity_Importance': importance_scores.get(feature, 0.0),
                    'Permutation_Importance': perm_importance.get(feature, 0.0)
                }
            combined_df = pd.DataFrame(combined_importance).T.reset_index()
            combined_df.columns = ['Feature', 'SHAP_Importance', 'Sensitivity_Importance', 'Permutation_Importance']
            # Sort by SHAP importance, but handle case where all SHAP values might be 0
            if combined_df['SHAP_Importance'].sum() > 0:
                combined_df = combined_df.sort_values('SHAP_Importance', ascending=False)
            elif combined_df['Permutation_Importance'].sum() > 0:
                combined_df = combined_df.sort_values('Permutation_Importance', ascending=False)
            else:
                combined_df = combined_df.sort_values('Sensitivity_Importance', ascending=False)
            combined_csv_path = os.path.join(genus_output_dir, "combined_importance_scores.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"    Combined importance scores saved to: {combined_csv_path}")
            print(f"\nAll results for {genus_name} ({analysis_type}) saved to: {genus_output_dir}")

    def apply_bias_correction(self, presence_df, absence_df, feature_cols):
        """
        Apply bias correction to presence and absence data using ecoregion-based weighting.
        Now includes feature-based reliability calculation for absence points.

        Parameters:
        -----------
        presence_df : DataFrame
            Presence points with features
        absence_df : DataFrame
            Absence points with features
        feature_cols : list
            List of feature column names

        Returns:
        --------
        tuple
            X (features), y (labels), reliability_weights, bias_weights, combined_weights
        """
        import os
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely import wkt
        from sklearn.utils import shuffle

        print(f"   Applying bias correction with feature-based reliability...")

        # ------------------------------------
        # 1. Feature-based Reliability Weights
        # ------------------------------------
        # For presence samples, set weight = 1 (assume high reliability)
        reliability_presence = np.ones(len(presence_df))

        # For absence samples, calculate feature-based reliability scores
        print(f"   Calculating feature-based reliability for {len(absence_df)} absence points...")

        # Get presence features for comparison
        presence_features = presence_df[feature_cols].dropna().values.astype(float)

        absence_df = absence_df.dropna(subset=feature_cols)
        reliability_scores = []

        for idx, row in absence_df.iterrows():
            absence_features = row[feature_cols].values.astype(float)
            reliability = calculate_feature_based_reliability(absence_features, presence_features, threshold=0.03)
            reliability_scores.append(reliability)


        # Add reliability scores to absence dataframe
        absence_df_with_reliability = absence_df.copy()
        absence_df_with_reliability['reliability_score'] = reliability_scores

        # Print reliability statistics
        print(f"   Absence point reliability - Mean: {np.mean(reliability_scores):.3f}, "
              f"Min: {np.min(reliability_scores):.3f}, "
              f"Max: {np.max(reliability_scores):.3f}")

        # Normalize reliability values to [0,1] range (same as custom_loss_trainers.py)
        min_rel = np.min(reliability_scores)
        max_rel = np.max(reliability_scores)
        print(min_rel)
        print(max_rel)
        if max_rel != min_rel:
            reliability_absence = (np.array(reliability_scores) - min_rel) / (max_rel - min_rel)
        else:
            reliability_absence = np.ones(len(reliability_scores))

        # Combine reliability weights for presence and absence data
        reliability_weights = np.hstack([reliability_presence, reliability_absence])

        # ------------------------------------
        # 2. Bias-correction Weights
        # ------------------------------------
        # Read eco-region counts file to understand sampling density per ecoregion
        counts_file = "outputs/testing_SDM_out/species_ecoregion_count_1.csv"
        if os.path.exists(counts_file):
            region_counts_df = pd.read_csv(counts_file)

            # Compute raw weight: inverse relationship with count (fewer samples = higher weight)
            region_counts_df['raw_weight'] = 1 / (region_counts_df['count'] + 1)

            # Normalize raw weights to a subtle range [0.5, 1.5] to avoid extreme values
            min_w = region_counts_df['raw_weight'].min()
            max_w = region_counts_df['raw_weight'].max()
            if max_w != min_w:
                region_counts_df['eco_weight'] = 0.5 + (region_counts_df['raw_weight'] - min_w) / (max_w - min_w)
            else:
                region_counts_df['eco_weight'] = 1.0

            # Create mapping dictionary: eco_region -> eco_weight
            eco_weight_dict = region_counts_df.set_index('ecoregion')['eco_weight'].to_dict()
            print(f"   Loaded ecoregion weights for {len(eco_weight_dict)} regions")
        else:
            print(f"   Warning: {counts_file} not found. Defaulting eco weights to 1.")
            eco_weight_dict = {}

        # To assign bias weights, we need to determine which eco-region each point falls into
        ecoregion_folder = "data/eco_regions_polygon"

        # Convert lat/lon coordinates to Point geometries for spatial operations

        presence_df_copy = presence_df.copy()
        absence_df_copy = absence_df.copy()
        presence_df_copy["geometry"] = presence_df_copy.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        absence_df_copy["geometry"] = absence_df_copy.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)

        print("reached2")

        # Create GeoDataFrames with proper coordinate reference system (WGS84)
        presence_gdf = gpd.GeoDataFrame(presence_df_copy, geometry="geometry", crs="EPSG:4326")
        absence_gdf = gpd.GeoDataFrame(absence_df_copy, geometry="geometry", crs="EPSG:4326")

        # Combine presence and absence data while preserving original ordering
        combined_gdf = pd.concat([presence_gdf, absence_gdf], ignore_index=True)

        # Function to load eco-region polygons from WKT (Well-Known Text) files
        def load_ecoregions(folder):
            """Load ecoregion polygons from WKT files in the specified folder"""
            ecoregions = []
            for file in os.listdir(folder):
                if file.endswith(".wkt"):
                    with open(os.path.join(folder, file), "r") as f:
                        wkt_text = f.read().strip()
                        poly = wkt.loads(wkt_text)  # Parse WKT to geometry
                        ecoregions.append({"ecoregion": file.replace(".wkt", ""), "geometry": poly})
            return gpd.GeoDataFrame(ecoregions, geometry="geometry", crs="EPSG:4326")

        # Load eco-region polygons from WKT files
        ecoregion_gdf = load_ecoregions(ecoregion_folder)

        # Perform spatial join to assign each point to its corresponding eco-region
        # Uses "within" predicate to find which polygon contains each point
        combined_with_ecoregion = gpd.sjoin(combined_gdf, ecoregion_gdf, how="left", predicate="within")

        # Define function to retrieve bias weight for a given eco-region
        def get_bias_weight(eco):
            """Return bias weight for ecoregion, default to 1 if not found or NaN"""
            if pd.isna(eco):
                return 1
            else:
                return eco_weight_dict.get(eco, 1)

        # Apply the mapping to get bias weights for each data point
        bias_weights = combined_with_ecoregion['ecoregion'].apply(get_bias_weight).values

        # Save bias weights with coordinates to CSV for inspection and debugging
        coords_bias = np.column_stack((combined_with_ecoregion.geometry.x, combined_with_ecoregion.geometry.y))
        bias_df = pd.DataFrame(coords_bias, columns=["longitude", "latitude"])
        bias_df["bias_weight"] = bias_weights
        output_bias_file = "outputs/bias_weights.csv"
        os.makedirs(os.path.dirname(output_bias_file), exist_ok=True)
        bias_df.to_csv(output_bias_file, index=False)
        print(f"   Bias weights saved to {output_bias_file}")

        # ------------------------------------
        # 3. Feature Extraction & Combination
        # ------------------------------------
        # Extract environmental features from the original CSV data
        presence_features = presence_df[feature_cols].values
        absence_features = absence_df[feature_cols].values

        # Combine features and create binary labels (1=presence, 0=absence)
        X = np.vstack([presence_features, absence_features])
        y = np.hstack([np.ones(len(presence_features)), np.zeros(len(absence_features))])

        # ------------------------------------
        # Shuffle the data along with both sets of weights
        # ------------------------------------
        # Randomly shuffle all arrays together to ensure proper randomization
        X, y, reliability_weights, bias_weights = shuffle(
            X, y, reliability_weights, bias_weights, random_state=42
        )

        # Combine weights
        combined_weights = reliability_weights * bias_weights

        print(f"   Applied bias correction to {len(X)} samples")
        print(f"   Reliability weights range: [{np.min(reliability_weights):.3f}, {np.max(reliability_weights):.3f}]")
        print(f"   Bias weights range: [{np.min(bias_weights):.3f}, {np.max(bias_weights):.3f}]")
        print(f"   Combined weights range: [{np.min(combined_weights):.3f}, {np.max(combined_weights):.3f}]")

        return X, y, reliability_weights, bias_weights, combined_weights

    # def create_prediction_map_for_genus(self, clf, feature_cols, genus_name, presence_df, absence_df):
    #     """
    #     Create a folium map showing model predictions across India with probabilities.
        
    #     Parameters:
    #     -----------
    #     clf : trained classifier
    #     feature_cols : list of feature column names
    #     genus_name : str, name of the genus
    #     presence_df : DataFrame with presence points
    #     absence_df : DataFrame with absence points
        
    #     Returns:
    #     --------
    #     str : path to the generated HTML file
    #     """
    #     try:
    #         import folium
    #         from folium import plugins
    #         import numpy as np
    #         import pandas as pd
    #     except ImportError:
    #         print("Required libraries not installed. Installing...")
    #         import subprocess
    #         subprocess.check_call(["pip", "install", "folium"])
    #         import folium
    #         from folium import plugins
    #         import numpy as np
    #         import pandas as pd
        
    #     print(f"Creating prediction map for {genus_name}...")
        
    #     try:
    #         # Create output file
    #         output_file = f'outputs/testing_SDM_out/{genus_name.replace(" ", "_")}_prediction_map.html'
    #         output_dir = os.path.dirname(output_file)
    #         if output_dir:
    #             os.makedirs(output_dir, exist_ok=True)
        
    #         # Create a grid of points across India for prediction
    #         lat_min, lat_max = 6.5, 37.5
    #         lon_min, lon_max = 68.0, 97.0
            
    #         # Create grid with 0.5 degree spacing
    #         lats = np.arange(lat_min, lat_max + 0.5, 0.5)
    #         lons = np.arange(lon_min, lon_max + 0.5, 0.5)
            
    #         # Create grid points
    #         grid_points = []
    #         for lat in lats:
    #             for lon in lons:
    #                 grid_points.append([lat, lon])
            
    #         print(f"Created {len(grid_points)} grid points for prediction")
        
    #         # Extract features for grid points using Earth Engine
    #         try:
    #             import ee
    #             from .features_extractor import Feature_Extractor
                
    #             # Initialize Earth Engine if not already done
                
    #             ee.Authenticate()
    #             ee.Initialize(project='ee-mtpictd')
                
    #             # Create DataFrame for grid points
    #             grid_df = pd.DataFrame(grid_points, columns=['latitude', 'longitude'])
                
    #             # Extract features for grid points
    #             fe = Feature_Extractor(ee)
    #             grid_with_features = fe.add_features(grid_df)
                
    #             # Prepare features for prediction
    #             X_grid = grid_with_features[feature_cols].values.astype(float)
                
    #             # Make predictions
    #             probabilities = clf.predict_proba(X_grid)[:, 1]  # Probability of presence
                
    #             # Add probabilities to grid DataFrame
    #             grid_with_features['probability'] = probabilities
                
    #             # Add elevation data if available
    #             if 'elevation' in grid_with_features.columns:
    #                 print(f"Elevation data found in features")
    #             else:
    #                 print(f"Warning: Elevation not found in features")
    #                 grid_with_features['elevation'] = 0  # Default elevation
                
    #             print(f"Generated predictions for {len(grid_with_features)} grid points")
            
    #             # Analyze correlation between probability and elevation
    #             if 'elevation' in grid_with_features.columns and grid_with_features['elevation'].sum() > 0:
    #                 prob_elev_corr = grid_with_features['probability'].corr(grid_with_features['elevation'])
    #                 print(f"Probability-Elevation correlation: {prob_elev_corr:.4f}")
                    
    #                 # Print elevation statistics
    #                 elev_stats = grid_with_features['elevation'].describe()
    #                 print(f"Elevation statistics:")
    #                 print(f"  Mean: {elev_stats['mean']:.1f}m")
    #                 print(f"  Min: {elev_stats['min']:.1f}m")
    #                 print(f"  Max: {elev_stats['max']:.1f}m")
                    
    #                 # Print probability statistics by elevation ranges
    #                 high_elev = grid_with_features[grid_with_features['elevation'] > 1000]
    #                 low_elev = grid_with_features[grid_with_features['elevation'] <= 1000]
                    
    #                 if len(high_elev) > 0:
    #                     print(f"High elevation (>1000m) - Mean probability: {high_elev['probability'].mean():.3f}")
    #                 if len(low_elev) > 0:
    #                     print(f"Low elevation (≤1000m) - Mean probability: {low_elev['probability'].mean():.3f}")
                
    #         except Exception as e:
    #             print(f"Error extracting features for grid: {e}")
    #             return None
        
    #     except Exception as e:
    #         print(f"Error creating prediction map: {e}")
    #         return None
        
    #     # Create the map centered on India
    #     m = folium.Map(
    #         location=[22.0, 78.0],  # Center of India
    #         zoom_start=5,
    #         tiles='OpenStreetMap'
    #     )
        
        
        
    #     # Add grid points colored by probability and sized by elevation
    #     for idx, row in grid_with_features.iterrows():
    #         lat, lon = row['latitude'], row['longitude']
    #         prob = row['probability']
    #         elevation = row['elevation']
            
    #         # Color based on probability: red (low) to green (high)
    #         if prob < 0.3:
    #             color = 'red'
    #         elif prob < 0.6:
    #             color = 'orange'
    #         else:
    #             color = 'green'
            
    #         # Size based on elevation (larger = higher elevation)
    #         # Scale elevation to reasonable marker size (2-12 pixels)
    #         if elevation > 0:
    #             radius = max(2, min(12, int(elevation / 200) + 2))
    #         else:
    #             radius = 3  # Default size for unknown elevation
            
    #         # Create popup with both probability and elevation
    #         popup_text = f'Probability: {prob:.3f}<br>Elevation: {elevation:.0f}m'
            
    #         folium.CircleMarker(
    #             location=[lat, lon],
    #             radius=radius,
    #             color=color,
    #             fill=True,
    #             fillColor=color,
    #             fillOpacity=0.7,
    #             weight=1,
    #             popup=popup_text
    #         ).add_to(m)
        
    #     # Add presence points (blue)
    #     presence_group = folium.FeatureGroup(name=f'Actual Presence (Blue, n={len(presence_df)})', overlay=True)
    #     for _, row in presence_df.iterrows():
    #         folium.CircleMarker(
    #             location=[row['latitude'], row['longitude']],
    #             radius=3,
    #             color='blue',
    #             fill=True,
    #             fillColor='blue',
    #             fillOpacity=0.8,
    #             weight=2
    #         ).add_to(presence_group)
    #     presence_group.add_to(m)
        
    #     # Add absence points (red)
    #     absence_group = folium.FeatureGroup(name=f'Training Absence (Red, n={len(absence_df)})', overlay=True)
    #     for _, row in absence_df.iterrows():
    #         folium.CircleMarker(
    #             location=[row['latitude'], row['longitude']],
    #             radius=2,
    #             color='red',
    #             fill=True,
    #             fillColor='red',
    #             fillOpacity=0.6,
    #             weight=1
    #         ).add_to(absence_group)
    #     absence_group.add_to(m)
        
    #     # Add layer control
    #     folium.LayerControl(collapsed=False).add_to(m)
        
    #     # Add legend to explain color and size coding
    #     legend_html = '''
    #     <div style="position: fixed; 
    #                 bottom: 50px; left: 50px; width: 200px; height: 120px; 
    #                 background-color: white; border:2px solid grey; z-index:9999; 
    #                 font-size:14px; padding: 10px">
    #     <p><b>Legend</b></p>
    #     <p><i class="fa fa-circle" style="color:red"></i> Low Probability (&lt;0.3)</p>
    #     <p><i class="fa fa-circle" style="color:orange"></i> Medium Probability (0.3-0.6)</p>
    #     <p><i class="fa fa-circle" style="color:green"></i> High Probability (&gt;0.6)</p>
    #     <p><b>Size:</b> Larger = Higher Elevation</p>
    #     </div>
    #     '''
    #     m.get_root().html.add_child(folium.Element(legend_html))
        
    #     # Save the map
    #     m.save(output_file)
    #     print(f"Prediction map saved to: {output_file}")
        
    #     return output_file
        
        # except Exception as e:
        #     print(f"Error creating prediction map: {e}")
        #     return None

    
    
    # def create_prediction_map_for_genus(self, clf, feature_cols, genus_name, presence_df, absence_df):
    #     """
    #     Create an interactive Folium map showing:
    #     - Probability of species presence (color-coded)
    #     - Elevation (shown via marker size, popup, and heatmap)

    #     Parameters
    #     ----------
    #     clf : trained classifier
    #         Model used for prediction.
    #     feature_cols : list of str
    #         Feature column names used for prediction.
    #     genus_name : str
    #         Name of the genus.
    #     presence_df : pd.DataFrame
    #         DataFrame with presence points.
    #     absence_df : pd.DataFrame
    #         DataFrame with absence points.

    #     Returns
    #     -------
    #     str : path to the generated HTML map file
    #     """
    #     import os
    #     import folium
    #     import numpy as np
    #     import pandas as pd
    #     from folium import plugins

    #     print(f"Creating prediction map for {genus_name}...")

    #     output_file = f'outputs/testing_SDM_out/{genus_name.replace(" ", "_")}_prediction_map.html'
    #     os.makedirs(os.path.dirname(output_file), exist_ok=True)

    #     # --- Create prediction grid over India ---
    #     lat_min, lat_max = 6.5, 37.5
    #     lon_min, lon_max = 68.0, 97.0
    #     lats = np.arange(lat_min, lat_max + 0.5, 0.5)
    #     lons = np.arange(lon_min, lon_max + 0.5, 0.5)
    #     grid_points = [[lat, lon] for lat in lats for lon in lons]
    #     print(f"Created {len(grid_points)} grid points for prediction")

    #     # --- Extract features using Earth Engine ---
    #     try:
    #         import ee
    #         from .features_extractor import Feature_Extractor

    #         ee.Authenticate()
    #         ee.Initialize(project='ee-mtpictd')

    #         grid_df = pd.DataFrame(grid_points, columns=['latitude', 'longitude'])
    #         fe = Feature_Extractor(ee)
    #         grid_with_features = fe.add_features(grid_df)
    #         print(grid_with_features.columns)
            
    #         X_grid = grid_with_features[feature_cols].values.astype(float)
    #         probabilities = clf.predict_proba(X_grid)[:, 1]
    #         grid_with_features['probability'] = probabilities

    #         if 'elevation' not in grid_with_features.columns:
    #             grid_with_features['elevation'] = 0
    #             print("Warning: Elevation feature not found. Defaulting to 0.")
    #     except Exception as e:
    #         print(f"Error extracting features or predicting: {e}")
    #         return None

    #     print(f"Generated predictions for {len(grid_with_features)} grid points")

    #     # --- Create Folium Map ---
    #     m = folium.Map(location=[22.0, 78.0], zoom_start=5, tiles='CartoDB positron')

    #     # --- 1️⃣ Add Probability Heatmap ---
    #     heat_data = [
    #         [row['latitude'], row['longitude'], row['probability']]
    #         for _, row in grid_with_features.iterrows()
    #     ]
    #     plugins.HeatMap(heat_data, name='Probability Heatmap', radius=20, blur=15, max_zoom=6).add_to(m)

    #     # --- 2️⃣ Add Elevation Heatmap (for visual comparison) ---
    #     elev_data = [
    #         [row['latitude'], row['longitude'], row['elevation']]
    #         for _, row in grid_with_features.iterrows() if row['elevation'] > 0
    #     ]
    #     plugins.HeatMap(elev_data, name='Elevation Heatmap', radius=20, blur=15, gradient={0.2: 'blue', 0.5: 'orange', 0.8: 'red'}).add_to(m)

    #     # --- 3️⃣ Add Probability + Elevation Points ---
    #     prob_points = folium.FeatureGroup(name='Prediction Points', overlay=True)
    #     for _, row in grid_with_features.iterrows():
    #         lat, lon, prob, elev = row['latitude'], row['longitude'], row['probability'], row['elevation']

    #         # Color based on probability
    #         if prob < 0.3:
    #             color = 'red'
    #         elif prob < 0.6:
    #             color = 'orange'
    #         else:
    #             color = 'green'

    #         # Marker size based on elevation
    #         # Marker size based on elevation
    #         if pd.isna(elev) or elev <= 0:
    #             radius = 3  # Default for missing or invalid elevation
    #         else:
    #             radius = max(3, min(12, int(elev / 200) + 3))

    #         popup_text = f"<b>Prob:</b> {prob:.2f}<br><b>Elevation:</b> {0 if pd.isna(elev) else elev:.0f} m"


    #         folium.CircleMarker(
    #             location=[lat, lon],
    #             radius=radius,
    #             color=color,
    #             fill=True,
    #             fillColor=color,
    #             fillOpacity=0.6,
    #             popup=popup_text
    #         ).add_to(prob_points)
    #     prob_points.add_to(m)

    #     # --- 4️⃣ Add Actual Presence and Absence Points ---
    #     presence_group = folium.FeatureGroup(name=f'Presence (n={len(presence_df)})', overlay=True)
    #     for _, row in presence_df.iterrows():
    #         folium.CircleMarker(
    #             location=[row['latitude'], row['longitude']],
    #             radius=3,
    #             color='blue',
    #             fill=True,
    #             fillColor='blue',
    #             fillOpacity=0.8
    #         ).add_to(presence_group)
    #     presence_group.add_to(m)

    #     absence_group = folium.FeatureGroup(name=f'Absence (n={len(absence_df)})', overlay=True)
    #     for _, row in absence_df.iterrows():
    #         folium.CircleMarker(
    #             location=[row['latitude'], row['longitude']],
    #             radius=2,
    #             color='gray',
    #             fill=True,
    #             fillColor='gray',
    #             fillOpacity=0.5
    #         ).add_to(absence_group)
    #     absence_group.add_to(m)

    #     # --- 5️⃣ Add Legend + Layer Control ---
    #     legend_html = """
    #     <div style="position: fixed; bottom: 50px; left: 50px; width: 230px; 
    #                 background-color: white; border:2px solid grey; z-index:9999; 
    #                 font-size:14px; padding: 10px">
    #         <b>Legend</b><br>
    #         <span style="color:green;">●</span> High Probability (>0.6)<br>
    #         <span style="color:orange;">●</span> Medium (0.3–0.6)<br>
    #         <span style="color:red;">●</span> Low (<0.3)<br>
    #         <b>Circle Size:</b> Elevation (larger = higher)
    #     </div>
    #     """
    #     m.get_root().html.add_child(folium.Element(legend_html))
    #     folium.LayerControl(collapsed=False).add_to(m)

    #     # --- Save and Return ---
    #     m.save(output_file)
    #     print(f"✅ Prediction map with probability and elevation saved to: {output_file}")

    #     return output_file

        
    def create_prediction_map_for_genus(self, clf, feature_cols, genus_name, presence_df, absence_df):
        """
        Create an interactive Folium map showing:
        - Probability of species presence (color-coded)
        - Elevation (shown via marker size, popup, and heatmap)
        Only grid points with valid (non-NaN) elevation values are used for prediction.
        """

        import os
        import folium
        import numpy as np
        import pandas as pd
        from folium import plugins

        print(f"Creating prediction map for {genus_name}...")

        output_file = f'outputs/testing_SDM_out/{genus_name.replace(" ", "_")}_prediction_map.html'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # --- 1️⃣ Create prediction grid over India ---
        lat_min, lat_max = 6.5, 37.5
        lon_min, lon_max = 68.0, 97.0
        lats = np.arange(lat_min, lat_max + 0.5, 0.5)
        lons = np.arange(lon_min, lon_max + 0.5, 0.5)
        grid_points = [[lat, lon] for lat in lats for lon in lons]
        print(f"Created {len(grid_points)} grid points for prediction")

        # --- 2️⃣ Extract features using Earth Engine ---
        try:
            import ee
            from .features_extractor import Feature_Extractor

            ee.Authenticate()
            ee.Initialize(project='ee-mtpictd')

            grid_df = pd.DataFrame(grid_points, columns=['latitude', 'longitude'])
            fe = Feature_Extractor(ee)
            grid_with_features = fe.add_features(grid_df)
            print(grid_with_features[['elevation']].describe())

            # --- Filter only rows with valid elevation ---
            if 'elevation' not in grid_with_features.columns:
                print("⚠️ Elevation column missing. Cannot continue prediction.")
                return None

            before_filter = len(grid_with_features)
            grid_with_features = grid_with_features.dropna(subset=['elevation'])
            after_filter = len(grid_with_features)
            print(f"⚠️ Skipped {before_filter - after_filter} grid points due to missing elevation.")

            if after_filter == 0:
                print("❌ No valid grid points with elevation found. Exiting.")
                return None

            # --- 3️⃣ Ensure numeric features only ---
            for col in feature_cols:
                if col in grid_with_features.columns:
                    grid_with_features[col] = pd.to_numeric(grid_with_features[col], errors='coerce')

            X_grid = grid_with_features[feature_cols].dropna().values.astype(float)

            # --- Keep only rows that remain valid after NaN removal ---
            grid_with_features = grid_with_features.iloc[:len(X_grid)]

            # --- 4️⃣ Predict probabilities ---
            probabilities = clf.predict_proba(X_grid)[:, 1]
            grid_with_features['probability'] = probabilities

            print(f"✅ Generated predictions for {len(grid_with_features)} valid grid points with elevation.")

        except Exception as e:
            print(f"Error extracting features or predicting: {e}")
            return None

        # --- 5️⃣ Create Folium Map ---
        m = folium.Map(location=[22.0, 78.0], zoom_start=5, tiles='CartoDB positron')

        # --- Probability Heatmap ---
        heat_data = [
            [row['latitude'], row['longitude'], row['probability']]
            for _, row in grid_with_features.iterrows()
        ]
        plugins.HeatMap(heat_data, name='Probability Heatmap', radius=20, blur=15, max_zoom=6).add_to(m)

        # --- Elevation Heatmap ---
        elev_data = [
            [row['latitude'], row['longitude'], row['elevation']]
            for _, row in grid_with_features.iterrows() if not pd.isna(row['elevation']) and row['elevation'] > 0
        ]
        plugins.HeatMap(
            elev_data,
            name='Elevation Heatmap',
            radius=20,
            blur=15,
            gradient={0.2: 'blue', 0.5: 'orange', 0.8: 'red'}
        ).add_to(m)

        # --- Add Prediction Points ---
        prob_points = folium.FeatureGroup(name='Prediction Points', overlay=True)
        for _, row in grid_with_features.iterrows():
            lat, lon, prob, elev = row['latitude'], row['longitude'], row['probability'], row['elevation']

            # Color based on probability
            if prob < 0.3:
                color = 'red'
            elif prob < 0.6:
                color = 'orange'
            else:
                color = 'green'

            # Marker size based on elevation
            if pd.isna(elev) or elev <= 0:
                radius = 3
            else:
                radius = max(3, min(12, int(elev / 200) + 3))

            popup_text = f"<b>Prob:</b> {prob:.2f}<br><b>Elevation (rel):</b> {elev:.2f}"


            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                popup=popup_text
            ).add_to(prob_points)
        prob_points.add_to(m)

        # --- Add Presence / Absence Points ---
        presence_group = folium.FeatureGroup(name=f'Presence (n={len(presence_df)})', overlay=True)
        for _, row in presence_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='blue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.8
            ).add_to(presence_group)
        presence_group.add_to(m)

        absence_group = folium.FeatureGroup(name=f'Absence (n={len(absence_df)})', overlay=True)
        for _, row in absence_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=2,
                color='gray',
                fill=True,
                fillColor='gray',
                fillOpacity=0.5
            ).add_to(absence_group)
        absence_group.add_to(m)

        # --- Legend and Layer Control ---
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; width: 230px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
            <b>Legend</b><br>
            <span style="color:green;">●</span> High Probability (>0.6)<br>
            <span style="color:orange;">●</span> Medium (0.3–0.6)<br>
            <span style="color:red;">●</span> Low (<0.3)<br>
            <b>Circle Size:</b> Elevation (larger = higher)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl(collapsed=False).add_to(m)

        # --- Save and Return ---
        m.save(output_file)
        print(f"✅ Prediction map saved to: {output_file}")

        return output_file

        
    
    def train_on_max_ecoregion_for_genus(self, genus_name, model_type, loss_func, is_bias, is_endemic, presence_path=None):
        """
        For a given genus, find the ecoregion with the most presence points, filter both presence and 
        pseudo-absence points to that ecoregion, and run the modeling pipeline on the filtered data.
        
        Args:
            genus_name: Name of the genus
            model_type: Type of model ('Random Forest', 'Logistic Regression', etc.)
            loss_func: Loss function to use ('Dice', 'Tversky', 'Focal', etc.)
            is_bias: Whether to apply bias correction (True/False)
            is_endemic: Whether the genus is endemic (True/False) - affects alpha/beta and optimization metric
            presence_path: Optional path to presence points CSV (if None, loads from all_presence_point.csv)
        """
        import os
        from shapely.wkt import loads as load_wkt
        from shapely.geometry import Point
        from .features_extractor import Feature_Extractor
        
        # Create output folder automatically based on genus name
        output_folder = f"outputs/ecoregion_modeling_{genus_name.replace(' ', '_')}"
        os.makedirs(output_folder, exist_ok=True)
        
        # Set optimization metric and alpha/beta based on endemic status
        if is_endemic:
            optimize_for = 'tnr'  # Endemic species: optimize for True Negative Rate
            alpha = 0.3
            beta = 0.7
            print(f"Endemic genus detected: Optimizing for TNR with alpha={alpha}, beta={beta}")
        else:
            optimize_for = 'tpr'  # Non-endemic species: optimize for True Positive Rate
            alpha = 0.7
            beta = 0.3
            print(f"Non-endemic genus detected: Optimizing for TPR with alpha={alpha}, beta={beta}")
        
        print(f"\n{'='*60}")
        print(f"ECOREGION-LEVEL MODELING FOR GENUS: {genus_name}")
        print(f"{'='*60}")
        print(f"  Model Type: {model_type}")
        print(f"  Loss Function: {loss_func}")
        print(f"  Bias Correction: {is_bias}")
        print(f"  Endemic Status: {is_endemic}")
        print(f"  Optimization Metric: {optimize_for}")
        print(f"  Output Folder: {output_folder}")
        
        # Load all presence points for the genus
        if presence_path is None:
            presence_csv_path = "data/testing_SDM/all_presence_point.csv"
        else:
            presence_csv_path = presence_path
            
        all_genus_df = pd.read_csv(presence_csv_path)
        
        # Check if we're loading from all_presence_point.csv or from a genus-specific file
        if 'genus' in all_genus_df.columns:
            # Loading from all_presence_point.csv
            required_columns = ['genus', 'order', 'decimalLatitude', 'decimalLongitude']
            missing_columns = [col for col in required_columns if col not in all_genus_df.columns]
            if missing_columns:
                raise ValueError(f"CSV must contain columns: {missing_columns}")
            presence_df = all_genus_df[all_genus_df['genus'] == genus_name].copy()
            if len(presence_df) == 0:
                raise ValueError(f"No presence points found for genus: {genus_name}")
        else:
            # Loading from genus-specific file (already filtered)
            presence_df = all_genus_df.copy()
            # Ensure coordinate columns exist
            if 'longitude' in presence_df.columns and 'latitude' in presence_df.columns:
                presence_df = presence_df.rename(columns={'longitude': 'decimalLongitude', 'latitude': 'decimalLatitude'})
        
        # Load all ecoregion polygons
        eco_dir = "data/eco_regions_polygon"
        eco_polygons = {}
        for fname in os.listdir(eco_dir):
            if fname.endswith('.wkt'):
                eco_name = fname.replace('.wkt', '')
                with open(os.path.join(eco_dir, fname), 'r') as f:
                    eco_polygons[eco_name] = load_wkt(f.read().strip())
        
        # Find which ecoregion has the most presence points
        presence_points = [Point(lon, lat) for lon, lat in zip(presence_df['decimalLongitude'], presence_df['decimalLatitude'])]
        eco_counts = {eco: 0 for eco in eco_polygons}
        eco_assignments = []
        for pt in presence_points:
            found = False
            for eco, poly in eco_polygons.items():
                if poly.contains(pt):
                    eco_counts[eco] += 1
                    eco_assignments.append(eco)
                    found = True
                    break
            if not found:
                eco_assignments.append(None)
        
        # Get the ecoregion with the maximum count
        max_eco = max(eco_counts, key=eco_counts.get)
        print(f"   Ecoregion with max presence points: {max_eco} ({eco_counts[max_eco]} points)")
        
        # Filter presence points to only those in the max ecoregion
        presence_df = presence_df[[eco == max_eco for eco in eco_assignments]].copy()
        presence_df = presence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
        presence_df = presence_df[['longitude', 'latitude']]
        
        # Prepare pseudo-absence points
        # Load all genus data for pseudo-absences
        all_presence_csv = "data/testing_SDM/all_presence_point.csv"
        all_genus_data = pd.read_csv(all_presence_csv)
        
        presence_coords = set(zip(presence_df['longitude'], presence_df['latitude']))
        pseudo_absence_df = all_genus_data[(all_genus_data['genus'] != genus_name)].copy()
        pseudo_absence_df = pseudo_absence_df.rename(columns={
            'decimalLatitude': 'latitude',
            'decimalLongitude': 'longitude'
        })
        pseudo_absence_df = pseudo_absence_df[['longitude', 'latitude']]
        
        # Only keep pseudo-absence points in the max ecoregion and not in presence
        initial_absence_count = len(pseudo_absence_df)
        pseudo_absence_points = [Point(lon, lat) for lon, lat in zip(pseudo_absence_df['longitude'], pseudo_absence_df['latitude'])]
        pseudo_absence_df = pseudo_absence_df[[eco_polygons[max_eco].contains(pt) and (pt.x, pt.y) not in presence_coords for pt in pseudo_absence_points]]
        filtered_absence_count = len(pseudo_absence_df)
        removed_duplicates = initial_absence_count - filtered_absence_count

        # Sample the same number of pseudo-absences as presences (or all if fewer available)
        if len(pseudo_absence_df) > len(presence_df):
            pseudo_absence_df = pseudo_absence_df.sample(n=len(presence_df), random_state=42)

        print(f"   Filtered to {len(presence_df)} presence and {len(pseudo_absence_df)} pseudo-absence points in {max_eco}")
        print(f"🔍 ABSENCE POINT FILTERING: Removed {removed_duplicates} duplicate points that matched presence coordinates")
        print(f"   Initial absence points: {initial_absence_count}")
        print(f"   After filtering: {filtered_absence_count}")
        print(f"   Duplicates removed: {removed_duplicates}")

        # Create interactive map in output folder
        try:
            map_filename = f'{genus_name.replace(" ", "_")}_ecoregion_presence_absence_map.html'
            map_file_path = os.path.join(output_folder, map_filename)
            map_file = visualize_presence_absence_points(presence_df, pseudo_absence_df, genus_name, output_file=map_file_path)
            if map_file:
                print(f"✓ Interactive map created: {map_file}")
        except Exception as e:
            print(f"Warning: Could not create map visualization: {e}")
        
        # Feature extraction
        try:
            import ee
            ee.Authenticate()
            ee.Initialize(project='ee-mtpictd')
        except:
            print("   Warning: Earth Engine not initialized. Please initialize EE first.")
            return
        
        feature_extractor = Feature_Extractor(ee)
        presence_with_features = feature_extractor.add_features(presence_df)
        absence_with_features = feature_extractor.add_features(pseudo_absence_df)
        
        # Prepare data for modeling
        feature_cols = [col for col in presence_with_features.columns if col not in ['longitude', 'latitude']]

        # Apply bias correction with full ecoregion-based weighting
        X, y, reliability_weights, bias_weights, combined_weights = self.apply_bias_correction(
            presence_with_features, absence_with_features, feature_cols
        )

        # Choose sample weights based on bias correction parameter
        if is_bias:
            sample_weights = bias_weights
            print(f"Using bias correction weights")
        else:
            sample_weights = reliability_weights
            print(f"Using reliability weights only (no bias correction)")

        # Data is already cleaned of NaN values in apply_bias_correction
        print(f"   After bias correction and NaN filtering: {X.shape}")
        print(f"   Prepared {len(X)} total samples for modeling")

        # Configure model and loss function based on input parameters
        print(f"\n   Training with: {model_type} + {loss_func}")
        
        results_table = []
        
        try:
            # Train the specified model
            if model_type == 'Random Forest':
                clf, X_test, y_test, y_pred, y_proba = self.RandomForest(X, y, sample_weights=sample_weights)
            elif model_type == 'Logistic Regression':
                clf, X_test, y_test, y_pred, y_proba = self.logistic_regression_L2(X, y)
            elif model_type == 'Weighted Logistic Regression':
                clf, X_test, y_test, y_pred, y_proba = self.train_and_evaluate_model_logistic_weighted(X, y, sample_weights=sample_weights)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Apply the specified loss function for threshold optimization
            if loss_func == 'Dice':
                from .custom_losses import DiceScorer
                scorer = DiceScorer(smooth=1.0)
            elif loss_func == 'Focal':
                from .custom_losses import FocalScorer
                scorer = FocalScorer(alpha=0.25, gamma=2.0)
            elif loss_func == 'Tversky':
                from .custom_losses import TverskyScorer
                scorer = TverskyScorer(alpha=alpha, beta=beta)  # Use endemic-based alpha/beta
            else:
                scorer = None  # Original loss (no threshold optimization)
            
            # Threshold optimization based on loss function and optimization metric
            if scorer is not None:
                thresholds = np.linspace(0.1, 0.9, 20)
                best_score = -np.inf
                optimal_threshold = 0.5
                
                print(f"   Optimizing threshold for {optimize_for} with {loss_func} loss...")
                for threshold in thresholds:
                    score = scorer(y_test, y_proba, threshold)
                    if score > best_score:
                        best_score = score
                        optimal_threshold = threshold
                
                # Calculate metrics with optimal threshold
                y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                
                accuracy = accuracy_score(y_test, y_pred_optimal)
                precision = precision_score(y_test, y_pred_optimal, zero_division=0)
                recall = recall_score(y_test, y_pred_optimal, zero_division=0)
                f1 = f1_score(y_test, y_pred_optimal, zero_division=0)
                cm = confusion_matrix(y_test, y_pred_optimal)
                
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    tpr = tnr = None
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'confusion_matrix': cm
                }
                
                print(f"      Optimal threshold: {optimal_threshold:.4f}")
                print(f"      Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                print(f"      TPR: {tpr:.4f}, TNR: {tnr:.4f}")
            else:
                # No threshold optimization (original loss)
                metrics = self.evaluate_model(clf, X_test, y_test, dataset_name=f"{model_type}_{loss_func}")
                optimal_threshold = 0.5
                
                # Calculate TPR and TNR
                cm = metrics['confusion_matrix']
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    tpr = tnr = None
                
                print(metrics['classification_report'])
            
            # Store results
            combination_name = f"{model_type}_{loss_func}"
            row = {
                'model_loss': combination_name,
                'ecoregion': max_eco,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'tpr': tpr if tpr is not None else 0.0,
                'tnr': tnr if tnr is not None else 0.0,
                'optimal_threshold': optimal_threshold,
                'optimization_metric': optimize_for
            }
            results_table.append(row)
            
        except Exception as e:
            print(f"      Error in {model_type}_{loss_func}: {str(e)}")
            results_table.append({
                'model_loss': f"{model_type}_{loss_func}",
                'ecoregion': max_eco,
                'error': str(e)
            })

        # Save results as CSV in output folder
        results_df = pd.DataFrame(results_table)
        output_csv = os.path.join(output_folder, f"{genus_name}_ecoregion_results.csv")
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults table saved to: {output_csv}")
        
        # Print summary table
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE FOR {genus_name} in {max_eco}")
        print(f"{'='*80}")
        print(f"{'Model/Loss':<35} {'Accuracy':<10} {'F1-Score':<10} {'TPR':<8} {'TNR':<8} {'Threshold':<10} {'Opt For':<8}")
        print(f"{'-'*80}")
        for _, row in results_df.iterrows():
            if 'error' not in row:
                f1_score_val = row.get('f1', 'N/A')
                f1_display = f"{f1_score_val:.4f}" if isinstance(f1_score_val, (int, float)) else "N/A"
                print(f"{row['model_loss']:<35} {row['accuracy']:<10.4f} {f1_display:<10} {row['tpr']:<8.4f} {row['tnr']:<8.4f} {row['optimal_threshold']:<10.4f} {row['optimization_metric']:<8}")
        
        print(f"\n{'='*60}")
        print(f"Presence points: {len(presence_df)}")
        print(f"Absence points: {len(pseudo_absence_df)}")
        print(f"{'='*60}")

        # Save extracted features in output folder
        extracted_features_dir = os.path.join(output_folder, 'extracted_features')
        os.makedirs(extracted_features_dir, exist_ok=True)
        genus_safe = genus_name.replace(' ', '_').lower()
        presence_path_save = os.path.join(extracted_features_dir, f"{genus_safe}_ecoregion_presence_features.csv")
        absence_path_save = os.path.join(extracted_features_dir, f"{genus_safe}_ecoregion_absence_features.csv")
        presence_with_features.to_csv(presence_path_save, index=False)
        absence_with_features.to_csv(absence_path_save, index=False)
        print(f"Features saved to: {presence_path_save} and {absence_path_save}")
        
        # Create prediction map in output folder
        # if len(results_table) > 0 and 'error' not in results_table[0]:
        #     print(f"\nCreating prediction map for {model_type}_{loss_func} in {max_eco}")
        #     try:
        #         # Retrain on full dataset
        #         if model_type == 'Random Forest':
        #             from sklearn.ensemble import RandomForestClassifier
        #             clf = RandomForestClassifier(n_estimators=100, random_state=42)
        #             clf.fit(X, y, sample_weight=sample_weights)
        #         elif model_type == 'Logistic Regression':
        #             from sklearn.linear_model import LogisticRegression
        #             clf = LogisticRegression(random_state=42, max_iter=1000)
        #             clf.fit(X, y, sample_weight=sample_weights)
        #         else:
        #             clf = LogisticRegression(random_state=42, max_iter=1000)
        #             clf.fit(X, y, sample_weight=sample_weights)
                
        #         # Create prediction map for the ecoregion
        #         prediction_map_file = self.create_prediction_map_for_ecoregion(
        #             clf, feature_cols, genus_name, max_eco,
        #             presence_with_features, absence_with_features,
        #             output_folder=output_folder
        #         )
        #         if prediction_map_file:
        #             print(f"✓ Prediction map created: {prediction_map_file}")
        #     except Exception as e:
        #         print(f"Warning: Could not create prediction map: {e}")
        
        print(f"\nEcoregion modeling complete for genus: {genus_name}")
        
        return {
            'genus_name': genus_name,
            'ecoregion': max_eco,
            'presence_count': len(presence_df),
            'absence_count': len(pseudo_absence_df),
            'results': results_df
        }

    def perform_feature_importance_for_all_genera(self, genus_list):
        """
        For each genus, find the best model (India-level or ecoregion-level), retrain, and perform feature importance analysis.
        Prints and visualizes feature importance for each genus.
        Now also reports on missing data after feature extraction.
        Uses saved features from comprehensive_genus_modeling or train_on_max_ecoregion_for_genus if available.
        Also generates SHAP summary plots for model interpretability.
        """
        # Import SHAP at the beginning
        try:
            import shap
        except ImportError:
            print("SHAP library not found. Installing SHAP...")
            import subprocess
            subprocess.check_call(["pip", "install", "shap"])
            import shap

        import os

        for genus_name in genus_list:
            print(f"\n{'='*80}")
            print(f"Feature Importance Analysis for {genus_name}")
            print(f"{'='*80}")

            # 1. Check for both India-level and ecoregion-level results
            comprehensive_csv_path = f"outputs/{genus_name}_comprehensive_results.csv"
            ecoregion_csv_path = f"outputs/{genus_name}_ecoregion_results.csv"

            if os.path.exists(comprehensive_csv_path):
                csv_path = comprehensive_csv_path
                analysis_type = "India-level"
                print(f"  Using {analysis_type} results from: {csv_path}")
            elif os.path.exists(ecoregion_csv_path):
                csv_path = ecoregion_csv_path
                analysis_type = "ecoregion-level"
                print(f"  Using {analysis_type} results from: {csv_path}")
            else:
                print(f"  Skipping {genus_name}: No comprehensive or ecoregion results CSV found.")
                continue

            df = pd.read_csv(csv_path)
            # 2. Find best model/loss (highest accuracy)
            best_row = df.loc[df['accuracy'].idxmax()]
            best_combo = best_row['model_loss']
            print(f"  Best model/loss: {best_combo} (Accuracy: {best_row['accuracy']:.4f})")
            # 3. Parse model/loss
            if 'Random_Forest' in best_combo:
                model_type = 'rf'
            elif 'Logistic_Regression' in best_combo:
                model_type = 'logistic'
            elif 'Weighted_Logistic_Regression' in best_combo:
                model_type = 'logistic_weighted'
            else:
                print(f"  Unknown model type for {genus_name}, skipping.")
                continue
            if 'Tversky' in best_combo:
                loss_type = 'tversky'
            elif 'Focal' in best_combo:
                loss_type = 'focal'
            elif 'Dice' in best_combo:
                loss_type = 'dice'
            else:
                loss_type = 'original'
            # 4. Load or extract features
            features_output_dir = "outputs/extracted_features"
            genus_safe = genus_name.replace(' ', '_').lower()

            # Check for both types of feature files
            presence_features_file = os.path.join(features_output_dir, f"{genus_safe}_presence_features.csv")
            absence_features_file = os.path.join(features_output_dir, f"{genus_safe}_absence_features.csv")
            ecoregion_presence_features_file = os.path.join(features_output_dir, f"{genus_safe}_ecoregion_presence_features.csv")
            ecoregion_absence_features_file = os.path.join(features_output_dir, f"{genus_safe}_ecoregion_absence_features.csv")

            if analysis_type == "India-level" and os.path.exists(presence_features_file) and os.path.exists(absence_features_file):
                print(f"  Loading saved India-level features from files...")
                presence_with_features = pd.read_csv(presence_features_file)
                absence_with_features = pd.read_csv(absence_features_file)
                print(f"  Loaded {len(presence_with_features)} presence and {len(absence_with_features)} absence feature records")
            elif analysis_type == "ecoregion-level" and os.path.exists(ecoregion_presence_features_file) and os.path.exists(ecoregion_absence_features_file):
                print(f"  Loading saved ecoregion-level features from files...")
                presence_with_features = pd.read_csv(ecoregion_presence_features_file)
                absence_with_features = pd.read_csv(ecoregion_absence_features_file)
                print(f"  Loaded {len(presence_with_features)} presence and {len(absence_with_features)} absence feature records")
            else:
                print(f"  Saved features not found. Extracting features (this may take a while)...")
                # Prepare data based on analysis type
                all_points = pd.read_csv("data/testing_SDM/all_presence_point.csv")

                if analysis_type == "India-level":
                    # India-level data preparation (same as in comprehensive_genus_modeling)
                    presence_df = all_points[all_points['genus'] == genus_name].copy()
                    presence_df = presence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
                    presence_df = presence_df[['longitude', 'latitude']]
                    presence_coords = set(zip(presence_df['longitude'], presence_df['latitude']))
                    absence_df = all_points[all_points['genus'] != genus_name].copy()
                    absence_df = absence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
                    absence_df = absence_df[['longitude', 'latitude']]
                    absence_df = absence_df[~absence_df.apply(lambda r: (r['longitude'], r['latitude']) in presence_coords, axis=1)]
                    if len(absence_df) > len(presence_df):
                        absence_df = absence_df.sample(n=len(presence_df), random_state=42)
                else:
                    # Ecoregion-level data preparation (same as in train_on_max_ecoregion_for_genus)
                    from shapely.wkt import loads as load_wkt
                    from shapely.geometry import Point
                    import os

                    presence_df = all_points[all_points['genus'] == genus_name].copy()
                    if len(presence_df) == 0:
                        print(f"  No presence points found for genus: {genus_name}, skipping.")
                        continue

                    # Load all ecoregion polygons
                    eco_dir = "data/eco_regions_polygon"
                    eco_polygons = {}
                    for fname in os.listdir(eco_dir):
                        if fname.endswith('.wkt'):
                            eco_name = fname.replace('.wkt', '')
                            with open(os.path.join(eco_dir, fname), 'r') as f:
                                eco_polygons[eco_name] = load_wkt(f.read().strip())

                    # Find which ecoregion has the most presence points
                    presence_points = [Point(lon, lat) for lon, lat in zip(presence_df['decimalLongitude'], presence_df['decimalLatitude'])]
                    eco_counts = {eco: 0 for eco in eco_polygons}
                    eco_assignments = []
                    for pt in presence_points:
                        found = False
                        for eco, poly in eco_polygons.items():
                            if poly.contains(pt):
                                eco_counts[eco] += 1
                                eco_assignments.append(eco)
                                found = True
                                break
                        if not found:
                            eco_assignments.append(None)

                    # Get the ecoregion with the maximum count
                    max_eco = max(eco_counts, key=eco_counts.get)
                    print(f"  Using ecoregion: {max_eco} ({eco_counts[max_eco]} points)")

                    # Filter presence points to only those in the max ecoregion
                    presence_df = presence_df[[eco == max_eco for eco in eco_assignments]].copy()
                    presence_df = presence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
                    presence_df = presence_df[['longitude', 'latitude']]

                    # Prepare pseudo-absence points: use points from other genera, not in presence, and in the same ecoregion
                    presence_coords = set(zip(presence_df['longitude'], presence_df['latitude']))
                    absence_df = all_points[(all_points['genus'] != genus_name)].copy()
                    absence_df = absence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
                    absence_df = absence_df[['longitude', 'latitude']]

                    # Only keep pseudo-absence points in the max ecoregion and not in presence
                    initial_absence_count = len(absence_df)
                    pseudo_absence_points = [Point(lon, lat) for lon, lat in zip(absence_df['longitude'], absence_df['latitude'])]
                    absence_df = absence_df[[eco_polygons[max_eco].contains(pt) and (pt.x, pt.y) not in presence_coords for pt in pseudo_absence_points]]
                    filtered_absence_count = len(absence_df)
                    removed_duplicates = initial_absence_count - filtered_absence_count

                    # Sample the same number of pseudo-absences as presences (or all if fewer available)
                    if len(absence_df) > len(presence_df):
                        absence_df = absence_df.sample(n=len(presence_df), random_state=42)

                # Feature extraction
                features_extractor_obj = features_extractor.Feature_Extractor(ee)
                presence_with_features = features_extractor_obj.add_features(presence_df)
                absence_with_features = features_extractor_obj.add_features(absence_df)
                print(f"  Extracted features for {len(presence_with_features)} presence and {len(absence_with_features)} absence points")

            feature_cols = [col for col in presence_with_features.columns if col not in ['longitude', 'latitude']]
            # Concatenate and check for missing data
            combined_df = pd.concat([presence_with_features, absence_with_features], ignore_index=True)
            n_total = len(combined_df)
            n_missing = combined_df[feature_cols].isnull().any(axis=1).sum()
            percent_missing = 100 * n_missing / n_total if n_total > 0 else 0
            print(f"  Total points: {n_total}")
            print(f"  Points with missing features: {n_missing} ({percent_missing:.1f}%)")
            if percent_missing > 20:
                print(f"  WARNING: More than 20% of points have missing data. Feature importance results may be unreliable.")
            # Filter out points with missing data
            valid_mask = ~combined_df[feature_cols].isnull().any(axis=1)
            filtered_df = combined_df.loc[valid_mask].reset_index(drop=True)
            if len(filtered_df) == 0:
                print(f"  No valid points left after filtering missing data. Skipping {genus_name}.")
                continue
            X = filtered_df[feature_cols].values
            y = np.hstack([
                np.ones(len(presence_with_features)),
                np.zeros(len(absence_with_features))
            ])[valid_mask.values]
            sample_weights = np.ones(len(y))
            # 5. Retrain best model
            modelss = Models()
            if loss_type == 'original':
                if model_type == 'rf':
                    clf, _, _, _, _ = modelss.RandomForest(X, y, sample_weights=sample_weights)
                elif model_type == 'logistic':
                    clf, _, _, _, _ = modelss.logistic_regression_L2(X, y)
                else:
                    clf, _, _, _, _ = modelss.train_and_evaluate_model_logistic_weighted(X, y, sample_weights=sample_weights)
            elif loss_type == 'tversky':
                clf = modelss.train_with_tversky_scoring(X, y, sample_weights=sample_weights, model_type=model_type)
            elif loss_type == 'focal':
                clf = modelss.train_with_focal_scoring(X, y, sample_weights=sample_weights, model_type=model_type)
            elif loss_type == 'dice':
                clf = modelss.train_with_dice_scoring(X, y, sample_weights=sample_weights, model_type=model_type)
            else:
                print(f"  Unknown loss type for {genus_name}, skipping.")
                continue

            # 6. SHAP Analysis
            print(f"  Generating SHAP summary plots...")

            # Create genus-specific output directory with analysis type
            genus_safe_name = genus_name.replace(' ', '_').lower()
            genus_output_dir = f"outputs/testing_SDM_out/{genus_safe_name}_{analysis_type.replace('-', '_').lower()}"
            os.makedirs(genus_output_dir, exist_ok=True)

            # Initialize SHAP importance as empty dict in case SHAP fails
            shap_importance = {}
            shap_importance_sorted = []

            try:
                # Create SHAP explainer based on model type
                if model_type == 'rf':
                    # For Random Forest, use TreeExplainer
                    explainer = shap.TreeExplainer(clf)
                else:
                    # For logistic regression and other models, use KernelExplainer as fallback
                    # This is more robust than LinearExplainer
                    explainer = shap.KernelExplainer(clf.predict_proba, X[:100])  # Use small sample for background

                # Calculate SHAP values (use a sample if dataset is too large)
                if len(X) > 1000:
                    print(f"    Using sample of 1000 points for SHAP analysis (dataset has {len(X)} points)")
                    sample_indices = np.random.choice(len(X), 1000, replace=False)
                    X_sample = X[sample_indices]
                else:
                    X_sample = X

                # Get SHAP values with proper error handling
                try:
                    shap_values = explainer.shap_values(X_sample)

                    # Handle different SHAP values formats
                    if isinstance(shap_values, list):
                        # For binary classification, shap_values is a list [negative_class, positive_class]
                        if len(shap_values) == 2:
                            shap_values = shap_values[1]  # Use positive class SHAP values
                        else:
                            shap_values = shap_values[0]  # Use first class if more than 2
                    elif isinstance(shap_values, np.ndarray):
                        # If it's already a numpy array, use as is
                        if len(shap_values.shape) == 3:
                            # If 3D array, take positive class
                            shap_values = shap_values[:, :, 1]
                        elif len(shap_values.shape) == 2:
                            # If 2D array, use as is
                            shap_values = shap_values
                        else:
                            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
                    else:
                        raise ValueError(f"Unexpected SHAP values type: {type(shap_values)}")

                    # Ensure shap_values is 2D
                    if len(shap_values.shape) == 1:
                        shap_values = shap_values.reshape(1, -1)

                    print(f"    SHAP values shape: {shap_values.shape}")

                except Exception as shap_error:
                    print(f"    Error calculating SHAP values: {str(shap_error)}")
                    raise shap_error

                # Create SHAP summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
                plt.title(f'SHAP Summary Plot - {genus_name} ({analysis_type})', fontsize=16, fontweight='bold')

                # Save SHAP summary plot in genus folder
                shap_summary_path = os.path.join(genus_output_dir, "shap_summary.png")
                plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    SHAP summary plot saved to: {shap_summary_path}")

                # Create SHAP bar plot (feature importance)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type="bar", show=False)
                plt.title(f'SHAP Feature Importance - {genus_name} ({analysis_type})', fontsize=16, fontweight='bold')

                # Save SHAP bar plot in genus folder
                shap_bar_path = os.path.join(genus_output_dir, "shap_importance.png")
                plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    SHAP feature importance plot saved to: {shap_bar_path}")

                # Calculate and print SHAP-based feature importance
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                shap_importance = dict(zip(feature_cols, mean_abs_shap))
                shap_importance_sorted = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)

                print(f"\nSHAP-based Feature Importance:")
                for feature, importance in shap_importance_sorted:
                    print(f"  {feature}: {importance:.4f}")

                # Save SHAP importance scores to CSV
                shap_importance_df = pd.DataFrame(shap_importance_sorted, columns=['Feature', 'SHAP_Importance'])
                shap_csv_path = os.path.join(genus_output_dir, "shap_importance_scores.csv")
                shap_importance_df.to_csv(shap_csv_path, index=False)
                print(f"    SHAP importance scores saved to: {shap_csv_path}")

            except Exception as e:
                print(f"    Error in SHAP analysis: {str(e)}")
                print(f"    Continuing with feature sensitivity analysis...")
                # Set default values for SHAP importance
                shap_importance = {feature: 0.0 for feature in feature_cols}
                shap_importance_sorted = [(feature, 0.0) for feature in feature_cols]

            # 7. Feature sensitivity analysis (existing code)
            feature_ranges = {
                'annual_mean_temperature': (-185, 293),
                'mean_diurnal_range': (49, 163),
                'isothermality': (19, 69),
                'temperature_seasonality': (431, 11303),
                'max_temperature_warmest_month': (-51, 434),
                'min_temperature_coldest_month': (-369, 246),
                'temperature_annual_range': (74, 425),
                'mean_temperature_wettest_quarter': (-143, 339),
                'mean_temperature_driest_quarter': (-275, 309),
                'mean_temperature_warmest_quarter': (-97, 351),
                'mean_temperature_coldest_quarter': (-300, 275),
                'annual_precipitation': (51, 11401),
                'precipitation_wettest_month': (7, 2949),
                'precipitation_driest_month': (0, 81),
                'precipitation_seasonality': (27, 172),
                'precipitation_wettest_quarter': (18, 8019),
                'precipitation_driest_quarter': (0, 282),
                'precipitation_warmest_quarter': (10, 6090),
                'precipitation_coldest_quarter': (0, 5162),
                'aridity_index': (403, 65535),
                'topsoil_ph': (0, 8.3),
                'subsoil_ph': (0, 8.3),
                'topsoil_texture': (0, 3),
                'subsoil_texture': (0, 13),
                'elevation': (-54, 7548)
            }
            analyzer = FeatureSensitivityAnalyzer(clf, feature_cols, feature_ranges)
            try:
                base_point, base_prob = analyzer.find_high_probability_point(X, threshold=0.9)
                print(f"  Found point with probability: {base_prob:.4f}")
            except ValueError as e:
                print(f"  Warning: {e}. Using point with highest probability instead.")
                probs = clf.predict_proba(X)[:, 1]
                best_idx = np.argmax(probs)
                base_point = X[best_idx]
                base_prob = probs[best_idx]
                print(f"  Using point with probability: {base_prob:.4f}")
            results = analyzer.analyze_all_features(base_point, X)

            # Save feature sensitivity plots in genus folder
            plot_path = os.path.join(genus_output_dir, "feature_sensitivity.png")
            analyzer.plot_feature_sensitivity(results, save_path=plot_path)
            importance_scores = analyzer.get_feature_importance(results)
            print("\nFeature Sensitivity-based Importance Scores:")
            for feature, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {score:.4f}")
            print(f"  Feature sensitivity plots saved to: {plot_path}")

            # Save sensitivity-based importance scores to CSV
            sensitivity_importance_sorted = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            sensitivity_importance_df = pd.DataFrame(sensitivity_importance_sorted, columns=['Feature', 'Sensitivity_Importance'])
            sensitivity_csv_path = os.path.join(genus_output_dir, "sensitivity_importance_scores.csv")
            sensitivity_importance_df.to_csv(sensitivity_csv_path, index=False)
            print(f"    Sensitivity importance scores saved to: {sensitivity_csv_path}")

            # 8. Permutation importance (scikit-learn)
            from sklearn.inspection import permutation_importance
            print(f"\n  Calculating permutation importance...")
            try:
                perm_result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, scoring='accuracy')
                perm_importance = dict(zip(feature_cols, perm_result.importances_mean))
                perm_importance_sorted = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
                print("\nPermutation Importance Scores:")
                for feature, score in perm_importance_sorted:
                    print(f"  {feature}: {score:.4f}")
                # Save to CSV
                perm_df = pd.DataFrame(perm_importance_sorted, columns=['Feature', 'Permutation_Importance'])
                perm_csv_path = os.path.join(genus_output_dir, "permutation_importance_scores.csv")
                perm_df.to_csv(perm_csv_path, index=False)
                print(f"    Permutation importance scores saved to: {perm_csv_path}")
            except Exception as e:
                print(f"    Error in permutation importance: {str(e)}")
                perm_importance = {feature: 0.0 for feature in feature_cols}
                perm_importance_sorted = [(feature, 0.0) for feature in feature_cols]

            # 9. Create combined importance comparison (now with permutation importance)
            combined_importance = {}
            for feature in feature_cols:
                combined_importance[feature] = {
                    'SHAP_Importance': shap_importance.get(feature, 0.0),
                    'Sensitivity_Importance': importance_scores.get(feature, 0.0),
                    'Permutation_Importance': perm_importance.get(feature, 0.0)
                }
            combined_df = pd.DataFrame(combined_importance).T.reset_index()
            combined_df.columns = ['Feature', 'SHAP_Importance', 'Sensitivity_Importance', 'Permutation_Importance']
            # Sort by SHAP importance, but handle case where all SHAP values might be 0
            if combined_df['SHAP_Importance'].sum() > 0:
                combined_df = combined_df.sort_values('SHAP_Importance', ascending=False)
            elif combined_df['Permutation_Importance'].sum() > 0:
                combined_df = combined_df.sort_values('Permutation_Importance', ascending=False)
            else:
                combined_df = combined_df.sort_values('Sensitivity_Importance', ascending=False)
            combined_csv_path = os.path.join(genus_output_dir, "combined_importance_scores.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"    Combined importance scores saved to: {combined_csv_path}")
            print(f"\nAll results for {genus_name} ({analysis_type}) saved to: {genus_output_dir}")


    def resolution_sensitivity_analysis(self, genus_name, presence_path, model_type, loss_func, 
                                   is_bias, is_endemic, base_output_folder,
                                   resolutions=[10, 30, 50, 100, 250, 500, 1000]):
        """
        Performs resolution sensitivity analysis to find the optimal spatial resolution for modeling.
        
        Args:
            genus_name: Name of the genus
            presence_path: Path to presence points with features
            model_type: Model type to use
            loss_func: Loss function to use
            is_bias: Bias correction flag
            is_endemic: Endemic status flag
            base_output_folder: Base folder for outputs
            resolutions: List of resolutions to test (in meters)
        
        Returns:
            DataFrame with results for each resolution
        """
        import os
        import pandas as pd
        import numpy as np
        from scipy.ndimage import uniform_filter
        import ee
        from .features_extractor import Feature_Extractor
        
        print(f"\n{'='*80}")
        print(f"RESOLUTION SENSITIVITY ANALYSIS FOR: {genus_name}")
        print(f"Testing resolutions: {resolutions} meters")
        print(f"{'='*80}\n")
        
        # Load original presence points
        try:
            presence_df = pd.read_csv(presence_path, low_memory=False)
        except FileNotFoundError:
            print(f"Error: Presence file not found at {presence_path}")
            return None
        
        # Ensure coordinates exist
        if 'longitude' in presence_df.columns and 'latitude' in presence_df.columns:
            coord_cols = ['longitude', 'latitude']
        elif 'decimalLongitude' in presence_df.columns and 'decimalLatitude' in presence_df.columns:
            coord_cols = ['decimalLongitude', 'decimalLatitude']
        else:
            print("Error: Could not find coordinate columns")
            return None
        
        coords_only = presence_df[coord_cols].copy()
        
        # Initialize Earth Engine
        ee.Initialize(project='ee-mtpictd')
        fe = Feature_Extractor(ee)
        
        results_list = []
        
        for resolution in resolutions:
            print(f"\n{'='*60}")
            print(f"Testing Resolution: {resolution}m")
            print(f"{'='*60}")
            
            # Create output folder for this resolution
            resolution_output = os.path.join(base_output_folder, f"resolution_{resolution}m")
            os.makedirs(resolution_output, exist_ok=True)
            
            try:
                # Extract features at this resolution
                print(f"Extracting features at {resolution}m resolution...")
                presence_features_resampled = self.extract_features_at_resolution(
                    coords_only, fe, resolution
                )
                
                # Save resampled features
                resampled_path = os.path.join(resolution_output, f"presence_features_{resolution}m.csv")
                presence_features_resampled.to_csv(resampled_path, index=False)
                
                # Train model at this resolution
                print(f"Training model at {resolution}m resolution...")
                results = self.comprehensive_genus_modeling(
                    genus_name=genus_name,
                    model_type=model_type,
                    loss_func=loss_func,
                    is_bias=is_bias,
                    is_endemic=is_endemic,
                    output_folder=resolution_output,
                    presence_path=resampled_path
                )
                
                # Extract key metrics
                if results and 'error' not in results:
                    # Read the results CSV
                    results_csv_path = os.path.join(resolution_output, f"{genus_name}_comprehensive_results.csv")
                    if os.path.exists(results_csv_path):
                        results_df = pd.read_csv(results_csv_path)
                        
                        # Calculate additional metrics
                        feature_cols = [col for col in presence_features_resampled.columns 
                                    if col not in coord_cols]
                        
                        # Calculate spatial autocorrelation at this resolution
                        spatial_autocorr = self.calculate_spatial_autocorrelation(
                            presence_features_resampled, coord_cols, feature_cols
                        )
                        
                        # Calculate feature variance
                        feature_variance = presence_features_resampled[feature_cols].var().mean()
                        
                        # Store results
                        result_row = {
                            'resolution_m': resolution,
                            'accuracy': results_df['accuracy'].values[0] if len(results_df) > 0 else None,
                            'tpr': results_df['tpr'].values[0] if len(results_df) > 0 else None,
                            'tnr': results_df['tnr'].values[0] if len(results_df) > 0 else None,
                            'f1': results_df['f1'].values[0] if len(results_df) > 0 else None,
                            'optimal_threshold': results_df['optimal_threshold'].values[0] if len(results_df) > 0 else None,
                            'spatial_autocorrelation': spatial_autocorr,
                            'feature_variance': feature_variance,
                            'num_features': len(feature_cols),
                            'num_presence_points': len(presence_features_resampled)
                        }
                        results_list.append(result_row)
                        
                        print(f"✓ Resolution {resolution}m completed")
                        print(f"  Accuracy: {result_row['accuracy']:.4f}")
                        print(f"  TPR: {result_row['tpr']:.4f}, TNR: {result_row['tnr']:.4f}")
                        print(f"  F1: {result_row['f1']:.4f}")
                        print(f"  Spatial Autocorr: {result_row['spatial_autocorrelation']:.4f}")
                        
            except Exception as e:
                print(f"✗ Error at resolution {resolution}m: {str(e)}")
                results_list.append({
                    'resolution_m': resolution,
                    'error': str(e)
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results_list)
        
        # Save summary
        summary_path = os.path.join(base_output_folder, f"{genus_name}_resolution_sensitivity_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Create visualization
        self.plot_resolution_sensitivity(summary_df, genus_name, base_output_folder)
        
        # Find optimal resolution
        optimal_resolution = self.find_optimal_resolution(summary_df)
        
        print(f"\n{'='*80}")
        print(f"RESOLUTION SENSITIVITY ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Summary saved to: {summary_path}")
        print(f"\nRecommended Optimal Resolution: {optimal_resolution}m")
        print(f"{'='*80}\n")
        
        return summary_df, optimal_resolution


    def extract_features_at_resolution(self, coords_df, feature_extractor, resolution_m):
        """
        Extract features at a specific spatial resolution by resampling.
        
        Args:
            coords_df: DataFrame with coordinates
            feature_extractor: Feature_Extractor instance
            resolution_m: Target resolution in meters
        
        Returns:
            DataFrame with features extracted at the specified resolution
        """
        import ee
        
        # Convert resolution to degrees (approximate)
        # 1 degree ≈ 111,000 meters at equator
        resolution_deg = resolution_m / 111000.0
        
        # Create a grid at the target resolution
        coords_array = coords_df.values
        lon_col = 0 if 'longitude' in coords_df.columns or 'decimalLongitude' in coords_df.columns else None
        lat_col = 1 if 'latitude' in coords_df.columns or 'decimalLatitude' in coords_df.columns else None
        
        if lon_col is None or lat_col is None:
            raise ValueError("Could not find coordinate columns")
        
        # Round coordinates to the grid resolution
        coords_gridded = coords_array.copy()
        coords_gridded[:, lon_col] = np.round(coords_array[:, lon_col] / resolution_deg) * resolution_deg
        coords_gridded[:, lat_col] = np.round(coords_array[:, lat_col] / resolution_deg) * resolution_deg
        
        # Remove duplicate gridded points
        coords_gridded_df = pd.DataFrame(coords_gridded, columns=coords_df.columns)
        coords_gridded_df = coords_gridded_df.drop_duplicates()
        
        print(f"  Original points: {len(coords_df)}, Gridded points: {len(coords_gridded_df)}")
        
        # Extract features at gridded locations
        # Modify feature extractor to use specific resolution
        features_df = feature_extractor.add_features(coords_gridded_df, scale=resolution_m)
        
        return features_df


    def calculate_spatial_autocorrelation(self, data_df, coord_cols, feature_cols):
        """
        Calculate Moran's I spatial autocorrelation for the features.
        High autocorrelation suggests resolution may be too fine.
        
        Args:
            data_df: DataFrame with coordinates and features
            coord_cols: List of coordinate column names
            feature_cols: List of feature column names
        
        Returns:
            float: Average Moran's I across all features
        """
        from scipy.spatial.distance import pdist, squareform
        
        # Get coordinates
        coords = data_df[coord_cols].values
        
        # Calculate distance matrix
        distances = squareform(pdist(coords))
        
        # Create spatial weights (inverse distance)
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0 / distances
            weights[np.isinf(weights)] = 0
            weights[np.isnan(weights)] = 0
        
        # Normalize weights
        row_sums = weights.sum(axis=1)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums[:, np.newaxis]
        
        # Calculate Moran's I for each feature
        morans_i_list = []
        
        for feature in feature_cols:
            values = data_df[feature].values
            
            # Remove NaN values
            valid_idx = ~np.isnan(values)
            if valid_idx.sum() < 3:
                continue
            
            values_valid = values[valid_idx]
            weights_valid = weights[valid_idx][:, valid_idx]
            
            # Calculate Moran's I
            n = len(values_valid)
            mean_val = values_valid.mean()
            
            numerator = np.sum(weights_valid * np.outer(values_valid - mean_val, values_valid - mean_val))
            denominator = np.sum((values_valid - mean_val) ** 2)
            
            if denominator > 0:
                morans_i = (n / weights_valid.sum()) * (numerator / denominator)
                morans_i_list.append(morans_i)
        
        return np.mean(morans_i_list) if morans_i_list else 0.0


    def plot_resolution_sensitivity(self, summary_df, genus_name, output_folder):
        """
        Create visualization of resolution sensitivity analysis results.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Resolution Sensitivity Analysis: {genus_name}', fontsize=16, fontweight='bold')
        
        # Filter out rows with errors
        valid_df = summary_df[~summary_df['accuracy'].isna()].copy()
        
        if len(valid_df) == 0:
            print("Warning: No valid results to plot")
            return
        
        resolutions = valid_df['resolution_m'].values
        
        # Plot 1: Accuracy vs Resolution
        axes[0, 0].plot(resolutions, valid_df['accuracy'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Resolution (m)', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Accuracy vs Resolution')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
        
        # Plot 2: TPR and TNR vs Resolution
        axes[0, 1].plot(resolutions, valid_df['tpr'], 'go-', label='TPR', linewidth=2, markersize=8)
        axes[0, 1].plot(resolutions, valid_df['tnr'], 'ro-', label='TNR', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Resolution (m)', fontsize=12)
        axes[0, 1].set_ylabel('Rate', fontsize=12)
        axes[0, 1].set_title('TPR and TNR vs Resolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xscale('log')
        
        # Plot 3: F1 Score vs Resolution
        axes[0, 2].plot(resolutions, valid_df['f1'], 'mo-', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Resolution (m)', fontsize=12)
        axes[0, 2].set_ylabel('F1 Score', fontsize=12)
        axes[0, 2].set_title('F1 Score vs Resolution')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xscale('log')
        
        # Plot 4: Spatial Autocorrelation vs Resolution
        axes[1, 0].plot(resolutions, valid_df['spatial_autocorrelation'], 'co-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Resolution (m)', fontsize=12)
        axes[1, 0].set_ylabel('Moran\'s I', fontsize=12)
        axes[1, 0].set_title('Spatial Autocorrelation vs Resolution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xscale('log')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 5: Feature Variance vs Resolution
        axes[1, 1].plot(resolutions, valid_df['feature_variance'], 'yo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Resolution (m)', fontsize=12)
        axes[1, 1].set_ylabel('Mean Feature Variance', fontsize=12)
        axes[1, 1].set_title('Feature Variance vs Resolution')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale('log')
        
        # Plot 6: Number of Points vs Resolution
        axes[1, 2].plot(resolutions, valid_df['num_presence_points'], 'ko-', linewidth=2, markersize=8)
        axes[1, 2].set_xlabel('Resolution (m)', fontsize=12)
        axes[1, 2].set_ylabel('Number of Points', fontsize=12)
        axes[1, 2].set_title('Sample Size vs Resolution')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_xscale('log')
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(output_folder, f'{genus_name}_resolution_sensitivity.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Resolution sensitivity plot saved to: {plot_path}")


    def find_optimal_resolution(self, summary_df):
        """
        Find the optimal resolution based on multiple criteria.
        
        Criteria:
        1. Maximize accuracy
        2. Balance TPR and TNR
        3. Minimize spatial autocorrelation (avoid redundancy)
        4. Maintain sufficient sample size
        
        Args:
            summary_df: DataFrame with resolution analysis results
        
        Returns:
            int: Optimal resolution in meters
        """
        # Filter valid results
        valid_df = summary_df[~summary_df['accuracy'].isna()].copy()
        
        if len(valid_df) == 0:
            return None
        
        # Normalize metrics to 0-1 range for comparison
        def normalize(series):
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(series))
            return (series - min_val) / (max_val - min_val)
        
        # Calculate composite score
        valid_df['accuracy_norm'] = normalize(valid_df['accuracy'])
        valid_df['f1_norm'] = normalize(valid_df['f1'])
        valid_df['balance_score'] = 1 - abs(valid_df['tpr'] - valid_df['tnr'])  # Want TPR and TNR balanced
        valid_df['balance_norm'] = normalize(valid_df['balance_score'])
        
        # Lower spatial autocorrelation is better (less redundancy)
        valid_df['autocorr_norm'] = 1 - normalize(valid_df['spatial_autocorrelation'])
        
        # Higher sample size is better (but diminishing returns)
        valid_df['sample_size_norm'] = normalize(np.log1p(valid_df['num_presence_points']))
        
        # Composite score (weighted average)
        valid_df['composite_score'] = (
            0.30 * valid_df['accuracy_norm'] +
            0.25 * valid_df['f1_norm'] +
            0.20 * valid_df['balance_norm'] +
            0.15 * valid_df['autocorr_norm'] +
            0.10 * valid_df['sample_size_norm']
        )
        
        # Find resolution with highest composite score
        optimal_idx = valid_df['composite_score'].idxmax()
        optimal_resolution = valid_df.loc[optimal_idx, 'resolution_m']
        
        print(f"\nOptimal Resolution Selection:")
        print(f"  Resolution: {optimal_resolution}m")
        print(f"  Accuracy: {valid_df.loc[optimal_idx, 'accuracy']:.4f}")
        print(f"  F1 Score: {valid_df.loc[optimal_idx, 'f1']:.4f}")
        print(f"  TPR: {valid_df.loc[optimal_idx, 'tpr']:.4f}, TNR: {valid_df.loc[optimal_idx, 'tnr']:.4f}")
        print(f"  Spatial Autocorr: {valid_df.loc[optimal_idx, 'spatial_autocorrelation']:.4f}")
        print(f"  Sample Size: {int(valid_df.loc[optimal_idx, 'num_presence_points'])}")
        print(f"  Composite Score: {valid_df.loc[optimal_idx, 'composite_score']:.4f}")
        
        return int(optimal_resolution)

if __name__ == "__main__":
    import ee
    from . import utility  # Add this import

    # Authenticate and initialize Earth Engine with project
    ee.Authenticate()
    ee.Initialize(project='ee-mtpictd')

    genus_list = ["Syzygium"]

    models = Models()

    for genus in genus_list:
    # Write genus name to file
        with open(f"{genus}_name.txt", 'w') as f:
            f.write(genus)
        
        # 1. Load/fetch GBIF data and extract features
        output_file = utility.fetch_gbif_and_extract_features(
            ee=ee,
            max_points=3000,
            filter_forest=True
        )
        # output_file=f"presence_points_{genus}.csv"
        
        
        # base_output = f"outputs/resolution_analysis_{genus}"
        # summary_df, optimal_resolution = models.resolution_sensitivity_analysis(
        #     genus_name=genus,
        #     presence_path=output_file,
        #     model_type="Random Forest",
        #     loss_func="Dice",
        #     is_bias=False,
        #     is_endemic=False,
        #     base_output_folder=base_output,
        #     resolutions=[10, 30, 50, 100, 250, 500, 1000]  # Test these resolutions
        # )
    
        # print(f"\n✓ Optimal resolution for {genus}: {optimal_resolution}m")
        
        # 2. Calculate endemicity index (number of ecoregions)
        endemicity_index, ecoregions_list, endemic_classification = utility.calculate_endemicity_index(
            presence_points_csv=output_file,
            ecoregion_wkt_directory="data/eco_regions_polygon"
        )
        
        # 4. Calculate entropy index
        entropy_index, entropy_contributions, ecoregion_counts, entropy_classification = utility.calculate_entropy_index(
            presence_points_csv=output_file,
            ecoregion_wkt_directory="data/eco_regions_polygon"
        )
        
        # Print the values
        print(f"\n{'='*60}")
        print(f"Results for {genus}:")
        print(f"{'='*60}")
        print(f"Endemicity Index: {endemicity_index} ({endemic_classification})")
        print(f"  - Species is {endemic_classification}")
        print(f"Entropy Index: {entropy_index:.4f} ({entropy_classification})")
        print(f"  - Species is {entropy_classification}")
        print(f"Ecoregions list: {ecoregions_list}")
        print(f"{'='*60}\n")
        
        # Determine training region, model, loss function, and bias correction based on classification
        # Based on the table in the image
        
        # Check if species is concentrated or non-concentrated
        is_concentrated = (entropy_classification == "concentrated")
        
        # Check if species is endemic or non-endemic
        is_endemic = (endemic_classification == "endemic")
        
        # Determine parameters based on classification
        if not is_concentrated and not is_endemic:
            # Non Concentrated, Non Endemic -> All India, Random Forest, Dice, No Bias
            training_region = "All India"
            model_type = "Random Forest"
            loss_function = "Dice"
            bias_correction = False
            print(f"Classification: Non Concentrated, Non Endemic")
            print(f"Training on: {training_region}, Model: {model_type}, Loss: {loss_function}, Bias Correction: {bias_correction}")
            models.comprehensive_genus_modeling(
                genus_name=genus,
                model=model_type,
                loss_function=loss_function,
                bias_correction=bias_correction,
                presence_path=output_file,
                is_endemic=False
            )
        
        elif not is_concentrated and is_endemic:
            # Non Concentrated, Endemic -> All India, Random Forest, Tversky, Yes Bias
            training_region = "All India"
            model_type = "Random Forest"
            loss_function = "Dice"
            bias_correction = False
            print(f"Classification: Non Concentrated, Endemic")
            print(f"Training on: {training_region}, Model: {model_type}, Loss: {loss_function}, Bias Correction: {bias_correction}")
            models.comprehensive_genus_modeling(
                genus_name=genus,
                model_type=model_type,
                loss_func=loss_function,
                is_bias=bias_correction,
                presence_path=output_file,
                is_endemic=True
            )
        
        elif is_concentrated and not is_endemic:
            # Concentrated, Non Endemic -> Ecoregion, Random Forest, Tversky, Yes Bias
            training_region = "Ecoregion"
            model_type = "Random Forest"
            loss_function = "Tversky"
            bias_correction = True
            print(f"Classification: Concentrated, Non Endemic")
            print(f"Training on: {training_region}, Model: {model_type}, Loss: {loss_function}, Bias Correction: {bias_correction}")
            models.comprehensive_genus_modeling(
                genus_name=genus,
                model_type=model_type,
                loss_func=loss_function,
                is_bias=bias_correction,
                presence_path=output_file,
                is_endemic=False
            )
        
        elif is_concentrated and is_endemic:
            # Concentrated, Endemic -> Ecoregion, Random Forest, Dice, Yes Bias
            training_region = "Ecoregion"
            model_type = "Random Forest"
            loss_function = "Dice"
            bias_correction = True
            print(f"Classification: Concentrated, Endemic")
            print(f"Training on: {training_region}, Model: {model_type}, Loss: {loss_function}, Bias Correction: {bias_correction}")
            models.train_on_max_ecoregion_for_genus(
                genus_name=genus,
                model_type=model_type,
                loss_func=loss_function,
                is_bias=bias_correction,
                presence_path=output_file,
                is_endemic=True
            )
        
        print(f"\nTraining completed for {genus}\n")