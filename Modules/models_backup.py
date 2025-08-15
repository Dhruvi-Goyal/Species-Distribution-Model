# Import necessary libraries for data manipulation, machine learning, and geospatial analysis
import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.linear_model import LogisticRegression
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
        reliability_absence = np.array([w**(0.03) for w in reliability_absence])
        
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
        # Shuffle the data along with both sets of weights
        # ------------------------------------
        # Randomly shuffle all arrays together to ensure proper randomization
        X, y, coords, reliability_weights, bias_weights = shuffle(
            X, y, coords, reliability_weights, bias_weights, random_state=42
        )
        
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
                ee.Initialize()
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
            
            reliability_presence = np.ones(len(presence_features))
            reliability_absence = np.ones(len(absence_features))
            reliability_weights = np.hstack([reliability_presence, reliability_absence])
            # Apply the same mask to weights
            reliability_weights = reliability_weights[mask]
            # Normalize reliability weights
            min_rel = np.min(reliability_weights)
            max_rel = np.max(reliability_weights)
            if max_rel != min_rel:
                reliability_weights = (reliability_weights - min_rel) / (max_rel - min_rel)
            reliability_weights = np.array([w**(0.03) for w in reliability_weights])
            bias_weights = np.ones(len(X))
            combined_weights = reliability_weights * bias_weights
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
            print(f"{'Model/Loss':<35} {'Accuracy':<10} {'TPR':<8} {'TNR':<8} {'Threshold':<10}")
            print(f"{'-'*80}")
            for _, row in results_df.iterrows():
                print(f"{row['model_loss']:<35} {row['accuracy']:<10.4f} {row['tpr']:<8.4f} {row['tnr']:<8.4f} {row['optimal_threshold']:<10.4f}")
            
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
        pseudo_absence_df = pseudo_absence_df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
        pseudo_absence_df = pseudo_absence_df[['longitude', 'latitude']]
        # Only keep pseudo-absence points in the max ecoregion and not in presence
        pseudo_absence_points = [Point(lon, lat) for lon, lat in zip(pseudo_absence_df['longitude'], pseudo_absence_df['latitude'])]
        pseudo_absence_df = pseudo_absence_df[[eco_polygons[max_eco].contains(pt) and (pt.x, pt.y) not in presence_coords for pt in pseudo_absence_points]]
        # Sample the same number of pseudo-absences as presences (or all if fewer available)
        if len(pseudo_absence_df) > len(presence_df):
            pseudo_absence_df = pseudo_absence_df.sample(n=len(presence_df), random_state=42)
        print(f"   Filtered to {len(presence_df)} presence and {len(pseudo_absence_df)} pseudo-absence points in {max_eco}")
        # Feature extraction
        try:
            import ee
            ee.Initialize()
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
        reliability_presence = np.ones(len(presence_features))
        reliability_absence = np.ones(len(absence_features))
        reliability_weights = np.hstack([reliability_presence, reliability_absence])
        reliability_weights = reliability_weights[mask]
        min_rel = np.min(reliability_weights)
        max_rel = np.max(reliability_weights)
        if max_rel != min_rel:
            reliability_weights = (reliability_weights - min_rel) / (max_rel - min_rel)
        reliability_weights = np.array([w**(0.03) for w in reliability_weights])
        bias_weights = np.ones(len(X))
        combined_weights = reliability_weights * bias_weights
        print(f"   Prepared {len(X)} total samples after cleaning")
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
                            scorer = DiceScorer(smooth=loss_params.get('smooth', 1.0))
                        elif loss_type == 'focal':
                            scorer = FocalScorer(alpha=loss_params.get('alpha', 0.25), gamma=loss_params.get('gamma', 2.0))
                        elif loss_type == 'tversky':
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
                        # Print classification report
                        print(metrics['classification_report'])
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
                except Exception as e:
                    print(f"      Error in {combination_name}: {str(e)}")
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

    def comprehensive_genus_modeling(self, genus_name):
        """
        Comprehensive modeling for genus-level distribution modeling.
        This function filters data by genus and calls the species modeling pipeline.
        
        Parameters:
        -----------
        genus_name : str
            Name of the genus to model (e.g., "Ficus", "Erythrina")
        """
        print(f"\n{'='*80}")
        print(f"STARTING GENUS-LEVEL MODELING FOR: {genus_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Load all presence data
            all_presence_path = "data/testing_SDM/all_presence_point.csv"
            if not os.path.exists(all_presence_path):
                print(f"Error: {all_presence_path} not found!")
                return
            
            all_presence_df = pd.read_csv(all_presence_path)
            
            # Check if genus column exists
            if 'genus' not in all_presence_df.columns:
                print("Error: 'genus' column not found in presence data!")
                return
            
            # Filter data for the specific genus
            genus_presence_df = all_presence_df[all_presence_df['genus'].str.lower() == genus_name.lower()]
            
            if len(genus_presence_df) == 0:
                print(f"Error: No presence points found for genus '{genus_name}'")
                return
            
            print(f"Found {len(genus_presence_df)} presence points for genus {genus_name}")
            
            # Create temporary presence file for the genus
            temp_presence_path = f"data/temp_presence_{genus_name.lower()}.csv"
            genus_presence_df.to_csv(temp_presence_path, index=False)
            
            # Load and filter absence data (if it has genus column)
            all_absence_path = "data/testing_SDM/absence_points_all_india.csv"
            if os.path.exists(all_absence_path):
                all_absence_df = pd.read_csv(all_absence_path)
                if 'genus' in all_absence_df.columns:
                    genus_absence_df = all_absence_df[all_absence_df['genus'].str.lower() == genus_name.lower()]
                    temp_absence_path = f"data/temp_absence_{genus_name.lower()}.csv"
                    genus_absence_df.to_csv(temp_absence_path, index=False)
                else:
                    # Use all absence points if no genus column
                    temp_absence_path = all_absence_path
            else:
                # Use default absence file
                temp_absence_path = "data/pseudo_absence.csv"
            
            # Call the comprehensive species modeling with genus name
            self.comprehensive_species_modeling(genus_name, 
                                              presence_path=temp_presence_path,
                                              absence_path=temp_absence_path)
            
            # Clean up temporary files
            if os.path.exists(temp_presence_path):
                os.remove(temp_presence_path)
            if os.path.exists(temp_absence_path) and temp_absence_path != all_absence_path:
                os.remove(temp_absence_path)
                
            print(f"\n{'='*80}")
            print(f"COMPLETED GENUS-LEVEL MODELING FOR: {genus_name.upper()}")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"Error in genus-level modeling for {genus_name}: {str(e)}")
            import traceback
            traceback.print_exc()


def perform_feature_importance_for_all_species(species_list):
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


if __name__ == "__main__":
    # Initialize Earth Engine (required for feature extraction)
    import ee
    
    ee.Initialize()
    
    # Initialize models
    models = Models()
    
    # Define taxa for modeling
    genus_list = ["Ficus", "Erythrina"]  # Genus-level modeling
    species_list = ["Dalbergia sissoo", "Syzygium cumini"]  # Species-level modeling
    
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE TAXA MODELING")
    print("="*80)
    
    # Run genus-level modeling
    print(f"\n{'='*50}")
    print("GENUS-LEVEL MODELING")
    print(f"{'='*50}")
    for genus in genus_list:
        try:
            models.comprehensive_genus_modeling(genus)
        except Exception as e:
            print(f"Error modeling genus {genus}: {str(e)}")
    
    # Run species-level modeling
    print(f"\n{'='*50}")
    print("SPECIES-LEVEL MODELING")
    print(f"{'='*50}")
    for species in species_list:
        try:
            models.comprehensive_species_modeling(species)
        except Exception as e:
            print(f"Error modeling species {species}: {str(e)}")
    
    print(f"\n{'='*80}")
    print("COMPLETED ALL TAXA MODELING")
    print(f"{'='*80}")