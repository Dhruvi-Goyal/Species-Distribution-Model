"""
Custom Loss Trainers Module - Docstrings for Automatic Documentation

This module contains docstrings for all functions in custom_loss_trainers.py
that can be used for automatic documentation generation using tools like Sphinx.
"""

def dice_loss_docstring():
    """
    Calculate Dice loss for binary classification.
    
    Dice loss is a metric that measures the overlap between predicted and actual
    binary masks. It's particularly useful for imbalanced datasets.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth binary labels (0 or 1)
    y_pred : numpy.ndarray
        Predicted probabilities (values between 0 and 1)
    smooth : float, default=1.0
        Smoothing factor to prevent division by zero
        
    Returns
    -------
    float
        Dice loss value between 0 and 1
        
    Formula
    -------
    Dice Loss = 1 - (2 * intersection + smooth) / (union + smooth)
    
    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1])
    >>> y_pred = np.array([0.9, 0.1, 0.8, 0.7])
    >>> loss = dice_loss(y_true, y_pred)
    """
    pass

def focal_loss_docstring():
    """
    Calculate Focal loss for binary classification.
    
    Focal loss is designed to address class imbalance by down-weighting
    easy examples and focusing on hard examples.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth binary labels (0 or 1)
    y_pred : numpy.ndarray
        Predicted probabilities (values between 0 and 1)
    alpha : float, default=0.25
        Weight for the rare class (alpha > 0.5 down-weights the rare class)
    gamma : float, default=2.0
        Focusing parameter (gamma > 0 down-weights easy examples)
        
    Returns
    -------
    float
        Focal loss value
        
    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1])
    >>> y_pred = np.array([0.9, 0.1, 0.8, 0.7])
    >>> loss = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)
    """
    pass

def tversky_loss_docstring():
    """
    Calculate Tversky loss for binary classification.
    
    Tversky loss is a generalization of Dice loss that allows different
    weights for false positives and false negatives.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth binary labels (0 or 1)
    y_pred : numpy.ndarray
        Predicted probabilities (values between 0 and 1)
    alpha : float, default=0.3
        Weight for false positives (alpha + beta should sum to 1)
    beta : float, default=0.7
        Weight for false negatives (alpha + beta should sum to 1)
    smooth : float, default=1.0
        Smoothing factor to prevent division by zero
        
    Returns
    -------
    float
        Tversky loss value
        
    Notes
    -----
    When alpha = beta = 0.5, Tversky loss becomes Dice loss.
    """
    pass

def comprehensive_species_with_precomputed_features_docstring():
    """
    Perform comprehensive species-level modeling with precomputed features.
    
    This function provides end-to-end species distribution modeling including:
    - Data preprocessing and quality checks
    - Automatic absence point selection
    - Multiple model types (Random Forest, Logistic Regression, Weighted Logistic Regression)
    - Multiple loss functions (Dice, Focal, Tversky)
    - Comprehensive evaluation metrics
    - Optional bias correction for ecoregion-based weighting
    - Feature importance analysis using SHAP
    - Model persistence and result saving
    
    Parameters
    ----------
    species : str
        Name of the species to model
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to the CSV file containing presence points with extracted features
    optimize_for : str, default='tpr'
        Metric to optimize for ('tpr', 'f1', 'auc')
    reliability_threshold : float, default=0.07
        Threshold for filtering absence points based on reliability scores
    bias_correction : bool, default=False
        Whether to apply bias correction using ecoregion-based weighting
        
    Returns
    -------
    dict
        Dictionary containing comprehensive modeling results including:
        - Model performance metrics
        - Feature importance scores
        - Best model parameters
        - Training and validation results
        - Model files and output paths
        
    Examples
    --------
    >>> results = comprehensive_species_with_precomputed_features("Artemisia")
    >>> results_with_bias = comprehensive_species_with_precomputed_features(
    ...     "Artemisia", bias_correction=True
    ... )
    
    Notes
    -----
    The function automatically:
    1. Loads presence and absence data
    2. Performs data quality checks
    3. Trains multiple models with different loss functions
    4. Evaluates performance using cross-validation
    5. Saves models and results to the outputs directory
    6. Generates feature importance plots and SHAP analysis
    """
    pass

def comprehensive_genus_with_precomputed_features_docstring():
    """
    Perform comprehensive genus-level modeling with precomputed features.
    
    Similar to species-level modeling but operates at the genus level,
    aggregating data from multiple species within the same genus.
    
    Parameters
    ----------
    genus : str
        Name of the genus to model
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to the CSV file containing presence points with extracted features
    optimize_for : str, default='tpr'
        Metric to optimize for ('tpr', 'f1', 'auc')
    reliability_threshold : float, default=0.07
        Threshold for filtering absence points based on reliability scores
    bias_correction : bool, default=False
        Whether to apply bias correction using ecoregion-based weighting
        
    Returns
    -------
    dict
        Dictionary containing comprehensive modeling results
        
    Examples
    --------
    >>> results = comprehensive_genus_with_precomputed_features("Artemisia")
    """
    pass

def ecoregion_level_species_with_precomputed_features_docstring():
    """
    Perform ecoregion-level species modeling with precomputed features.
    
    This function performs species distribution modeling at the ecoregion level,
    where the model is trained and evaluated within specific ecoregion boundaries.
    
    Parameters
    ----------
    species : str
        Name of the species to model
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to the CSV file containing presence points with extracted features
    optimize_for : str, default='tpr'
        Metric to optimize for ('tpr', 'f1', 'auc')
    bias_correction : bool, default=False
        Whether to apply bias correction using ecoregion-based weighting
        
    Returns
    -------
    dict
        Dictionary containing ecoregion-level modeling results
        
    Notes
    -----
    Ecoregion-level modeling:
    1. Filters data to specific ecoregion boundaries
    2. Trains models on ecoregion-specific data
    3. Evaluates performance within ecoregion context
    4. Provides ecoregion-specific feature importance
    """
    pass

def ecoregion_level_genus_with_precomputed_features_docstring():
    """
    Perform ecoregion-level genus modeling with precomputed features.
    
    Similar to ecoregion-level species modeling but operates at the genus level.
    
    Parameters
    ----------
    genus : str
        Name of the genus to model
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to the CSV file containing presence points with extracted features
    optimize_for : str, default='tpr'
        Metric to optimize for ('tpr', 'f1', 'auc')
    bias_correction : bool, default=False
        Whether to apply bias correction using ecoregion-based weighting
        
    Returns
    -------
    dict
        Dictionary containing ecoregion-level genus modeling results
    """
    pass

def bias_correction_docstring():
    """
    Apply bias correction using ecoregion-based weighting.
    
    This function implements a bias correction scheme that:
    1. Assigns each data point to its corresponding ecoregion
    2. Computes raw weights based on ecoregion occurrence counts
    3. Normalizes weights to a specified range (default: [0.5, 1.5])
    4. Applies weights during model training
    
    Parameters
    ----------
    pres_clean : pandas.DataFrame
        Presence point data with coordinates
    absence_selected : pandas.DataFrame
        Selected absence point data with coordinates
    eco_polygons : dict
        Dictionary mapping ecoregion names to polygon objects
        
    Returns
    -------
    tuple
        (presence_weights, absence_weights) - Normalized weights for presence and absence points
        
    Weighting Formula
    ----------------
    1. Raw weight: w_raw = 1/(c+1) where c is occurrence count
    2. Normalized weight: w = 0.5 + (w_raw - w_min)/(w_max - w_min)
    
    Examples
    --------
    >>> presence_weights, absence_weights = bias_correction(
    ...     presence_data, absence_data, ecoregion_polygons
    ... )
    """
    pass

def evaluate_summary_docstring():
    """
    Perform comprehensive model evaluation with multiple metrics and loss functions.
    
    This function evaluates models using multiple loss functions (Dice, Focal, Tversky)
    and provides comprehensive performance metrics including accuracy, precision,
    recall, F1-score, AUC, and balanced accuracy.
    
    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix for training/evaluation
    y : numpy.ndarray
        Target labels (0 or 1)
    all_df : pandas.DataFrame, optional
        Additional data for analysis
    min_points : int, default=10
        Minimum number of points required for evaluation
    optimize_for : str, default='tpr'
        Metric to optimize for ('tpr', 'f1', 'auc')
    species_name : str, optional
        Name of the species for logging
    genus_name : str, optional
        Name of the genus for logging
        
    Returns
    -------
    dict
        Dictionary containing evaluation results for all models and metrics
        
    Examples
    --------
    >>> results = evaluate_summary(X_train, y_train, optimize_for='f1')
    >>> print(f"Best F1 Score: {results['best_f1_score']}")
    """
    pass

def perform_feature_importance_for_all_species_docstring():
    """
    Perform feature importance analysis for multiple species using SHAP.
    
    This function analyzes feature importance for multiple species using SHAP
    (SHapley Additive exPlanations) values, providing both global and local
    feature importance measures.
    
    Parameters
    ----------
    species_list : list
        List of species names to analyze
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to the CSV file containing presence points with extracted features
        
    Returns
    -------
    dict
        Dictionary containing feature importance results for all species
        
    Output Files
    ------------
    For each species, generates:
    - SHAP importance scores CSV file
    - SHAP importance plot (PNG)
    - SHAP summary plot (PNG)
    
    Examples
    --------
    >>> species_list = ["Artemisia", "Artocarpus heterophyllus"]
    >>> results = perform_feature_importance_for_all_species(species_list)
    """
    pass

def test_custom_model_on_all_ecoregions_docstring():
    """
    Test a trained model on all ecoregions in parallel.
    
    This function applies a trained model to all ecoregion polygons and
    calculates average probability scores for each ecoregion.
    
    Parameters
    ----------
    trained_model : object
        Trained model object with predict_proba method
    species_name : str
        Name of the species for output file naming
    output_file : str, optional
        Path to save results (if None, uses default naming)
    num_workers : int, default=16
        Number of parallel workers for processing
        
    Returns
    -------
    dict
        Dictionary containing ecoregion testing results with average probabilities
        
    Examples
    --------
    >>> results = test_custom_model_on_all_ecoregions(
    ...     trained_model, "Artemisia", num_workers=8
    ... )
    """
    pass

def visualize_presence_absence_points_docstring():
    """
    Create interactive map visualization of presence and absence points.
    
    This function creates an interactive HTML map showing the distribution
    of presence and absence points for a given species.
    
    Parameters
    ----------
    presence_df : pandas.DataFrame
        DataFrame containing presence point data with coordinates
    absence_df : pandas.DataFrame
        DataFrame containing absence point data with coordinates
    species_name : str
        Name of the species for map title
    output_file : str, optional
        Path to save the HTML map file
        
    Returns
    -------
    None
        Saves interactive map to HTML file
        
    Examples
    --------
    >>> visualize_presence_absence_points(
    ...     presence_data, absence_data, "Artemisia", "map.html"
    ... )
    """
    pass

def CustomLossRandomForest_docstring():
    """
    Random Forest classifier with custom loss functions.
    
    This class extends scikit-learn's RandomForestClassifier to work with
    custom loss functions (Dice, Focal, Tversky) for binary classification.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest
    loss_type : str, default='dice'
        Type of loss function ('dice', 'focal', 'tversky')
    loss_params : dict, optional
        Parameters for the loss function
        
    Methods
    -------
    fit(X, y, sample_weights=None, max_iter=100, restarts=10)
        Train the model with custom loss optimization
    predict_proba(X)
        Predict class probabilities
    predict(X)
        Predict class labels
        
    Examples
    --------
    >>> clf = CustomLossRandomForest(loss_type='tversky', loss_params={'alpha': 0.3})
    >>> clf.fit(X_train, y_train, sample_weights=weights)
    >>> predictions = clf.predict_proba(X_test)
    """
    pass

def CustomLossLogisticRegression_docstring():
    """
    Logistic Regression with custom loss functions and L2 regularization.
    
    This class implements logistic regression with custom loss functions
    and includes L2 regularization for model stability.
    
    Parameters
    ----------
    loss_type : str, default='dice'
        Type of loss function ('dice', 'focal', 'tversky')
    loss_params : dict, optional
        Parameters for the loss function
    max_iter : int, default=1000
        Maximum iterations for optimization
    lr : float, default=0.01
        Learning rate for gradient descent
    l2 : float, default=1e-3
        L2 regularization strength
        
    Methods
    -------
    fit(X, y, sample_weights=None)
        Train the model with custom loss
    predict_proba(X)
        Predict class probabilities
    predict(X)
        Predict class labels
        
    Examples
    --------
    >>> clf = CustomLossLogisticRegression(loss_type='focal', l2=0.01)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict_proba(X_test)
    """
    pass

def CustomLossWeightedLogisticRegression_docstring():
    """
    Weighted Logistic Regression with custom loss functions.
    
    This class extends CustomLossLogisticRegression to handle weighted samples,
    useful for addressing class imbalance or applying bias correction.
    
    Parameters
    ----------
    loss_type : str, default='dice'
        Type of loss function ('dice', 'focal', 'tversky')
    loss_params : dict, optional
        Parameters for the loss function
    max_iter : int, default=1000
        Maximum iterations for optimization
    lr : float, default=0.01
        Learning rate for gradient descent
    l2 : float, default=1e-3
        L2 regularization strength
        
    Methods
    -------
    fit(X, y, sample_weights=None)
        Train the model with sample weights
    predict_proba(X)
        Predict class probabilities
    predict(X)
        Predict class labels
        
    Examples
    --------
    >>> clf = CustomLossWeightedLogisticRegression(loss_type='dice')
    >>> clf.fit(X_train, y_train, sample_weights=weights)
    >>> predictions = clf.predict_proba(X_test)
    """
    pass 