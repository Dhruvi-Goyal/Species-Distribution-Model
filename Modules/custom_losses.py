import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import binary_crossentropy

# ----------------------------------------
# FOCAL LOSS CLASS
# ----------------------------------------
class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha  # Class balancing parameter
        self.gamma = gamma  # Focusing parameter to reduce impact of easy examples

    def __call__(self, y_true, y_pred):
        # Clip predictions to avoid log(0) errors
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Compute standard cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Probability of correct classification
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        # Weighting factors
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        # Final focal loss
        return alpha_factor * modulating_factor * cross_entropy

# ----------------------------------------
# DICE LOSS CLASS
# ----------------------------------------
class DiceLoss:
    def __init__(self, smooth=1.0):
        self.smooth = smooth  # Smoothing factor to avoid division by zero

    def __call__(self, y_true, y_pred):
        # Clip predictions to avoid numerical instability
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Compute Dice coefficient
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss
        return 1. - dice

# ----------------------------------------
# TVERSKY LOSS CLASS
# ----------------------------------------
class TverskyLoss:
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        self.alpha = alpha  # Penalty for false positives
        self.beta = beta    # Penalty for false negatives
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        # Clip predictions
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate TP, FP, FN
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        # Tversky index
        numerator = tp + self.smooth
        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth

        # Return loss
        return 1 - (numerator / denominator)

# ----------------------------------------
# CUSTOM NEURAL NETWORK WRAPPER
# ----------------------------------------
class CustomNeuralNetwork:
    def __init__(self, loss_fn='focal', alpha=0.25, gamma=2.0, smooth=1.0, beta=0.7):
        self.loss_fn = loss_fn  # Which loss function to use
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.beta = beta
        self.model = None

    def build_model(self, input_shape):
        # Define simple dense model
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
        ])

        # Select appropriate loss function
        if self.loss_fn == 'focal':
            loss = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        elif self.loss_fn == 'dice':
            loss = DiceLoss(smooth=self.smooth)
        elif self.loss_fn == 'tversky':
            loss = TverskyLoss(alpha=self.alpha, beta=self.beta, smooth=self.smooth)
        else:
            loss = 'binary_crossentropy'

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy', tf.keras.metrics.TruePositives()]
        )

        self.model = model
        return model

    def fit(self, X, y, sample_weights=None, epochs=50, batch_size=32, validation_split=0.2):
        # Fit the model if not already built
        if self.model is None:
            self.build_model(X.shape[1])

        # Train the model
        return self.model.fit(
            X, y,
            sample_weight=sample_weights,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

    def predict(self, X):
        # Predict binary classes
        return (self.model.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        # Predict probabilities
        return self.model.predict(X)

# ----------------------------------------
# CUSTOM SCORER: TVERSKY SCORE
# ----------------------------------------
class TverskyScorer:
    def __init__(self, alpha=0.3, beta=0.7):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        numerator = tp
        denominator = tp + self.alpha * fp + self.beta * fn

        return 0.0 if denominator == 0 else numerator / denominator

# ----------------------------------------
# CUSTOM SCORER: FOCAL SCORE
# ----------------------------------------
class FocalScorer:
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Prob of correct class
        p_t = np.where(y_true == 1, y_pred_proba, 1 - y_pred_proba)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        modulating_factor = np.power(1.0 - p_t, self.gamma)

        # Weighted accuracy
        weighted_correct = np.sum(alpha_t * modulating_factor * (y_true == y_pred))
        total_weight = np.sum(alpha_t * modulating_factor)

        return 0.0 if total_weight == 0 else weighted_correct / total_weight

# ----------------------------------------
# CUSTOM SCORER: DICE SCORE
# ----------------------------------------
class DiceScorer:
    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def __call__(self, y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba >= threshold).astype(int)
        intersection = np.sum((y_true == 1) & (y_pred == 1))
        union = np.sum(y_true == 1) + np.sum(y_pred == 1)

        return (2. * intersection + self.smooth) / (union + self.smooth)

# ----------------------------------------
# THRESHOLD OPTIMIZATION FUNCTION
# ----------------------------------------
def optimize_threshold_for_tpr(y_true, y_pred_proba, min_accuracy=0.5):
    """
    Optimize the decision threshold to maximize TPR (recall)
    while keeping accuracy above a minimum acceptable level.
    """
    thresholds = np.linspace(0.1, 0.9, 20)
    best_threshold = 0.5
    best_tpr = 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tpr = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        if tpr > best_tpr and accuracy >= min_accuracy:
            best_tpr = tpr
            best_threshold = threshold

    return best_threshold, best_tpr

def optimize_threshold_for_metric(y_true, y_pred_proba, metric='tpr', min_accuracy=0.5):
    """
    Generic threshold optimization function that can optimize for different metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    metric : str, default='tpr'
        Metric to optimize for: 'tpr', 'tnr', 'f1', 'balanced_accuracy'
    min_accuracy : float, default=0.5
        Minimum accuracy threshold
        
    Returns:
    --------
    tuple
        (optimal_threshold, best_metric_value)
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
    
    thresholds = np.linspace(0.10, 0.90, 100)
    best_threshold = 0.5
    best_metric_value = 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        
        if accuracy < min_accuracy:
            continue
            
        if metric == 'tpr':
            metric_value = recall_score(y_true, y_pred)
        elif metric == 'tnr':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metric_value = tn / (tn + fp) if (tn + fp) > 0 else 0
        elif metric == 'f1':
            metric_value = f1_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            metric_value = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}. Supported: 'tpr', 'tnr', 'f1', 'balanced_accuracy'")

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold

    return best_threshold, best_metric_value
