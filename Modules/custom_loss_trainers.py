'''
All-in-one custom loss trainers for Species Distribution Modeling
- Random Forest, Logistic Regression (with L2 regularization), Weighted Logistic Regression
- Each with Dice, Focal, and Tversky variants
- Includes multi-restart ensemble optimization, loss sweeps, AUC evaluation,
  automatic best-Tversky selection by F1 score, and full metrics summary
'''
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, recall_score, balanced_accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import concurrent.futures
import os
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
import queue
import threading
import time
from . import utility
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from .features_extractor import Feature_Extractor
import glob
import shap
import joblib
from geopy.distance import geodesic
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Assume Feature_Extractor is available
from .features_extractor import Feature_Extractor



def process_single_ecoregion(filename, polygon_dir, clf, Features_extractor, modelss):
    """Process a single ecoregion file and return the average probability."""
    ecoregion_name = os.path.splitext(filename)[0]
    polygon_path = os.path.join(polygon_dir, filename)
    
    # Read the polygon WKT
    with open(polygon_path, 'r') as file:
        polygon_wkt = file.read().strip()
    
    # Generate test data for the current ecoregion
    X_dissimilar = Features_extractor.add_features(
        utility.divide_polygon_to_grids(polygon_wkt, grid_size=1, points_per_cell=20)
    )
    
    # Create a unique temporary file for this thread
    thread_id = threading.get_ident()
    timestamp = int(time.time() * 1000)
    temp_file = f'data/temp_presence_{thread_id}_{timestamp}.csv'
    pd.DataFrame(X_dissimilar).to_csv(temp_file, index=False)
    
    try:
        X_test, y_test, _, _, _,bias_weights = modelss.load_data(
            presence_path=temp_file,
            absence_path='data/test_absence.csv'
        )
        
        # Remove NaN and infinite values from test set
        X_test = np.array(X_test, dtype=float)
        mask = np.isfinite(X_test).all(axis=1)
        X_test = X_test[mask]
        
        if X_test.shape[0] == 0:  # If no valid samples remain
            print(f'No valid samples for {ecoregion_name}. Setting average probability to 0.')
            avg_probability = 0
        else:
            # Make predictions
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            # Calculate the average probability
            avg_probability = y_proba.mean()
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
    
    return avg_probability


# =========================
# Loss Functions
# =========================
def dice_loss(y_true, y_pred, smooth=1.0):
    I = np.sum(y_true * y_pred)
    U = np.sum(y_true) + np.sum(y_pred)
    return 1 - (2 * I + smooth) / (U + smooth)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    eps = 1e-7
    p = np.clip(y_pred, eps, 1 - eps)
    pt = np.where(y_true == 1, p, 1 - p)
    af = np.where(y_true == 1, alpha, 1 - alpha)
    mod = (1 - pt) ** gamma
    return np.mean(-af * mod * np.log(pt))

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1.0):
    I = np.sum(y_true * y_pred)
    FP = np.sum((1 - y_true) * y_pred)
    FN = np.sum(y_true * (1 - y_pred))
    return 1 - (I + smooth) / (I + alpha * FP + beta * FN + smooth)

# =========================
# Custom Random Forest
# =========================
class CustomLossRandomForest:
    def __init__(self, n_estimators=100, loss_type='dice', loss_params=None):
        self.n_estimators = n_estimators
        self.loss_type = loss_type
        self.loss_params = loss_params or {}
        self.trees = []
        self.weights = None

    def _loss(self, y, p):
        return {
            'dice': dice_loss,
            'focal': focal_loss,
            'tversky': tversky_loss
        }[self.loss_type](y, p, **self.loss_params)

    def fit(self, X, y, sample_weights=None, max_iter=100, restarts=10):
        rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
        rf.fit(X, y, sample_weight=sample_weights)
        P = np.vstack([t.predict_proba(X)[:, 1] for t in rf.estimators_]).T
        best_loss, best_w = np.inf, None
        for _ in range(restarts):
            init = np.random.dirichlet(np.ones(P.shape[1]))
            def obj(w):
                w_ = np.abs(w)
                w_ /= w_.sum()
                return self._loss(y, P.dot(w_))
            try:
                res = minimize(obj, init, bounds=[(0, 1)] * P.shape[1], method='L-BFGS-B', options={'maxiter': max_iter})
                w = np.abs(res.x)
                w /= w.sum()
                lv = self._loss(y, P.dot(w))
                if lv < best_loss:
                    best_loss, best_w = lv, w
            except:
                pass
        self.weights = best_w if best_w is not None else np.ones(P.shape[1]) / P.shape[1]
        self.trees = rf.estimators_
        return self

    def predict_proba(self, X):
        P = np.vstack([t.predict_proba(X)[:, 1] for t in self.trees]).T
        ens = P.dot(self.weights)
        return np.column_stack([1 - ens, ens])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# =========================
# Custom Logistic Regression
# =========================
class CustomLossLogisticRegression:
    def __init__(self, loss_type='dice', loss_params=None, max_iter=1000, lr=0.01, l2=1e-3):
        self.loss_type, self.loss_params = loss_type, loss_params or {}
        self.max_iter, self.lr, self.l2 = max_iter, lr, l2
        self.coef_, self.intercept_ = None, None

    def _sig(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _loss(self, y, p):
        return {
            'dice': dice_loss,
            'focal': focal_loss,
            'tversky': tversky_loss
        }[self.loss_type](y, p, **self.loss_params)

    def fit(self, X, y, sample_weights=None):
        X, y = X.astype(float), y.astype(float)
        n, m = X.shape
        coef, intc = np.zeros(m), 0.0
        for _ in range(self.max_iter):
            z = X.dot(coef) + intc
            p = self._sig(z)
            loss = self._loss(y, p) + 0.5 * self.l2 * np.sum(coef**2)
            grad = p - y
            if sample_weights is not None:
                grad *= sample_weights
            denom = sample_weights.sum() if sample_weights is not None else n
            grad_c = (X.T.dot(grad) + self.l2 * coef) / denom
            grad_i = grad.mean()
            coef -= self.lr * grad_c
            intc -= self.lr * grad_i
            if loss < 1e-6:
                break
        self.coef_, self.intercept_ = coef.reshape(1, -1), np.array([intc])
        return self

    def predict_proba(self, X):
        z = X.dot(self.coef_.T) + self.intercept_
        p = self._sig(z).flatten()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# =========================
# Custom Weighted Logistic Regression
# =========================
class CustomLossWeightedLogisticRegression:
    def __init__(self, loss_type='dice', loss_params=None, max_iter=1000, lr=0.01, l2=1e-3):
        self.loss_type, self.loss_params = loss_type, loss_params or {}
        self.max_iter, self.lr, self.l2 = max_iter, lr, l2
        self.coef_, self.intercept_ = None, None

    def _sig(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _loss(self, y, p):
        return {
            'dice': dice_loss,
            'focal': focal_loss,
            'tversky': tversky_loss
        }[self.loss_type](y, p, **self.loss_params)

    def fit(self, X, y, sample_weights=None):
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
        
        n, m = X.shape
        coef, intc = np.zeros(m), 0.0
        
        # Use sample weights in the optimization process
        for _ in range(self.max_iter):
            z = X.dot(coef) + intc
            p = self._sig(z)
            
            # Calculate loss with L2 regularization
            loss = self._loss(y, p) + 0.5 * self.l2 * np.sum(coef**2)
            
            # Calculate gradient
            grad = p - y
            
            # Apply sample weights to gradient
            if sample_weights is not None:
                grad *= sample_weights
                denom = sample_weights.sum()
            else:
                denom = n
            
            # Calculate gradients for coefficients and intercept
            grad_c = (X.T.dot(grad) + self.l2 * coef) / denom
            grad_i = grad.mean()
            
            # Update parameters
            coef -= self.lr * grad_c
            intc -= self.lr * grad_i
            
            if loss < 1e-6:
                break
        
        self.coef_, self.intercept_ = coef.reshape(1, -1), np.array([intc])
        return self

    def predict_proba(self, X):
        z = X.dot(self.coef_.T) + self.intercept_
        p = self._sig(z).flatten()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# =========================
# Helpers & Configs
# =========================
train_funcs = {
    'RF': lambda X, y, sw, lt, lp: CustomLossRandomForest(100, lt, lp).fit(X, y, sw),
    'LR': lambda X, y, sw, lt, lp: CustomLossLogisticRegression(lt, lp, 500, 0.05, 1e-3).fit(X, y, sw),
    'WLR': lambda X, y, sw, lt, lp: CustomLossWeightedLogisticRegression(lt, lp, 500, 0.05, 1e-3).fit(X, y, sw)
}
loss_cfgs = [('Dice', 'dice', {}), ('Focal', 'focal', {'alpha': 0.25, 'gamma': 2}), ('Tversky', 'tversky', {'alpha': 0.3, 'beta': 0.7})]
model_names = ['RF', 'LR', 'WLR']

# =========================
# Metrics Summary
# =========================
def evaluate_summary(X, y, all_df=None, min_points=10, optimize_for='tpr', species_name=None, genus_name=None):
    """
    Evaluate models with flexible threshold optimization and, if all_df is provided, also compute per-ecoregion accuracy on the test set.
    all_df must have the same row order as X/y and contain 'longitude' and 'latitude' columns.
    """
    from shapely.wkt import loads as load_wkt
    from shapely.geometry import Point
    import numpy as np
    import pandas as pd
   
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    
    # Add diagnostic information
    print(f"\nDIAGNOSTIC INFORMATION:")
    print(f"Dataset size: {len(X)} samples")
    print(f"Class distribution: {np.sum(y==1)} presence, {np.sum(y==0)} absence")
    print(f"Class balance: {np.sum(y==1)/len(y)*100:.1f}% presence")
    
    # Check for potential issues
    if np.sum(y==1) == np.sum(y==0):
        print("⚠ WARNING: Perfectly balanced dataset - may indicate artificial balancing")
    
    # Check feature separability
    presence_features = X[y == 1]
    absence_features = X[y == 0]
    
    # Check if any feature perfectly separates classes
    perfect_separation = False
    for i in range(min(5, X.shape[1])):  # Check first 5 features
        pres_unique = np.unique(presence_features[:, i])
        abs_unique = np.unique(absence_features[:, i])
        
        # More robust check: look for actual overlap in value ranges
        pres_min, pres_max = np.min(pres_unique), np.max(pres_unique)
        abs_min, abs_max = np.min(abs_unique), np.max(abs_unique)
        
        # Check if there's any overlap in the value ranges
        has_overlap = not (pres_max < abs_min or abs_max < pres_min)
        
        if not has_overlap:
            print(f"⚠ WARNING: Feature {i} has no value range overlap between presence and absence!")
            print(f"   Presence range: [{pres_min:.3f}, {pres_max:.3f}]")
            print(f"   Absence range: [{abs_min:.3f}, {abs_max:.3f}]")
            perfect_separation = True
    
    if perfect_separation:
        print("⚠ CRITICAL: Perfect feature separation detected - results may be unreliable!")
    
    print(f"\nModel | Loss | Accuracy | F1 | Precision | Recall | TPR | TNR | Optimal Threshold ({optimize_for.upper()})")
    print("-" * 100)
    
    # Store results for analysis
    all_results = []
    for m in model_names:
        for name, lt, lp in loss_cfgs:
            # Use stratified split to maintain class balance
            indices = np.arange(len(y))
            X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
                X, y, indices, test_size=0.2, random_state=42, stratify=y
            )
            print(f"Model: {m}, Loss: {name} | Train size: {len(X_tr)}, Test size: {len(X_te)}")
            sw = np.ones(len(y_tr)) if m == 'WLR' else None
            clf = train_funcs[m](X_tr, y_tr, sw, lt, lp)
            y_proba = clf.predict_proba(X_te)[:, 1]
            from .custom_losses import optimize_threshold_for_metric
            optimal_threshold, best_metric_value = optimize_threshold_for_metric(y_te, y_proba, metric=optimize_for, min_accuracy=0.5)
            y_pr = (y_proba >= optimal_threshold).astype(int)
            acc = accuracy_score(y_te, y_pr)
            f1s = f1_score(y_te, y_pr)
            tn, fp, fn, tp = confusion_matrix(y_te, y_pr).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0
            tnr = tn / (tn + fp) if (tn + fp) else 0
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            all_results.append({
                'model': m, 'loss': name, 'accuracy': acc, 'f1': f1s, 
                'precision': precision, 'recall': recall, 'tpr': tpr, 'tnr': tnr, 'threshold': optimal_threshold
            })
            print(f"{m} | {name} | {acc:.3f} | {f1s:.3f} | {precision:.3f} | {recall:.3f} | {tpr:.3f} | {tnr:.3f} | {optimal_threshold:.3f}")
            # --- Per-ecoregion accuracy on test set ---
            if all_df is not None:
                # Map test indices to lat/lon
                test_df = all_df.iloc[idx_te].copy().reset_index(drop=True)
                test_df['y_true'] = y_te
                test_df['y_pred'] = y_pr
                

                # Load ecoregion polygons
                eco_dir = "data/eco_regions_polygon"
                eco_polygons = {}
                for fname in os.listdir(eco_dir):
                    if fname.endswith('.wkt'):
                        eco_name = fname.replace('.wkt', '')
                        with open(os.path.join(eco_dir, fname), 'r') as f:
                            eco_polygons[eco_name] = load_wkt(f.read().strip())
                # Assign each test point to an ecoregion
                test_df['ecoregion'] = None
                test_points = [Point(lon, lat) for lon, lat in zip(test_df['longitude'], test_df['latitude'])]
                for i, pt in enumerate(test_points):
                    for eco_name, poly in eco_polygons.items():
                        if poly.contains(pt):
                            test_df.at[i, 'ecoregion'] = eco_name
                            break
                # Per-ecoregion accuracy
                results = []
                for eco_name, group in test_df.groupby('ecoregion'):
                    if eco_name is None or len(group) < min_points:
                        continue
                    acc = accuracy_score(group['y_true'], group['y_pred'])
                    
                    # Calculate TPR and TNR manually to avoid classification_report issues
                    cm = confusion_matrix(group['y_true'], group['y_pred'])
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                    elif cm.shape == (1, 1):
                        # Only one class present in y_true
                        if group['y_true'].iloc[0] == 0:
                            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                        else:
                            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                    elif cm.shape == (1, 2):
                        tn, fp = cm[0, 0], cm[0, 1]
                        fn, tp = 0, 0
                    elif cm.shape == (2, 1):
                        tn, fp = 0, 0
                        fn, tp = cm[1, 0], cm[0, 0]
                    else:
                        tn = fp = fn = tp = 0
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    results.append({
                        'ecoregion': eco_name,
                        'accuracy': acc,
                        'tpr': tpr,
                        'tnr': tnr,
                        'n_test_points': len(group),
                        'n_presence_test': int((group['y_true'] == 1).sum()),
                        'n_absence_test': int((group['y_true'] == 0).sum())
                    })
                    print(f"Ecoregion: {eco_name}, Accuracy: {acc:.3f}, TPR: {tpr:.3f}, TNR: {tnr:.3f}, n_test_points: {len(group)}")
                
                # Save to CSV for all loss functions and models
                if species_name:
                    out_csv = f"outputs/{species_name.replace(' ', '_')}_{m}_{name}_per_ecoregion_results.csv"
                elif genus_name:
                    out_csv = f"outputs/{genus_name.replace(' ', '_')}_{m}_{name}_per_ecoregion_results.csv"
                else:
                    out_csv = f"outputs/per_ecoregion_test_accuracy_{m}_{name}.csv"
                pd.DataFrame(results).to_csv(out_csv, index=False)
                print(f"Per-ecoregion test accuracy results saved to {out_csv}")
    
    # Print detailed accuracy summary
    print(f"\n{'='*80}")
    print(f"DETAILED ACCURACY SUMMARY")
    print(f"{'='*80}")
    
    # Find best performing models
    best_accuracy = max(all_results, key=lambda x: x['accuracy'])
    best_f1 = max(all_results, key=lambda x: x['f1'])
    best_tpr = max(all_results, key=lambda x: x['tpr'])
    best_tnr = max(all_results, key=lambda x: x['tnr'])
    
    print(f"🏆 BEST ACCURACY: {best_accuracy['model']}+{best_accuracy['loss']} = {best_accuracy['accuracy']:.4f}")
    print(f"🎯 BEST F1-SCORE: {best_f1['model']}+{best_f1['loss']} = {best_f1['f1']:.4f}")
    print(f"✅ BEST TPR: {best_tpr['model']}+{best_tpr['loss']} = {best_tpr['tpr']:.4f}")
    print(f"❌ BEST TNR: {best_tnr['model']}+{best_tnr['loss']} = {best_tnr['tnr']:.4f}")
    
    # Print confusion matrix for best accuracy model
    print(f"\n📊 CONFUSION MATRIX (Best Accuracy Model: {best_accuracy['model']}+{best_accuracy['loss']}):")
    print(f"Threshold: {best_accuracy['threshold']:.4f}")
    print(f"Accuracy: {best_accuracy['accuracy']:.4f}")
    print(f"Precision: {best_accuracy['precision']:.4f}")
    print(f"Recall: {best_accuracy['recall']:.4f}")
    print(f"F1-Score: {best_accuracy['f1']:.4f}")
    print(f"TPR: {best_accuracy['tpr']:.4f}")
    print(f"TNR: {best_accuracy['tnr']:.4f}")
    
    # Calculate and print balanced accuracy
    balanced_acc = (best_accuracy['tpr'] + best_accuracy['tnr']) / 2
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Print model comparison
    print(f"\n📈 MODEL COMPARISON:")
    print(f"{'Model+Loss':<25} {'Accuracy':<10} {'F1':<8} {'TPR':<8} {'TNR':<8}")
    print("-" * 60)
    for result in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['model']}+{result['loss']:<25} {result['accuracy']:<10.4f} {result['f1']:<8.4f} {result['tpr']:<8.4f} {result['tnr']:<8.4f}")
    
    # Analyze results
    print(f"\nPERFORMANCE ANALYSIS:")
    accuracies = [r['accuracy'] for r in all_results]
    f1_scores = [r['f1'] for r in all_results]
    thresholds = [r['threshold'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    
    print(f"Accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}")
    print(f"F1 score range: {min(f1_scores):.3f} - {max(f1_scores):.3f}")
    print(f"Threshold range: {min(thresholds):.3f} - {max(thresholds):.3f}")
    print(f"Precision range: {min(precisions):.3f} - {max(precisions):.3f}")
    print(f"Recall range: {min(recalls):.3f} - {max(recalls):.3f}")
    
    # Check for suspiciously high performance
    if max(accuracies) > 0.95:
        print("⚠ WARNING: Very high accuracy detected (>0.95) - results may be unreliable!")
        print("   Possible causes:")
        print("   - Data leakage")
        print("   - Perfect feature separation")
        print("   - Overly simple problem")
        print("   - Insufficient test set size")
    
    if max(f1_scores) > 0.95:
        print("⚠ WARNING: Very high F1 score detected (>0.95) - results may be unreliable!")
    
    # Check for threshold issues
    if min(thresholds) < 0.2:
        print("⚠ WARNING: Very low optimal threshold detected (<0.2) - possible calibration issues!")
        print("   This suggests:")
        print("   - Models are predicting very low probabilities")
        print("   - Poor model calibration")
        print("   - Threshold optimization may be unreliable")
    
    if max(thresholds) > 0.8:
        print("⚠ WARNING: Very high optimal threshold detected (>0.8) - possible calibration issues!")
    
    # Check for precision/recall imbalance
    precision_recall_gaps = [abs(r['precision'] - r['recall']) for r in all_results]
    if max(precision_recall_gaps) > 0.3:
        print("⚠ WARNING: Large precision-recall gaps detected - possible class imbalance issues!")
        print("   This suggests the model is biased toward one class")
    
    # Check for consistency across models
    acc_std = np.std(accuracies)
    if acc_std < 0.01:
        print("⚠ WARNING: Very low variance in accuracy across models - suspicious!")
    
    # Check for model type differences
    rf_acc = [r['accuracy'] for r in all_results if r['model'] == 'RF']
    lr_acc = [r['accuracy'] for r in all_results if r['model'] in ['LR', 'WLR']]
    
    if rf_acc and lr_acc:
        rf_avg = np.mean(rf_acc)
        lr_avg = np.mean(lr_acc)
        if abs(rf_avg - lr_avg) > 0.2:
            print("⚠ WARNING: Large performance gap between Random Forest and Linear models!")
            print(f"   RF average: {rf_avg:.3f}, Linear average: {lr_avg:.3f}")
            print("   This suggests data quality issues or overfitting")
    
    return all_results
def evaluate_summary2(X, y, optimize_for='tpr', sample_weights=None):
    """
    Evaluate models with flexible threshold optimization.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    optimize_for : str, default='tpr'
        Metric to optimize threshold for: 'tpr', 'tnr', 'f1', 'balanced_accuracy'
    sample_weights : array-like, optional
        Sample weights for training (e.g., bias correction weights)
    """
    # Add diagnostic information
    print(f"\nDIAGNOSTIC INFORMATION:")
    print(f"Dataset size: {len(X)} samples")
    print(f"Class distribution: {np.sum(y==1)} presence, {np.sum(y==0)} absence")
    print(f"Class balance: {np.sum(y==1)/len(y)*100:.1f}% presence")
    
    # Check for potential issues
    if np.sum(y==1) == np.sum(y==0):
        print("⚠ WARNING: Perfectly balanced dataset - may indicate artificial balancing")
    
    # Check feature separability
    presence_features = X[y == 1]
    absence_features = X[y == 0]
    
    # Check if any feature perfectly separates classes
    perfect_separation = False
    for i in range(min(5, X.shape[1])):  # Check first 5 features
        pres_unique = np.unique(presence_features[:, i])
        abs_unique = np.unique(absence_features[:, i])
        
        # More robust check: look for actual overlap in value ranges
        pres_min, pres_max = np.min(pres_unique), np.max(pres_unique)
        abs_min, abs_max = np.min(abs_unique), np.max(abs_unique)
        
        # Check if there's any overlap in the value ranges
        has_overlap = not (pres_max < abs_min or abs_max < pres_min)
        
        if not has_overlap:
            print(f"⚠ WARNING: Feature {i} has no value range overlap between presence and absence!")
            print(f"   Presence range: [{pres_min:.3f}, {pres_max:.3f}]")
            print(f"   Absence range: [{abs_min:.3f}, {abs_max:.3f}]")
            perfect_separation = True
    
    if perfect_separation:
        print("⚠ CRITICAL: Perfect feature separation detected - results may be unreliable!")
    
    print(f"\nModel | Loss | Accuracy | F1 | Precision | Recall | TPR | TNR | Optimal Threshold ({optimize_for.upper()})")
    print("-" * 100)
    
    # Store results for analysis
    all_results = []
    
    for m in model_names:
        for name, lt, lp in loss_cfgs:
            # Use stratified split to maintain class balance
            if sample_weights is not None:
                # Split both data and weights
                X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42, stratify=y)
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                sw_tr = np.ones(len(y_tr)) if m == 'WLR' else None
            
            clf = train_funcs[m](X_tr, y_tr, sw_tr, lt, lp)
            
            # Get probability predictions
            y_proba = clf.predict_proba(X_te)[:, 1]
            
            # Optimize threshold for specified metric
            from .custom_losses import optimize_threshold_for_metric
            optimal_threshold, best_metric_value = optimize_threshold_for_metric(y_te, y_proba, metric=optimize_for, min_accuracy=0.5)
            
            # Use optimal threshold for predictions
            y_pr = (y_proba >= optimal_threshold).astype(int)
            
            acc = accuracy_score(y_te, y_pr)
            f1s = f1_score(y_te, y_pr)
            tn, fp, fn, tp = confusion_matrix(y_te, y_pr).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0
            tnr = tn / (tn + fp) if (tn + fp) else 0
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            
            # Store results
            all_results.append({
                'model': m, 'loss': name, 'accuracy': acc, 'f1': f1s, 
                'precision': precision, 'recall': recall, 'tpr': tpr, 'tnr': tnr, 'threshold': optimal_threshold
            })
            
            print(f"{m} | {name} | {acc:.3f} | {f1s:.3f} | {precision:.3f} | {recall:.3f} | {tpr:.3f} | {tnr:.3f} | {optimal_threshold:.3f}")
    
    # Print detailed accuracy summary
    print(f"\n{'='*80}")
    print(f"DETAILED ACCURACY SUMMARY")
    print(f"{'='*80}")
    
    # Find best performing models
    best_accuracy = max(all_results, key=lambda x: x['accuracy'])
    best_f1 = max(all_results, key=lambda x: x['f1'])
    best_tpr = max(all_results, key=lambda x: x['tpr'])
    best_tnr = max(all_results, key=lambda x: x['tnr'])
    
    print(f"🏆 BEST ACCURACY: {best_accuracy['model']}+{best_accuracy['loss']} = {best_accuracy['accuracy']:.4f}")
    print(f"🎯 BEST F1-SCORE: {best_f1['model']}+{best_f1['loss']} = {best_f1['f1']:.4f}")
    print(f"✅ BEST TPR: {best_tpr['model']}+{best_tpr['loss']} = {best_tpr['tpr']:.4f}")
    print(f"❌ BEST TNR: {best_tnr['model']}+{best_tnr['loss']} = {best_tnr['tnr']:.4f}")
    
    # Print confusion matrix for best accuracy model
    print(f"\n📊 CONFUSION MATRIX (Best Accuracy Model: {best_accuracy['model']}+{best_accuracy['loss']}):")
    print(f"Threshold: {best_accuracy['threshold']:.4f}")
    print(f"Accuracy: {best_accuracy['accuracy']:.4f}")
    print(f"Precision: {best_accuracy['precision']:.4f}")
    print(f"Recall: {best_accuracy['recall']:.4f}")
    print(f"F1-Score: {best_accuracy['f1']:.4f}")
    print(f"TPR: {best_accuracy['tpr']:.4f}")
    print(f"TNR: {best_accuracy['tnr']:.4f}")
    
    # Calculate and print balanced accuracy
    balanced_acc = (best_accuracy['tpr'] + best_accuracy['tnr']) / 2
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Print model comparison
    print(f"\n📈 MODEL COMPARISON (Train/Test Split):")
    print(f"{'Model+Loss':<25} {'Accuracy':<10} {'F1':<8} {'TPR':<8} {'TNR':<8}")
    print("-" * 60)
    for result in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['model']}+{result['loss']:<25} {result['accuracy']:<10.4f} {result['f1']:<8.4f} {result['tpr']:<8.4f} {result['tnr']:<8.4f}")
    
    # Analyze results
    print(f"\nPERFORMANCE ANALYSIS:")
    accuracies = [r['accuracy'] for r in all_results]
    f1_scores = [r['f1'] for r in all_results]
    thresholds = [r['threshold'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    
    print(f"Accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}")
    print(f"F1 score range: {min(f1_scores):.3f} - {max(f1_scores):.3f}")
    print(f"Threshold range: {min(thresholds):.3f} - {max(thresholds):.3f}")
    print(f"Precision range: {min(precisions):.3f} - {max(precisions):.3f}")
    print(f"Recall range: {min(recalls):.3f} - {max(recalls):.3f}")
    
    # Check for suspiciously high performance
    if max(accuracies) > 0.95:
        print("⚠ WARNING: Very high accuracy detected (>0.95) - results may be unreliable!")
        print("   Possible causes:")
        print("   - Data leakage")
        print("   - Perfect feature separation")
        print("   - Overly simple problem")
        print("   - Insufficient test set size")
    
    if max(f1_scores) > 0.95:
        print("⚠ WARNING: Very high F1 score detected (>0.95) - results may be unreliable!")
    
    # Check for threshold issues
    if min(thresholds) < 0.2:
        print("⚠ WARNING: Very low optimal threshold detected (<0.2) - possible calibration issues!")
        print("   This suggests:")
        print("   - Models are predicting very low probabilities")
        print("   - Poor model calibration")
        print("   - Threshold optimization may be unreliable")
    
    if max(thresholds) > 0.8:
        print("⚠ WARNING: Very high optimal threshold detected (>0.8) - possible calibration issues!")
    
    # Check for precision/recall imbalance
    precision_recall_gaps = [abs(r['precision'] - r['recall']) for r in all_results]
    if max(precision_recall_gaps) > 0.3:
        print("⚠ WARNING: Large precision-recall gaps detected - possible class imbalance issues!")
        print("   This suggests the model is biased toward one class")
    
    # Check for consistency across models
    acc_std = np.std(accuracies)
    if acc_std < 0.01:
        print("⚠ WARNING: Very low variance in accuracy across models - suspicious!")
    
    # Check for model type differences
    rf_acc = [r['accuracy'] for r in all_results if r['model'] == 'RF']
    lr_acc = [r['accuracy'] for r in all_results if r['model'] in ['LR', 'WLR']]
    
    if rf_acc and lr_acc:
        rf_avg = np.mean(rf_acc)
        lr_avg = np.mean(lr_acc)
        if abs(rf_avg - lr_avg) > 0.2:
            print("⚠ WARNING: Large performance gap between Random Forest and Linear models!")
            print(f"   RF average: {rf_avg:.3f}, Linear average: {lr_avg:.3f}")
            print("   This suggests data quality issues or overfitting")
    
    return all_results

# =========================
# Best Tversky Selection
# =========================
def evaluate_best_tversky(X, y, context="", optimize_for='f1'):
    """
    Find the best Tversky configuration with flexible metric optimization.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    context : str, default=""
        Context string for output
    optimize_for : str, default='f1'
        Metric to optimize threshold for: 'tpr', 'tnr', 'f1', 'balanced_accuracy'
    """
    best_metric_value, best_combo = -1, None
    for m in model_names:
        for name, lt, lp in loss_cfgs:
            if lt != 'tversky':
                continue
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
            sw = np.ones(len(y_tr)) if m == 'WLR' else None
            clf = train_funcs[m](X_tr, y_tr, sw, lt, lp)
            
            # Get probability predictions
            y_proba = clf.predict_proba(X_te)[:, 1]
            
            # Optimize threshold for specified metric
            from .custom_losses import optimize_threshold_for_metric
            optimal_threshold, best_metric_value_current = optimize_threshold_for_metric(y_te, y_proba, metric=optimize_for, min_accuracy=0.5)
            
            # Use optimal threshold for predictions
            y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
            
            # Calculate the metric we're optimizing for
            if optimize_for == 'f1':
                metric_value = f1_score(y_te, y_pred_optimal)
            elif optimize_for == 'tpr':
                metric_value = recall_score(y_te, y_pred_optimal)
            elif optimize_for == 'tnr':
                tn, fp, fn, tp = confusion_matrix(y_te, y_pred_optimal).ravel()
                metric_value = tn / (tn + fp) if (tn + fp) > 0 else 0
            elif optimize_for == 'balanced_accuracy':
                metric_value = balanced_accuracy_score(y_te, y_pred_optimal)
            else:
                metric_value = best_metric_value_current
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_combo = f"{m}+{name}"
    
    print(f"Best Tversky for {context} (optimizing for {optimize_for.upper()}): {best_combo} with {optimize_for.upper()}={best_metric_value:.3f}")

# =========================
# Data Quality Checks
# =========================
def check_data_quality(X, y, species_name=""):
    """
    Check for potential data quality issues that could lead to artificially high accuracy.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    species_name : str
        Species name for reporting
    """
    print(f"\nDATA QUALITY CHECK for {species_name}:")
    print("-" * 50)
    
    # Check 1: Dataset size
    print(f"1. Dataset size: {len(X)} samples")
    if len(X) < 50:
        print("   ⚠ WARNING: Very small dataset - results may not be reliable")
    
    # Check 2: Class balance
    n_presence = np.sum(y == 1)
    n_absence = np.sum(y == 0)
    balance_ratio = min(n_presence, n_absence) / max(n_presence, n_absence)
    
    print(f"2. Class balance: {n_presence} presence, {n_absence} absence (ratio: {balance_ratio:.3f})")
    if balance_ratio > 0.9:
        print("   ⚠ WARNING: Perfectly balanced dataset - may indicate artificial balancing")
    
    # Check 3: Feature separability
    print(f"3. Feature separability check:")
    presence_features = X[y == 1]
    absence_features = X[y == 0]
    
    separable_features = 0
    for i in range(min(10, X.shape[1])):  # Check first 10 features
        pres_mean = presence_features[:, i].mean()
        abs_mean = absence_features[:, i].mean()
        pres_std = presence_features[:, i].std()
        abs_std = absence_features[:, i].std()
        
        # Check if means are very different relative to standard deviations
        if abs(pres_mean - abs_mean) > 3 * (pres_std + abs_std):
            separable_features += 1
            print(f"   ⚠ Feature {i}: Very strong separation (mean diff: {abs(pres_mean - abs_mean):.3f})")
    
    if separable_features > 0:
        print(f"   ⚠ WARNING: {separable_features} features show very strong separation!")
    
    return separable_features == 0




# =========================
# Main routines
# =========================
def calculate_feature_based_reliability(absence_point_features, presence_features_df, threshold=0.2, power_transform=None):
    """
    Calculate reliability score for an absence point based on feature similarity to presence points.
    Uses the same Gaussian kernel approach as the original pseudo-absence generator.
    
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
    
    # Convert distances to similarities using Gaussian kernel (same as original)
    # The divisor (2 * number of features) normalizes for the feature space dimensionality
    similarities = np.exp(-distances**2 / (2 * presence_features_df.shape[1]))
    
    # Calculate mean similarity across all presence points
    mean_similarity = np.nanmean(similarities)
    
    # Reliability is inverse of similarity: less similar = more reliable as pseudo-absence
    reliability = 1 - mean_similarity
    
    # Apply power transformation using threshold value (or specified power_transform)
    if power_transform is None:
        power_transform = threshold  # Use threshold as power transform
    reliability = reliability ** power_transform
    
    return reliability


def comprehensive_species_with_precomputed_features(species, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', reliability_threshold=0.07, bias_correction=False):
    print(f"Starting species modeling: {species}")
    filtered_absence_path = f"data/filtered_presence_points_{species}.csv"
    if os.path.exists(filtered_absence_path):
        print(f"Using filtered absence points from: {filtered_absence_path}")
        features_df = pd.read_csv(features_csv_path, low_memory=False)
        all_absence_df = pd.read_csv(filtered_absence_path, low_memory=False)
    else:
        # Load the pre-computed features CSV for absence points
        try:
            features_df = pd.read_csv(features_csv_path, low_memory=False)
        except FileNotFoundError:
            print(f"Error: Features CSV not found at {features_csv_path}")
            return None, None, None, None
        all_absence_df = features_df.copy()
    
    # Load presence points from all_presence_point.csv
    try:
        presence_df = pd.read_csv('data/testing_SDM/all_presence_point.csv', low_memory=False)
    except FileNotFoundError:
        print(f"Error: all_presence_point.csv not found")
        return None, None, None, None
    
    # Get presence points for the target species from all_presence_point.csv
    pres = presence_df[presence_df['species'] == species].copy()
    if len(pres) == 0:
        print(f"No presence points found for species: {species}")
        return None, None, None, None
    
    print(f"Found {len(pres)} presence points for {species}")
    
    # Get the order of the target species
    target_order = pres['order'].iloc[0]
    
    # Remove any points that are the same as presence points
    presence_coords = set(zip(pres['decimalLongitude'], pres['decimalLatitude']))
    initial_absence_count = len(all_absence_df)
    all_absence_df = all_absence_df[~all_absence_df.apply(
        lambda r: (r['decimalLongitude'], r['decimalLatitude']) in presence_coords, axis=1
    )]
    filtered_absence_count = len(all_absence_df)
    removed_duplicates = initial_absence_count - filtered_absence_count
    
    print(f"Found {len(all_absence_df)} potential absence points (all available)")
    print(f"🔍 ABSENCE POINT FILTERING: Removed {removed_duplicates} duplicate points that matched presence coordinates")
    print(f"   Initial absence points: {initial_absence_count}")
    print(f"   After filtering: {filtered_absence_count}")
    print(f"   Duplicates removed: {removed_duplicates}")
    
    # Define feature columns (exclude taxonomic and coordinate columns)
    exclude_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 
                   'decimalLongitude', 'decimalLatitude']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    # Check if presence features are already saved for this species
    species_name = species.replace(" ", "_")
    presence_features_file = f"data/presence_features_{species_name}.csv"
    
    if os.path.exists(presence_features_file):
        pres_with_features = pd.read_csv(presence_features_file, low_memory=False)
    else:
        # Extract features for presence points using Earth Engine
        print("Extracting features for presence points...")
        import ee
        ee.Initialize()
        fe = Feature_Extractor(ee)
        
        # Prepare presence coordinates for feature extraction
        pres_coords = pres[['decimalLongitude', 'decimalLatitude']].copy()
        pres_coords.columns = ['longitude', 'latitude']
        
        # Extract features for presence points
        pres_with_features = fe.add_features(pres_coords)
        
        # Save presence features for future use
        pres_with_features.to_csv(presence_features_file, index=False)
    
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
    
    # Caching for absence selection and reliability scores
    cache_file = f"outputs/absence_reliability_{species.replace(' ', '_')}.csv"
    if os.path.exists(cache_file):
        print(f"Loading absence selection and reliability scores from cache: {cache_file}")
        absence_selected = pd.read_csv(cache_file)
    else:
        # Calculate reliability scores for all potential absence points
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
    
    
    
    # Prepare final datasets
    X_presence = pres_clean[feature_cols].values.astype(float)
    X_absence = absence_selected[feature_cols].values.astype(float)
    
    # Combine datasets
    X = np.vstack([X_presence, X_absence])
    y = np.concatenate([np.ones(len(X_presence)), np.zeros(len(X_absence))])
    
    # Create sample weights: presence points get weight 1, absence points get reliability-based weights
    presence_weights = np.ones(len(X_presence))  # Presence points get weight 1
    absence_weights = absence_selected['reliability_score'].values  # Absence points get reliability weights
    
    # Normalize absence weights to [0,1] range
    if len(absence_weights) > 0:
        min_weight = np.min(absence_weights)
        max_weight = np.max(absence_weights)
        if max_weight != min_weight:
            absence_weights = (absence_weights - min_weight) / (max_weight - min_weight)
        else:
            absence_weights = np.ones(len(absence_weights))
    
    # Combine weights
    sample_weights = np.concatenate([presence_weights, absence_weights])
    
    print(f"Final dataset: {len(X)} samples ({np.sum(y)} presence, {len(y)-np.sum(y)} absence)")
    
    # Create interactive map visualization
    try:
        map_file = visualize_presence_absence_points(pres_clean, absence_selected, species if 'species' in locals() else genus)
        if map_file:
            print(f"✓ Interactive map created: {map_file}")
    except Exception as e:
        print(f"Warning: Could not create map visualization: {e}")
    
    # After preparing pres_clean and absence_selected, save feature distribution plots
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
        plt.savefig(os.path.join(plot_dir, f'{col}_hist_{species if "species" in locals() else genus}.png'))
        plt.close()
    print(f'Feature distribution plots saved to {plot_dir}')
    
    # Print accuracy table
    # Create full DataFrame with features and lat/lon for per-ecoregion analysis
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
    evaluate_summary(X, y, all_df=all_df, optimize_for=optimize_for, species_name=species)
    
    # Train and save the best model
    best_model = None
    best_score = -1
    best_combo = ""
    for model_name in model_names:
        for loss_name, loss_type, loss_params in loss_cfgs:
            model = train_funcs[model_name](X, y, sample_weights, loss_type, loss_params)
            y_pred = model.predict(X)
            score = f1_score(y, y_pred)
            if score > best_score:
                best_score = score
                best_model = model
                best_combo = f"{model_name}_{loss_name}"
    model_path = f"outputs/best_model_{species.replace(' ', '_')}.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
  

    # --- Bias Correction (Optional) ---
    if bias_correction:
        # Assign ecoregion to each point (presence and absence)
        
        from shapely.wkt import loads as load_wkt
        from shapely.geometry import Point
        eco_dir = "data/eco_regions_polygon"
        eco_polygons = {}
        for fname in os.listdir(eco_dir):
            if fname.endswith('.wkt'):
                eco_name = fname.replace('.wkt', '')
                with open(os.path.join(eco_dir, fname), 'r') as f:
                    eco_polygons[eco_name] = load_wkt(f.read().strip())
        def assign_ecoregion(row):
            pt = Point(row['longitude'], row['latitude'])
            for eco, poly in eco_polygons.items():
                if poly.contains(pt):
                    return eco
            return None
        # Standardize coordinate columns
        pres_clean_bc = pres_clean.copy()
        absence_selected_bc = absence_selected.copy()
        if 'longitude' not in pres_clean_bc.columns and 'decimalLongitude' in pres_clean_bc.columns:
            pres_clean_bc = pres_clean_bc.rename(columns={'decimalLongitude': 'longitude', 'decimalLatitude': 'latitude'})
        if 'longitude' not in absence_selected_bc.columns and 'decimalLongitude' in absence_selected_bc.columns:
            absence_selected_bc = absence_selected_bc.rename(columns={'decimalLongitude': 'longitude', 'decimalLatitude': 'latitude'})
        pres_clean_bc['ecoregion'] = pres_clean_bc.apply(assign_ecoregion, axis=1)
        absence_selected_bc['ecoregion'] = absence_selected_bc.apply(assign_ecoregion, axis=1)
        all_ecoregions = pd.concat([pres_clean_bc['ecoregion'], absence_selected_bc['ecoregion']])
        eco_counts = all_ecoregions.value_counts().to_dict()
        # Compute raw weights
        eco_weights_raw = {eco: 1/(c+1) for eco, c in eco_counts.items()}
        w_min = min(eco_weights_raw.values())
        w_max = max(eco_weights_raw.values())
        # Normalize to [0.5, 1.5]
        eco_weights = {eco: 0.5 + (w_raw - w_min)/(w_max - w_min) if w_max > w_min else 1.0 for eco, w_raw in eco_weights_raw.items()}
        # Assign weights
        presence_weights_bc = pres_clean_bc['ecoregion'].map(eco_weights).fillna(1.0).values
        absence_weights_bc = absence_selected_bc['ecoregion'].map(eco_weights).fillna(1.0).values
        sample_weights = np.concatenate([presence_weights_bc, absence_weights_bc])

    return X, y, sample_weights, absence_selected, pres_clean, model_path


def comprehensive_genus_with_precomputed_features(genus, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', reliability_threshold=0.07, bias_correction=False):
    print(f"Starting genus modeling: {genus}")
    filtered_absence_path = f"data/filtered_presence_points_{genus}.csv"
    if os.path.exists(filtered_absence_path):
        print(f"Using filtered absence points from: {filtered_absence_path}")
        features_df = pd.read_csv(features_csv_path, low_memory=False)
        all_absence_df = pd.read_csv(filtered_absence_path, low_memory=False)
    else:
        # Load the pre-computed features CSV for absence points
        try:
            features_df = pd.read_csv(features_csv_path, low_memory=False)
        except FileNotFoundError:
            print(f"Error: Features CSV not found at {features_csv_path}")
            return None, None, None, None
        all_absence_df = features_df.copy()
    
    # Load presence points from all_presence_point.csv
    try:
        presence_df = pd.read_csv('data/testing_SDM/all_presence_point.csv', low_memory=False)
    except FileNotFoundError:
        print(f"Error: all_presence_point.csv not found")
        return None, None, None, None
    
    # Get presence points for the target genus from all_presence_point.csv
    pres = presence_df[presence_df['genus'] == genus].copy()
    if len(pres) == 0:
        print(f"No presence points found for genus: {genus}")
        return None, None, None, None
    
    print(f"Found {len(pres)} presence points for {genus}")
    
    # Get the order of the target genus
    target_order = pres['order'].iloc[0]
    
    # Remove any points that are the same as presence points
    presence_coords = set(zip(pres['decimalLongitude'], pres['decimalLatitude']))
    initial_absence_count = len(all_absence_df)
    all_absence_df = all_absence_df[~all_absence_df.apply(
        lambda r: (r['decimalLongitude'], r['decimalLatitude']) in presence_coords, axis=1
    )]
    filtered_absence_count = len(all_absence_df)
    removed_duplicates = initial_absence_count - filtered_absence_count
    
    print(f"Found {len(all_absence_df)} potential absence points (all available)")
    print(f"🔍 ABSENCE POINT FILTERING: Removed {removed_duplicates} duplicate points that matched presence coordinates")
    print(f"   Initial absence points: {initial_absence_count}")
    print(f"   After filtering: {filtered_absence_count}")
    print(f"   Duplicates removed: {removed_duplicates}")
    
    # Define feature columns (exclude taxonomic and coordinate columns)
    exclude_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 
                   'decimalLongitude', 'decimalLatitude']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    # Check if presence features are already saved for this genus
    presence_features_file = f"data/presence_features_{genus.replace(' ', '_')}.csv"
    
    if os.path.exists(presence_features_file):
        pres_with_features = pd.read_csv(presence_features_file, low_memory=False)
    else:
        # Extract features for presence points using Earth Engine
        print("Extracting features for presence points...")
        import ee
        ee.Initialize()
        fe = Feature_Extractor(ee)
        
        # Prepare presence coordinates for feature extraction
        pres_coords = pres[['decimalLongitude', 'decimalLatitude']].copy()
        pres_coords.columns = ['longitude', 'latitude']
        
        # Extract features for presence points
        pres_with_features = fe.add_features(pres_coords)
        
        # Save presence features for future use
        pres_with_features.to_csv(presence_features_file, index=False)
    
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
    
    # Caching for absence selection and reliability scores
    cache_file = f"outputs/absence_reliability_{genus.replace(' ', '_')}.csv"
    if os.path.exists(cache_file):
        print(f"Loading absence selection and reliability scores from cache: {cache_file}")
        absence_selected = pd.read_csv(cache_file)
    else:
        # Calculate reliability scores for all potential absence points
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
                overlap = np.minimum(hist_pres, hist_abs).sum() / 20  # Normalized overlap
                
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
    
    # Prepare final datasets
    X_presence = pres_clean[feature_cols].values.astype(float)
    X_absence = absence_selected[feature_cols].values.astype(float)
    
    # Combine datasets
    X = np.vstack([X_presence, X_absence])
    y = np.concatenate([np.ones(len(X_presence)), np.zeros(len(X_absence))])
    
    # Create sample weights: presence points get weight 1, absence points get reliability-based weights
    presence_weights = np.ones(len(X_presence))  # Presence points get weight 1
    absence_weights = absence_selected['reliability_score'].values  # Absence points get reliability weights
    
    # Normalize absence weights to [0,1] range
    if len(absence_weights) > 0:
        min_weight = np.min(absence_weights)
        max_weight = np.max(absence_weights)
        if max_weight != min_weight:
            absence_weights = (absence_weights - min_weight) / (max_weight - min_weight)
        else:
            absence_weights = np.ones(len(absence_weights))
    
    # Combine weights
    sample_weights = np.concatenate([presence_weights, absence_weights])
    
    print(f"Final dataset: {len(X)} samples ({np.sum(y)} presence, {len(y)-np.sum(y)} absence)")
    
    # Create interactive map visualization
    try:
        map_file = visualize_presence_absence_points(pres_clean, absence_selected, genus)
        if map_file:
            print(f"✓ Interactive map created: {map_file}")
    except Exception as e:
        print(f"Warning: Could not create map visualization: {e}")
    
    # After preparing pres_clean and absence_selected, save feature distribution plots
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
    
    # Print accuracy table
    # Create full DataFrame with features and lat/lon for per-ecoregion analysis
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
    evaluate_summary(X, y, all_df=all_df, optimize_for=optimize_for, genus_name=genus)
    
    # Train and save the best model
    best_model = None
    best_score = -1
    best_combo = ""
    for model_name in model_names:
        for loss_name, loss_type, loss_params in loss_cfgs:
            model = train_funcs[model_name](X, y, sample_weights, loss_type, loss_params)
            y_pred = model.predict(X)
            score = f1_score(y, y_pred)
            if score > best_score:
                best_score = score
                best_model = model
                best_combo = f"{model_name}_{loss_name}"
    model_path = f"outputs/best_model_{genus.replace(' ', '_')}.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
   

    # --- Bias Correction (Optional) ---
    if bias_correction:
        # Assign ecoregion to each point (presence and absence)
        
        from shapely.wkt import loads as load_wkt
        from shapely.geometry import Point
        eco_dir = "data/eco_regions_polygon"
        eco_polygons = {}
        for fname in os.listdir(eco_dir):
            if fname.endswith('.wkt'):
                eco_name = fname.replace('.wkt', '')
                with open(os.path.join(eco_dir, fname), 'r') as f:
                    eco_polygons[eco_name] = load_wkt(f.read().strip())
        def assign_ecoregion(row):
            pt = Point(row['longitude'], row['latitude'])
            for eco, poly in eco_polygons.items():
                if poly.contains(pt):
                    return eco
            return None
        # Standardize coordinate columns
        pres_clean_bc = pres_clean.copy()
        absence_selected_bc = absence_selected.copy()
        if 'longitude' not in pres_clean_bc.columns and 'decimalLongitude' in pres_clean_bc.columns:
            pres_clean_bc = pres_clean_bc.rename(columns={'decimalLongitude': 'longitude', 'decimalLatitude': 'latitude'})
        if 'longitude' not in absence_selected_bc.columns and 'decimalLongitude' in absence_selected_bc.columns:
            absence_selected_bc = absence_selected_bc.rename(columns={'decimalLongitude': 'longitude', 'decimalLatitude': 'latitude'})
        pres_clean_bc['ecoregion'] = pres_clean_bc.apply(assign_ecoregion, axis=1)
        absence_selected_bc['ecoregion'] = absence_selected_bc.apply(assign_ecoregion, axis=1)
        all_ecoregions = pd.concat([pres_clean_bc['ecoregion'], absence_selected_bc['ecoregion']])
        eco_counts = all_ecoregions.value_counts().to_dict()
        # Compute raw weights
        eco_weights_raw = {eco: 1/(c+1) for eco, c in eco_counts.items()}
        w_min = min(eco_weights_raw.values())
        w_max = max(eco_weights_raw.values())
        # Normalize to [0.5, 1.5]
        eco_weights = {eco: 0.5 + (w_raw - w_min)/(w_max - w_min) if w_max > w_min else 1.0 for eco, w_raw in eco_weights_raw.items()}
        # Assign weights
        presence_weights_bc = pres_clean_bc['ecoregion'].map(eco_weights).fillna(1.0).values
        absence_weights_bc = absence_selected_bc['ecoregion'].map(eco_weights).fillna(1.0).values
        sample_weights = np.concatenate([presence_weights_bc, absence_weights_bc])

    return X, y, sample_weights, absence_selected, pres_clean, model_path


# =========================
# Ecoregion Testing for Custom Loss Models
# =========================

def test_custom_model_on_all_ecoregions(trained_model, species_name, output_file=None, num_workers=16):
    """
    Test a trained custom loss model on all ecoregions.
    
    Args:
        trained_model: Trained custom loss model (CustomLossRandomForest, CustomLossLogisticRegression, etc.)
        species_name: Name of the species for output file naming
        output_file: Output file path (if None, auto-generates based on species name)
        num_workers: Number of parallel workers
    """
    import ee
    from .features_extractor import Feature_Extractor
    
    # Initialize Earth Engine and feature extractor
    try:
        ee.Initialize()
    except:
        print("Warning: Earth Engine already initialized or failed to initialize")
    
    fe = Feature_Extractor(ee)
    
    # Auto-generate output file name if not provided
    if output_file is None:
        model_type = type(trained_model).__name__
        output_file = f'data/{species_name}_{model_type}_ecoregion_results.csv'
    
    polygon_dir = 'data/eco_regions_polygon'
    
    # Write the header for the output file if the file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as out_file:
            out_file.write('Ecoregion,Average_Probability\n')
    
    # Get all .wkt files
    wkt_files = [f for f in os.listdir(polygon_dir) if f.endswith('.wkt')]
    total_files = len(wkt_files)
    
    print(f'Starting processing of {total_files} ecoregions with {num_workers} workers')
    print(f'Model: {type(trained_model).__name__}')
    print(f'Species: {species_name}')
    print(f'Output: {output_file}')
    
    # Create a queue of files to process
    file_queue = queue.Queue()
    for filename in wkt_files:
        file_queue.put(filename)
    
    # Create a lock for file writing
    file_lock = threading.Lock()
    
    # Create a shared counter for completed files
    completed = [0]
    
    def worker():
        while not file_queue.empty():
            try:
                # Get a file from the queue with a timeout
                try:
                    filename = file_queue.get(timeout=1)
                except queue.Empty:
                    break
                
                # Process the file
                ecoregion_name = os.path.splitext(filename)[0]
                try:
                    avg_probability = process_single_ecoregion_custom_model(
                        filename, polygon_dir, trained_model, fe
                    )
                except Exception as e:
                    print(f'Error processing {ecoregion_name}: {str(e)}')
                    avg_probability = 0.0
                
                # Write the result to the output file
                with file_lock:
                    with open(output_file, 'a') as out_file:
                        out_file.write(f'{ecoregion_name},{avg_probability}\n')
                    completed[0] += 1
                    print(f'Completed {completed[0]}/{total_files}: {ecoregion_name} (prob: {avg_probability:.4f})')
                
                # Mark the task as done
                file_queue.task_done()
            except Exception as e:
                print(f'Worker error: {str(e)}')
    
    # Start worker threads
    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Wait for all files to be processed or timeout
    timeout = 60 * 60  # 1 hour timeout
    start_time = time.time()
    while completed[0] < total_files and time.time() - start_time < timeout:
        time.sleep(1)
    
    # Check if all files were processed
    if completed[0] < total_files:
        print(f'WARNING: Only {completed[0]}/{total_files} files were processed before timeout')
        
        # Process any remaining files sequentially
        remaining_files = []
        while not file_queue.empty():
            try:
                remaining_files.append(file_queue.get(timeout=1))
            except queue.Empty:
                break
        
        print(f'Processing remaining {len(remaining_files)} files sequentially')
        for filename in remaining_files:
            ecoregion_name = os.path.splitext(filename)[0]
            try:
                avg_probability = process_single_ecoregion_custom_model(
                    filename, polygon_dir, trained_model, fe
                )
            except Exception as e:
                print(f'Error processing {ecoregion_name}: {str(e)}')
                avg_probability = 0.0
            
            with open(output_file, 'a') as out_file:
                out_file.write(f'{ecoregion_name},{avg_probability}\n')
            completed[0] += 1
            print(f'Completed {completed[0]}/{total_files}: {ecoregion_name} (prob: {avg_probability:.4f})')
    
    print(f'All ecoregions processed. Average probabilities saved to {output_file}')
    return output_file


def process_single_ecoregion_custom_model(filename, polygon_dir, trained_model, feature_extractor):
    """Process a single ecoregion file and return the average probability for custom loss models."""
    ecoregion_name = os.path.splitext(filename)[0]
    polygon_path = os.path.join(polygon_dir, filename)
    
    # Read the polygon WKT
    with open(polygon_path, 'r') as file:
        polygon_wkt = file.read().strip()
    
    # Generate test data for the current ecoregion
    test_points = utility.divide_polygon_to_grids(polygon_wkt, grid_size=1, points_per_cell=20)
    
    # Extract features for test points
    test_features = feature_extractor.add_features(test_points)
    
    # Get feature columns (exclude longitude/latitude)
    feature_cols = [col for col in test_features.columns if col not in ['longitude', 'latitude']]
    X_test = test_features[feature_cols].values.astype(float)
    
    # Remove NaN and infinite values from test set
    mask = np.isfinite(X_test).all(axis=1)
    X_test = X_test[mask]
    
    if X_test.shape[0] == 0:  # If no valid samples remain
        print(f'No valid samples for {ecoregion_name}. Setting average probability to 0.')
        avg_probability = 0
    else:
        # Make predictions using the trained custom loss model
        y_proba = trained_model.predict_proba(X_test)[:, 1]
        
        # Calculate the average probability
        avg_probability = y_proba.mean()
    
    return avg_probability


def comprehensive_species_modeling_with_ecoregion_testing(species, features_csv_path="data/presence_points_with_features.csv", 
                                                        optimize_for='tpr', test_ecoregions=True, bias_correction=False):
    """
    Comprehensive species modeling with custom loss training and optional ecoregion testing.
    
    Parameters:
    -----------
    species : str
        Species name to model
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to pre-computed features CSV
    optimize_for : str, default='tpr'
        Metric to optimize threshold for: 'tpr', 'tnr', 'f1', 'balanced_accuracy'
    test_ecoregions : bool, default=True
        Whether to test the best model on all ecoregions
    """
    print(f"COMPREHENSIVE SPECIES MODELING WITH ECOREGION TESTING: {species}")
    
    # First, run the comprehensive species modeling
    result = comprehensive_species_with_precomputed_features(species, features_csv_path, optimize_for, bias_correction=bias_correction)
    
    if result is None:
        print("Failed to prepare data for modeling. Skipping ecoregion testing.")
        return None
    
    X, y, sample_weights, absence_selected, pres_clean, model_path = result
    
    if not test_ecoregions:
        print("Ecoregion testing disabled. Skipping.")
        return None
    
    # Train best model for ecoregion testing
    print("\nTraining best model for ecoregion testing...")
    best_model = None
    best_score = -1
    
    for model_name in model_names:
        for loss_name, loss_type, loss_params in loss_cfgs:
            print(f"Training {model_name} with {loss_name} loss...")
            model = train_funcs[model_name](X, y, sample_weights, loss_type, loss_params)
            
            # Evaluate model
            y_pred = model.predict(X)
            score = f1_score(y, y_pred)
            
            if score > best_score:
                best_score = score
                best_model = model
                
    
    if best_model is None:
        print("Failed to train any model. Skipping ecoregion testing.")
        return None
    
    # Test on all ecoregions
    print("\nTesting model on all ecoregions...")
    output_file = test_custom_model_on_all_ecoregions(best_model, species)
    
    print(f"\nEcoregion testing completed! Results saved to: {output_file}")
    return best_model, output_file, model_path


def comprehensive_genus_modeling_with_ecoregion_testing(genus, features_csv_path="data/presence_points_with_features.csv", 
                                                      optimize_for='tpr', test_ecoregions=True, bias_correction=False):
    """
    Comprehensive genus modeling with custom loss training and optional ecoregion testing.
    
    Parameters:
    -----------
    genus : str
        Genus name to model
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to pre-computed features CSV
    optimize_for : str, default='tpr'
        Metric to optimize threshold for: 'tpr', 'tnr', 'f1', 'balanced_accuracy'
    test_ecoregions : bool, default=True
        Whether to test the best model on all ecoregions
    """
    print(f"COMPREHENSIVE GENUS MODELING WITH ECOREGION TESTING: {genus}")
    
    # First, run the comprehensive genus modeling
    result = comprehensive_genus_with_precomputed_features(genus, features_csv_path, optimize_for, bias_correction=bias_correction)
    
    if result is None:
        print("Failed to prepare data for modeling. Skipping ecoregion testing.")
        return None
    
    X, y, sample_weights, absence_selected, pres_clean, model_path = result
    
    if not test_ecoregions:
        print("Ecoregion testing disabled. Skipping.")
        return None
    
    # Train best model for ecoregion testing
    print("\nTraining best model for ecoregion testing...")
    best_model = None
    best_score = -1
    
    for model_name in model_names:
        for loss_name, loss_type, loss_params in loss_cfgs:
            print(f"Training {model_name} with {loss_name} loss...")
            model = train_funcs[model_name](X, y, sample_weights, loss_type, loss_params)
            
            # Evaluate model
            y_pred = model.predict(X)
            score = f1_score(y, y_pred)
            
            if score > best_score:
                best_score = score
                best_model = model
                
    
    if best_model is None:
        print("Failed to train any model. Skipping ecoregion testing.")
        return None
    
    # Test on all ecoregions
    print("\nTesting model on all ecoregions...")
    output_file = test_custom_model_on_all_ecoregions(best_model, genus)
    
    print(f"\nEcoregion testing completed! Results saved to: {output_file}")
    return best_model, output_file, model_path


# =========================
# Rank Correlation Analysis
# =========================

def compute_rank_correlation(prob_file, similarity_file, similarity_col, output_csv):
    """
    Compute Spearman rank correlation between similarity and average probability for ecoregions.
    
    Args:
        prob_file: CSV file with columns ['Ecoregion', 'Average_Probability']
        similarity_file: Tab-separated text file with ecoregion similarity matrix
        similarity_col: Name of the similarity column to use (e.g., 'Euclidean_Similarity')
        output_csv: Path to save merged results with correlation value
    
    Returns:
        corr: Spearman rank correlation coefficient
        pval: p-value
    """
    from scipy.stats import spearmanr
    
    # Read probability file (CSV format)
    prob_df = pd.read_csv(prob_file)
    
    # Read similarity file (tab-separated text format)
    sim_df = pd.read_csv(similarity_file, sep='\t', index_col=0)
    
    # Clean ecoregion names by removing .wkt extension
    prob_df['Ecoregion_clean'] = prob_df['Ecoregion'].str.replace('.wkt', '')
    sim_df.index = sim_df.index.str.replace('.wkt', '')
    
    # Convert similarity matrix to long format for easier merging
    sim_long = sim_df.reset_index().melt(
        id_vars='index', 
        var_name='Ecoregion_clean', 
        value_name=similarity_col
    )
    sim_long['Ecoregion_clean'] = sim_long['Ecoregion_clean'].str.replace('.wkt', '')
    
    # Merge probability and similarity data
    merged = pd.merge(prob_df, sim_long, on='Ecoregion_clean', how='inner')
    
    if len(merged) == 0:
        raise ValueError("No matching ecoregions found between probability and similarity files")
    
    # Calculate correlation
    corr, pval = spearmanr(merged[similarity_col], merged['Average_Probability'])
    print(f"Spearman rank correlation: {corr:.4f} (p={pval:.4g})")
    print(f"Number of ecoregions analyzed: {len(merged)}")
    
    # Save results
    merged['Rank_Correlation'] = corr
    merged.to_csv(output_csv, index=False)
    
    return corr, pval


def analyze_ecoregion_correlations(species_name, model_type="CustomLossRandomForest"):
    """
    Analyze rank correlations between ecoregion probabilities and similarities.
    
    Args:
        species_name: Name of the species
        model_type: Type of model used (for file naming)
    
    Returns:
        dict: Dictionary with correlation results
    """
    print(f"\n{'='*60}")
    print(f"RANK CORRELATION ANALYSIS FOR {species_name}")
    print(f"{'='*60}")
    
    # Define file paths
    prob_file = f'data/{species_name}_{model_type}_ecoregion_results.csv'
    
    # Check if probability file exists
    if not os.path.exists(prob_file):
        print(f"❌ Probability file not found: {prob_file}")
        print("Please run ecoregion testing first.")
        return None
    
    # Similarity files to test
    similarity_files = {
        'Euclidean': 'data/euclidean_similarity_matrix.txt',
        'Cosine': 'data/cosine_similarity_matrix.txt'
    }
    
    results = {}
    
    for sim_type, sim_file in similarity_files.items():
        if os.path.exists(sim_file):
            print(f"\nAnalyzing {sim_type} similarity correlation...")
            
            # Create output file name
            output_csv = f'data/{species_name}_{model_type}_{sim_type.lower()}_correlation.csv'
            
            try:
                corr, pval = compute_rank_correlation(
                    prob_file=prob_file,
                    similarity_file=sim_file,
                    similarity_col=f'{sim_type}_Similarity',
                    output_csv=output_csv
                )
                
                results[sim_type] = {
                    'correlation': corr,
                    'p_value': pval,
                    'output_file': output_csv
                }
                
                print(f"✅ {sim_type} correlation analysis completed")
                print(f"   Correlation: {corr:.4f}")
                print(f"   P-value: {pval:.4g}")
                print(f"   Results saved to: {output_csv}")
                
            except Exception as e:
                print(f"❌ Error in {sim_type} correlation analysis: {str(e)}")
                results[sim_type] = {'error': str(e)}
        else:
            print(f"⚠️  Similarity file not found: {sim_file}")
            results[sim_type] = {'error': 'File not found'}
    
    return results


def comprehensive_analysis_with_correlations(species, features_csv_path="data/presence_points_with_features.csv", 
                                           optimize_for='tpr', test_ecoregions=True, analyze_correlations=True):
    """
    Comprehensive species modeling with ecoregion testing and rank correlation analysis.
    
    Parameters:
    -----------
    species : str
        Species name to model
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to pre-computed features CSV
    optimize_for : str, default='tpr'
        Metric to optimize threshold for: 'tpr', 'tnr', 'f1', 'balanced_accuracy'
    test_ecoregions : bool, default=True
        Whether to test the best model on all ecoregions
    analyze_correlations : bool, default=True
        Whether to analyze rank correlations between ecoregion probabilities and similarities
    """
    print(f"COMPREHENSIVE SPECIES ANALYSIS WITH CORRELATIONS: {species}")
    
    # Step 1: Run comprehensive species modeling with ecoregion testing
    result = comprehensive_species_modeling_with_ecoregion_testing(
        species=species,
        features_csv_path=features_csv_path,
        optimize_for=optimize_for,
        test_ecoregions=test_ecoregions
    )
    
    if result is None:
        print("Failed to complete species modeling. Skipping correlation analysis.")
        return None
    
    best_model, output_file, model_path = result
    
    # Step 2: Analyze correlations if requested
    correlation_results = None
    if analyze_correlations:
        print("\nAnalyzing rank correlations...")
        model_type = type(best_model).__name__
        correlation_results = analyze_ecoregion_correlations(species, model_type)
        
        if correlation_results:
            print("Correlation analysis completed!")
            # Check if results contain correlation values
            if 'Euclidean' in correlation_results and 'correlation' in correlation_results['Euclidean']:
                print(f"Euclidean correlation: {correlation_results['Euclidean']['correlation']:.3f}")
            if 'Cosine' in correlation_results and 'correlation' in correlation_results['Cosine']:
                print(f"Cosine correlation: {correlation_results['Cosine']['correlation']:.3f}")
        else:
            print("Correlation analysis failed or no results found.")
    
    # Return comprehensive results
    return {
        'species': species,
        'best_model': best_model,
        'ecoregion_results': output_file,
        'correlation_results': correlation_results,
        'model_path': model_path
    }


def comprehensive_genus_analysis_with_correlations(genus, features_csv_path="data/presence_points_with_features.csv", 
                                                 optimize_for='tpr', test_ecoregions=True, analyze_correlations=True):
    """
    Comprehensive genus modeling with ecoregion testing and rank correlation analysis.
    
    Parameters:
    -----------
    genus : str
        Genus name to model
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to pre-computed features CSV
    optimize_for : str, default='tpr'
        Metric to optimize threshold for: 'tpr', 'tnr', 'f1', 'balanced_accuracy'
    test_ecoregions : bool, default=True
        Whether to test the best model on all ecoregions
    analyze_correlations : bool, default=True
        Whether to analyze rank correlations between ecoregion probabilities and similarities
    """
    print(f"COMPREHENSIVE GENUS ANALYSIS WITH CORRELATIONS: {genus}")
    
    # Step 1: Run comprehensive genus modeling with ecoregion testing
    result = comprehensive_genus_modeling_with_ecoregion_testing(
        genus=genus,
        features_csv_path=features_csv_path,
        optimize_for=optimize_for,
        test_ecoregions=test_ecoregions
    )
    
    if result is None:
        print("Failed to complete genus modeling. Skipping correlation analysis.")
        return None
    
    best_model, output_file, model_path = result
    
    # Step 2: Analyze correlations if requested
    correlation_results = None
    if analyze_correlations:
        print("\nAnalyzing rank correlations...")
        model_type = type(best_model).__name__
        correlation_results = analyze_ecoregion_correlations(genus, model_type)
        
        if correlation_results:
            print("Correlation analysis completed!")
            # Check if results contain correlation values
            if 'Euclidean' in correlation_results and 'correlation' in correlation_results['Euclidean']:
                print(f"Euclidean correlation: {correlation_results['Euclidean']['correlation']:.3f}")
            if 'Cosine' in correlation_results and 'correlation' in correlation_results['Cosine']:
                print(f"Cosine correlation: {correlation_results['Cosine']['correlation']:.3f}")
        else:
            print("Correlation analysis failed or no results found.")
    
    # Return comprehensive results
    return {
        'genus': genus,
        'best_model': best_model,
        'ecoregion_results': output_file,
        'correlation_results': correlation_results,
        'model_path': model_path
    }


def batch_correlation_analysis(species_list=None, genus_list=None):
    """
    Perform batch correlation analysis for multiple species and genera.
    
    Args:
        species_list: List of species names to analyze
        genus_list: List of genus names to analyze
    """
    print(f"\n{'='*60}")
    print("BATCH CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    all_results = {}
    
    # Analyze species
    if species_list:
        print(f"\nAnalyzing {len(species_list)} species...")
        for species in species_list:
            print(f"\n{'='*40}")
            print(f"Processing species: {species}")
            print(f"{'='*40}")
            
            try:
                result = comprehensive_analysis_with_correlations(
                    species=species,
                    optimize_for='f1',
                    test_ecoregions=True,
                    analyze_correlations=True
                )
                all_results[f"species_{species}"] = result
            except Exception as e:
                print(f"❌ Error processing species {species}: {str(e)}")
                all_results[f"species_{species}"] = {'error': str(e)}
    
    # Analyze genera
    if genus_list:
        print(f"\nAnalyzing {len(genus_list)} genera...")
        for genus in genus_list:
            print(f"\n{'='*40}")
            print(f"Processing genus: {genus}")
            print(f"{'='*40}")
            
            try:
                result = comprehensive_genus_analysis_with_correlations(
                    genus=genus,
                    optimize_for='tpr',
                    test_ecoregions=True,
                    analyze_correlations=True
                )
                all_results[f"genus_{genus}"] = result
            except Exception as e:
                print(f"❌ Error processing genus {genus}: {str(e)}")
                all_results[f"genus_{genus}"] = {'error': str(e)}
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for name, result in all_results.items():
        if 'error' in result:
            print(f"❌ {name}: {result['error']}")
        else:
            print(f"✅ {name}: Completed successfully")
            if result['correlation_results']:
                for sim_type, corr_result in result['correlation_results'].items():
                    if 'error' not in corr_result:
                        print(f"   {sim_type}: r={corr_result['correlation']:.4f}")
    
    return all_results


# =========================
# Feature Importance Analysis
# =========================

def perform_feature_importance_for_all_species(species_list, features_csv_path="data/presence_points_with_features.csv"):
    """
    For each species, find the best custom loss model, retrain, and perform feature importance analysis.
    Prints and visualizes feature importance for each species using SHAP.
    
    Parameters:
    -----------
    species_list : list
        List of species names to analyze
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to pre-computed features CSV
    """
    # Import required libraries
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        print("SHAP or matplotlib library not found. Installing SHAP...")
        import subprocess
        subprocess.check_call(["pip", "install", "shap"])
        import shap
        import matplotlib.pyplot as plt
    
    for species_name in species_list:
        print(f"\n{'='*80}")
        print(f"Feature Importance Analysis for {species_name}")
        print(f"{'='*80}")
        
        # 1. Prepare data using the comprehensive species modeling function
        result = comprehensive_species_with_precomputed_features(
            species=species_name, 
            features_csv_path=features_csv_path,
            optimize_for='f1'
        )
        
        if result is None:
            print(f"  Skipping {species_name}: Failed to prepare data.")
            continue
        
        X, y, sample_weights, absence_selected, pres_clean, model_path = result
        
        # 2. Find best custom loss model by testing all combinations
        print(f"  Testing all custom loss model combinations...")
        best_model = None
        best_score = -1
        best_combo = ""
        
        for model_name in model_names:
            for loss_name, loss_type, loss_params in loss_cfgs:
                print(f"    Testing {model_name} with {loss_name} loss...")
                model = train_funcs[model_name](X, y, sample_weights, loss_type, loss_params)
                
                # Evaluate model
                y_pred = model.predict(X)
                score = f1_score(y, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_combo = f"{model_name}_{loss_name}"
                    print(f"    New best: {best_combo} (F1: {score:.4f})")
        
        if best_model is None:
            print(f"  Failed to train any model for {species_name}, skipping.")
            continue
        
        
        
        # 3. SHAP Analysis
        print(f"  Generating SHAP summary plots...")
        
        # Create species-specific output directory
        species_safe_name = species_name.replace(' ', '_').lower()
        species_output_dir = f"outputs/testing_SDM_out/{species_safe_name}"
        os.makedirs(species_output_dir, exist_ok=True)
        
        # Get feature names
        feature_cols = [col for col in pd.read_csv(features_csv_path).columns 
                       if col not in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 
                                     'decimalLongitude', 'decimalLatitude']]
        
        try:
            # Create SHAP explainer based on model type
            if isinstance(best_model, CustomLossRandomForest):
                # For Random Forest, use TreeExplainer
                explainer = shap.TreeExplainer(best_model.trees[0])  # Use first tree as representative
            else:
                # For logistic regression and other models, use KernelExplainer
                explainer = shap.KernelExplainer(best_model.predict_proba, X[:100])  # Use small sample for background
            
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
            print(f"    SHAP bar plot saved to: {shap_bar_path}")
            
            # Calculate and save feature importance scores
            feature_importance = np.abs(shap_values).mean(0)
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            importance_path = os.path.join(species_output_dir, "shap_importance_scores.csv")
            importance_df.to_csv(importance_path, index=False)
            print(f"    Feature importance scores saved to: {importance_path}")
            
            # Print top 10 features
            print(f"    Top 10 most important features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"      {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
            
        except Exception as e:
            print(f"    Error in SHAP analysis: {str(e)}")
            print(f"    Skipping SHAP analysis for {species_name}")
        
        print(f"  Feature importance analysis completed for {species_name}")


def perform_feature_importance_for_all_genera(genus_list, features_csv_path="data/presence_points_with_features.csv"):
    """
    For each genus, find the best custom loss model, retrain, and perform feature importance analysis.
    Prints and visualizes feature importance for each genus using SHAP.
    
    Parameters:
    -----------
    genus_list : list
        List of genus names to analyze
    features_csv_path : str, default="data/presence_points_with_features.csv"
        Path to pre-computed features CSV
    """
    # Import required libraries
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        print("SHAP or matplotlib library not found. Installing SHAP...")
        import subprocess
        subprocess.check_call(["pip", "install", "shap"])
        import shap
        import matplotlib.pyplot as plt
    
    for genus_name in genus_list:
        print(f"\n{'='*80}")
        print(f"Feature Importance Analysis for {genus_name}")
        print(f"{'='*80}")
        
        # 1. Prepare data using the comprehensive genus modeling function
        result = comprehensive_genus_with_precomputed_features(
            genus=genus_name, 
            features_csv_path=features_csv_path,
            optimize_for='f1'
        )
        
        if result is None:
            print(f"  Skipping {genus_name}: Failed to prepare data.")
            continue
        
        X, y, sample_weights, absence_selected, pres_clean, model_path = result
        
        # 2. Find best custom loss model by testing all combinations
        print(f"  Testing all custom loss model combinations...")
        best_model = None
        best_score = -1
        best_combo = ""
        
        for model_name in model_names:
            for loss_name, loss_type, loss_params in loss_cfgs:
                print(f"    Testing {model_name} with {loss_name} loss...")
                model = train_funcs[model_name](X, y, sample_weights, loss_type, loss_params)
                
                # Evaluate model
                y_pred = model.predict(X)
                score = f1_score(y, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_combo = f"{model_name}_{loss_name}"
                    print(f"    New best: {best_combo} (F1: {score:.4f})")
        
        if best_model is None:
            print(f"  Failed to train any model for {genus_name}, skipping.")
            continue
        
        
        
        # 3. SHAP Analysis
        print(f"  Generating SHAP summary plots...")
        
        # Create genus-specific output directory
        genus_safe_name = genus_name.replace(' ', '_').lower()
        genus_output_dir = f"outputs/testing_SDM_out/{genus_safe_name}"
        os.makedirs(genus_output_dir, exist_ok=True)
        
        # Get feature names
        feature_cols = [col for col in pd.read_csv(features_csv_path).columns 
                       if col not in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 
                                     'decimalLongitude', 'decimalLatitude']]
        
        try:
            # Create SHAP explainer based on model type
            if isinstance(best_model, CustomLossRandomForest):
                # For Random Forest, use TreeExplainer
                explainer = shap.TreeExplainer(best_model.trees[0])  # Use first tree as representative
            else:
                # For logistic regression and other models, use KernelExplainer
                explainer = shap.KernelExplainer(best_model.predict_proba, X[:100])  # Use small sample for background
            
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
            plt.title(f'SHAP Summary Plot - {genus_name}', fontsize=16, fontweight='bold')
            
            # Save SHAP summary plot in genus folder
            shap_summary_path = os.path.join(genus_output_dir, "shap_summary.png")
            plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    SHAP summary plot saved to: {shap_summary_path}")
            
            # Create SHAP bar plot (feature importance)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {genus_name}', fontsize=16, fontweight='bold')
            
            # Save SHAP bar plot in genus folder
            shap_bar_path = os.path.join(genus_output_dir, "shap_importance.png")
            plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    SHAP bar plot saved to: {shap_bar_path}")
            
            # Calculate and save feature importance scores
            feature_importance = np.abs(shap_values).mean(0)
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            importance_path = os.path.join(genus_output_dir, "shap_importance_scores.csv")
            importance_df.to_csv(importance_path, index=False)
            print(f"    Feature importance scores saved to: {importance_path}")
            
            # Print top 10 features
            print(f"    Top 10 most important features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"      {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
            
        except Exception as e:
            print(f"    Error in SHAP analysis: {str(e)}")
            print(f"    Skipping SHAP analysis for {genus_name}")
        
        print(f"  Feature importance analysis completed for {genus_name}")


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



def all_ecoregion_level_species_with_precomputed_features(species, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', bias_correction=False):
    """
    Restrict modeling to all ecoregions where presence points are found for the given species.
    Uses WKT files to determine which ecoregions have presence points.
    """
    
    from shapely.wkt import loads as load_wkt
    from shapely.geometry import Point
    import pandas as pd
    import numpy as np

    print(f"Starting all-ecoregion-level species modeling: {species}")
    try:
        features_df = pd.read_csv(features_csv_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Features CSV not found at {features_csv_path}")
        return None, None, None, None
    try:
        presence_df = pd.read_csv('data/testing_SDM/all_presence_point.csv', low_memory=False)
    except FileNotFoundError:
        print(f"Error: all_presence_point.csv not found")
        return None, None, None, None
    pres = presence_df[presence_df['species'] == species].copy()
    if len(pres) == 0:
        print(f"No presence points found for species: {species}")
        return None, None, None, None
    eco_dir = "data/eco_regions_polygon"
    eco_polygons = {}
    for fname in os.listdir(eco_dir):
        if fname.endswith('.wkt'):
            eco_name = fname.replace('.wkt', '')
            with open(os.path.join(eco_dir, fname), 'r') as f:
                eco_polygons[eco_name] = load_wkt(f.read().strip())
    presence_points = [Point(lon, lat) for lon, lat in zip(pres['decimalLongitude'], pres['decimalLatitude'])]
    eco_assignments = []
    for pt in presence_points:
        found = False
        for eco, poly in eco_polygons.items():
            if poly.contains(pt):
                eco_assignments.append(eco)
                found = True
                break
        if not found:
            eco_assignments.append(None)
    pres['ecoregion'] = eco_assignments
    pres = pres[pres['ecoregion'].notna()].copy()
    present_ecoregions = set(pres['ecoregion'])
    print(f"Ecoregions with presence points: {present_ecoregions}")
    print(f"Presence points in these ecoregions: {len(pres)}")
    all_absence_df = features_df.copy()
    presence_coords = set(zip(pres['decimalLongitude'], pres['decimalLatitude']))
    initial_absence_count = len(all_absence_df)
    all_absence_df = all_absence_df[~all_absence_df.apply(
        lambda r: (r['decimalLongitude'], r['decimalLatitude']) in presence_coords, axis=1
    )]
    absence_points = [Point(lon, lat) for lon, lat in zip(all_absence_df['decimalLongitude'], all_absence_df['decimalLatitude'])]
    absence_ecoregions = []
    for pt in absence_points:
        found = False
        for eco, poly in eco_polygons.items():
            if poly.contains(pt):
                absence_ecoregions.append(eco)
                found = True
                break
        if not found:
            absence_ecoregions.append(None)
    all_absence_df['ecoregion'] = absence_ecoregions
    all_absence_df = all_absence_df[all_absence_df['ecoregion'].isin(present_ecoregions)].copy()
    filtered_absence_count = len(all_absence_df)
    removed_duplicates = initial_absence_count - filtered_absence_count
    print(f"Found {len(all_absence_df)} potential absence points in ecoregions with presence")
    print(f"🔍 ABSENCE POINT FILTERING: Removed {removed_duplicates} duplicate points that matched presence coordinates")
    print(f"   Initial absence points: {initial_absence_count}")
    print(f"   After filtering: {filtered_absence_count}")
    print(f"   Duplicates removed: {removed_duplicates}")
    exclude_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'decimalLongitude', 'decimalLatitude', 'ecoregion']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    pres_with_features = pres.copy()
    pres_clean = pres_with_features[~pres_with_features[feature_cols].isna().any(axis=1)].copy()
    all_absence_clean = all_absence_df[~all_absence_df[feature_cols].isna().any(axis=1)].copy()
    for col in feature_cols:
        if col in pres_clean.columns:
            pres_clean.loc[:, col] = pd.to_numeric(pres_clean[col], errors='coerce')
        if col in all_absence_clean.columns:
            all_absence_clean.loc[:, col] = pd.to_numeric(all_absence_clean[col], errors='coerce')
    pres_clean = pres_clean.dropna(subset=feature_cols)
    all_absence_clean = all_absence_clean.dropna(subset=feature_cols)
    print(f"Valid presence points: {len(pres_clean)}")
    print(f"Valid potential absence points: {len(all_absence_clean)}")
    print("Calculating reliability scores...")
    reliability_scores = []
    for idx, row in all_absence_clean.iterrows():
        absence_features = row[feature_cols].values.astype(float)
        reliability = calculate_feature_based_reliability(absence_features, pres_clean[feature_cols].values.astype(float))
        reliability_scores.append(reliability)
    all_absence_clean['reliability_score'] = reliability_scores
    reliable_absences = all_absence_clean[all_absence_clean['reliability_score'] > 0.03]
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
    X_presence = pres_clean[feature_cols].values.astype(float)
    X_absence = absence_selected[feature_cols].values.astype(float)
    X = np.vstack([X_presence, X_absence])
    y = np.concatenate([np.ones(len(X_presence)), np.zeros(len(X_absence))])
    presence_weights = np.ones(len(X_presence))
    absence_weights = absence_selected['reliability_score'].values
    if len(absence_weights) > 0:
        min_weight = np.min(absence_weights)
        max_weight = np.max(absence_weights)
        if max_weight != min_weight:
            absence_weights = (absence_weights - min_weight) / (max_weight - min_weight)
        else:
            absence_weights = np.ones(len(absence_weights))
    sample_weights = np.concatenate([presence_weights, absence_weights])
    if bias_correction:
        
        from shapely.wkt import loads as load_wkt
        from shapely.geometry import Point
        eco_dir = "data/eco_regions_polygon"
        eco_polygons = {}
        for fname in os.listdir(eco_dir):
            if fname.endswith('.wkt'):
                eco_name = fname.replace('.wkt', '')
                with open(os.path.join(eco_dir, fname), 'r') as f:
                    eco_polygons[eco_name] = load_wkt(f.read().strip())
        def assign_ecoregion(row):
            pt = Point(row['decimalLongitude'], row['decimalLatitude'])
            for eco, poly in eco_polygons.items():
                if poly.contains(pt):
                    return eco
            return None
        pres_clean['ecoregion'] = pres_clean.apply(assign_ecoregion, axis=1)
        absence_selected['ecoregion'] = absence_selected.apply(assign_ecoregion, axis=1)
        all_ecoregions = pd.concat([pres_clean['ecoregion'], absence_selected['ecoregion']])
        eco_counts = all_ecoregions.value_counts().to_dict()
        eco_weights_raw = {eco: 1/(c+1) for eco, c in eco_counts.items()}
        w_min = min(eco_weights_raw.values())
        w_max = max(eco_weights_raw.values())
        eco_weights = {eco: 0.5 + (w_raw - w_min)/(w_max - w_min) if w_max > w_min else 1.0 for eco, w_raw in eco_weights_raw.items()}
        presence_weights_bc = pres_clean['ecoregion'].map(eco_weights).fillna(1.0).values
        absence_weights_bc = absence_selected['ecoregion'].map(eco_weights).fillna(1.0).values
        sample_weights = np.concatenate([presence_weights_bc, absence_weights_bc])
    return X, y, sample_weights, absence_selected, pres_clean, None



def all_ecoregion_level_genus_with_precomputed_features(genus, features_csv_path="data/presence_points_with_features.csv", optimize_for='tpr', bias_correction=False):
    """
    Restrict modeling to all ecoregions where presence points are found for the given genus.
    Uses WKT files to determine which ecoregions have presence points.
    """
   
    from shapely.wkt import loads as load_wkt
    from shapely.geometry import Point
    import pandas as pd
    import numpy as np

    print(f"Starting all-ecoregion-level genus modeling: {genus}")
    try:
        features_df = pd.read_csv(features_csv_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Features CSV not found at {features_csv_path}")
        return None, None, None, None
    try:
        presence_df = pd.read_csv('data/testing_SDM/all_presence_point.csv', low_memory=False)
    except FileNotFoundError:
        print(f"Error: all_presence_point.csv not found")
        return None, None, None, None
    pres = presence_df[presence_df['genus'] == genus].copy()
    if len(pres) == 0:
        print(f"No presence points found for genus: {genus}")
        return None, None, None, None
    eco_dir = "data/eco_regions_polygon"
    eco_polygons = {}
    for fname in os.listdir(eco_dir):
        if fname.endswith('.wkt'):
            eco_name = fname.replace('.wkt', '')
            with open(os.path.join(eco_dir, fname), 'r') as f:
                eco_polygons[eco_name] = load_wkt(f.read().strip())
    presence_points = [Point(lon, lat) for lon, lat in zip(pres['decimalLongitude'], pres['decimalLatitude'])]
    eco_assignments = []
    for pt in presence_points:
        found = False
        for eco, poly in eco_polygons.items():
            if poly.contains(pt):
                eco_assignments.append(eco)
                found = True
                break
        if not found:
            eco_assignments.append(None)
    pres['ecoregion'] = eco_assignments
    pres = pres[pres['ecoregion'].notna()].copy()
    present_ecoregions = set(pres['ecoregion'])
    print(f"Ecoregions with presence points: {present_ecoregions}")
    print(f"Presence points in these ecoregions: {len(pres)}")
    all_absence_df = features_df.copy()
    presence_coords = set(zip(pres['decimalLongitude'], pres['decimalLatitude']))
    initial_absence_count = len(all_absence_df)
    all_absence_df = all_absence_df[~all_absence_df.apply(
        lambda r: (r['decimalLongitude'], r['decimalLatitude']) in presence_coords, axis=1
    )]
    absence_points = [Point(lon, lat) for lon, lat in zip(all_absence_df['decimalLongitude'], all_absence_df['decimalLatitude'])]
    absence_ecoregions = []
    for pt in absence_points:
        found = False
        for eco, poly in eco_polygons.items():
            if poly.contains(pt):
                absence_ecoregions.append(eco)
                found = True
                break
        if not found:
            absence_ecoregions.append(None)
    all_absence_df['ecoregion'] = absence_ecoregions
    all_absence_df = all_absence_df[all_absence_df['ecoregion'].isin(present_ecoregions)].copy()
    filtered_absence_count = len(all_absence_df)
    removed_duplicates = initial_absence_count - filtered_absence_count
    print(f"Found {len(all_absence_df)} potential absence points in ecoregions with presence")
    print(f"🔍 ABSENCE POINT FILTERING: Removed {removed_duplicates} duplicate points that matched presence coordinates")
    print(f"   Initial absence points: {initial_absence_count}")
    print(f"   After filtering: {filtered_absence_count}")
    print(f"   Duplicates removed: {removed_duplicates}")
    exclude_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'decimalLongitude', 'decimalLatitude', 'ecoregion']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    # Fixed code:
# Merge presence points with features data
    pres_with_features = pres.merge(
        features_df, 
        on=['decimalLongitude', 'decimalLatitude'], 
        how='inner'
    )

    # Handle case where no matches found
    if len(pres_with_features) == 0:
        print(f"No matching feature data found for presence points of genus: {genus}")
        return None, None, None, None

    # Now safely access feature columns
    pres_clean = pres_with_features[~pres_with_features[feature_cols].isna().any(axis=1)].copy()

    all_absence_clean = all_absence_df[~all_absence_df[feature_cols].isna().any(axis=1)].copy()
    for col in feature_cols:
        if col in pres_clean.columns:
            pres_clean.loc[:, col] = pd.to_numeric(pres_clean[col], errors='coerce')
        if col in all_absence_clean.columns:
            all_absence_clean.loc[:, col] = pd.to_numeric(all_absence_clean[col], errors='coerce')
    pres_clean = pres_clean.dropna(subset=feature_cols)
    all_absence_clean = all_absence_clean.dropna(subset=feature_cols)
    print(f"Valid presence points: {len(pres_clean)}")
    print(f"Valid potential absence points: {len(all_absence_clean)}")
    print("Calculating reliability scores...")
    reliability_scores = []
    for idx, row in all_absence_clean.iterrows():
        absence_features = row[feature_cols].values.astype(float)
        reliability = calculate_feature_based_reliability(absence_features, pres_clean[feature_cols].values.astype(float))
        reliability_scores.append(reliability)
    all_absence_clean['reliability_score'] = reliability_scores
    reliable_absences = all_absence_clean[all_absence_clean['reliability_score'] > 0.03]
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
    X_presence = pres_clean[feature_cols].values.astype(float)
    X_absence = absence_selected[feature_cols].values.astype(float)
    X = np.vstack([X_presence, X_absence])
    y = np.concatenate([np.ones(len(X_presence)), np.zeros(len(X_absence))])
    presence_weights = np.ones(len(X_presence))
    absence_weights = absence_selected['reliability_score'].values
    if len(absence_weights) > 0:
        min_weight = np.min(absence_weights)
        max_weight = np.max(absence_weights)
        if max_weight != min_weight:
            absence_weights = (absence_weights - min_weight) / (max_weight - min_weight)
        else:
            absence_weights = np.ones(len(absence_weights))
    sample_weights = np.concatenate([presence_weights, absence_weights])
    if bias_correction:
        
        from shapely.wkt import loads as load_wkt
        from shapely.geometry import Point
        eco_dir = "data/eco_regions_polygon"
        eco_polygons = {}
        for fname in os.listdir(eco_dir):
            if fname.endswith('.wkt'):
                eco_name = fname.replace('.wkt', '')
                with open(os.path.join(eco_dir, fname), 'r') as f:
                    eco_polygons[eco_name] = load_wkt(f.read().strip())
        def assign_ecoregion(row):
            pt = Point(row['decimalLongitude'], row['decimalLatitude'])
            for eco, poly in eco_polygons.items():
                if poly.contains(pt):
                    return eco
            return None
        pres_clean['ecoregion'] = pres_clean.apply(assign_ecoregion, axis=1)
        absence_selected['ecoregion'] = absence_selected.apply(assign_ecoregion, axis=1)
        all_ecoregions = pd.concat([pres_clean['ecoregion'], absence_selected['ecoregion']])
        eco_counts = all_ecoregions.value_counts().to_dict()
        eco_weights_raw = {eco: 1/(c+1) for eco, c in eco_counts.items()}
        w_min = min(eco_weights_raw.values())
        w_max = max(eco_weights_raw.values())
        eco_weights = {eco: 0.5 + (w_raw - w_min)/(w_max - w_min) if w_max > w_min else 1.0 for eco, w_raw in eco_weights_raw.items()}
        presence_weights_bc = pres_clean['ecoregion'].map(eco_weights).fillna(1.0).values
        absence_weights_bc = absence_selected['ecoregion'].map(eco_weights).fillna(1.0).values
        sample_weights = np.concatenate([presence_weights_bc, absence_weights_bc])
    evaluate_summary2(X, y,  optimize_for=optimize_for, sample_weights=sample_weights)
    return X, y, sample_weights, absence_selected, pres_clean, None



def perform_feature_importance_for_ecoregion_species(species_list, features_csv_path="data/presence_points_with_features.csv"):
    """
    Generate SHAP feature importance analysis for ecoregion-level species models.
    """
    import shap
   
    
    print(f"Starting SHAP analysis for ecoregion-level species models...")
    
    for species in species_list:
        print(f"\n{'='*60}")
        print(f"Processing species: {species}")
        print(f"{'='*60}")
        
        try:
            # Run ecoregion-level modeling
            X, y, sample_weights, absence_selected, pres_clean, model_path = all_ecoregion_level_species_with_precomputed_features(
                species, features_csv_path=features_csv_path
            )
            
            if X is None or y is None:
                print(f"Skipping {species} - no valid data")
                continue
            
            # Get feature names
            exclude_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'decimalLongitude', 'decimalLatitude', 'ecoregion']
            features_df = pd.read_csv(features_csv_path, low_memory=False)
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            # Train the best model (Random Forest with Tversky loss)
            print(f"Training best model for SHAP analysis...")
            best_model = CustomLossRandomForest(n_estimators=100, loss_type='tversky', loss_params={'alpha': 0.3, 'beta': 0.7})
            best_model.fit(X, y, sample_weights=sample_weights)
            
            # Create output directory
            species_name = species.replace(" ", "_")
            output_dir = f"outputs/testing_SDM_out/{species_name}_ecoregion"
            os.makedirs(output_dir, exist_ok=True)
            
            # 3. SHAP Analysis
            print(f"  Generating SHAP summary plots...")
            
            try:
                # Create SHAP explainer based on model type
                if isinstance(best_model, CustomLossRandomForest):
                    # For Random Forest, use TreeExplainer
                    explainer = shap.TreeExplainer(best_model.trees[0])  # Use first tree as representative
                else:
                    # For logistic regression and other models, use KernelExplainer
                    explainer = shap.KernelExplainer(best_model.predict_proba, X[:100])  # Use small sample for background
                
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
                plt.title(f'SHAP Summary Plot - {species} (Ecoregion Level)', fontsize=16, fontweight='bold')
                
                # Save SHAP summary plot in species folder
                shap_summary_path = os.path.join(output_dir, "shap_summary.png")
                plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    SHAP summary plot saved to: {shap_summary_path}")
                
                # Create SHAP bar plot (feature importance)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type="bar", show=False)
                plt.title(f'SHAP Feature Importance - {species} (Ecoregion Level)', fontsize=16, fontweight='bold')
                
                # Save SHAP bar plot in species folder
                shap_bar_path = os.path.join(output_dir, "shap_importance.png")
                plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    SHAP bar plot saved to: {shap_bar_path}")
                
                # Calculate and save feature importance scores
                feature_importance = np.abs(shap_values).mean(0)
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                importance_path = os.path.join(output_dir, "shap_importance_scores.csv")
                importance_df.to_csv(importance_path, index=False)
                print(f"    Feature importance scores saved to: {importance_path}")
                
                # Print top 10 features
                print(f"    Top 10 most important features:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    print(f"      {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
                
            except Exception as e:
                print(f"    Error in SHAP analysis: {str(e)}")
                print(f"    Skipping SHAP analysis for {species}")
            
            print(f"  Feature importance analysis completed for {species}")
                    
        except Exception as e:
            print(f"Error processing {species}: {str(e)}")
            continue
    
    print(f"\nSHAP analysis for ecoregion-level species models completed!")

def perform_feature_importance_for_ecoregion_genera(genus_list, features_csv_path="data/presence_points_with_features.csv"):
    """
    Generate SHAP feature importance analysis for ecoregion-level genus models.
    """
    import shap
   
    
    print(f"Starting SHAP analysis for ecoregion-level genus models...")
    
    for genus in genus_list:
        print(f"\n{'='*60}")
        print(f"Processing genus: {genus}")
        print(f"{'='*60}")
        
        try:
            # Run ecoregion-level modeling
            X, y, sample_weights, absence_selected, pres_clean, model_path = all_ecoregion_level_genus_with_precomputed_features(
                genus, features_csv_path=features_csv_path
            )
            
            if X is None or y is None:
                print(f"Skipping {genus} - no valid data")
                continue
            
            # Get feature names
            exclude_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'decimalLongitude', 'decimalLatitude', 'ecoregion']
            features_df = pd.read_csv(features_csv_path, low_memory=False)
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            # Train the best model (Random Forest with Tversky loss)
            print(f"Training best model for SHAP analysis...")
            best_model = CustomLossRandomForest(n_estimators=100, loss_type='tversky', loss_params={'alpha': 0.3, 'beta': 0.7})
            best_model.fit(X, y, sample_weights=sample_weights)
            
            # Create output directory
            genus_name = genus.replace(" ", "_")
            output_dir = f"outputs/testing_SDM_out/{genus_name}_ecoregion"
            os.makedirs(output_dir, exist_ok=True)
            
            # 3. SHAP Analysis
            print(f"  Generating SHAP summary plots...")
            
            try:
                # Create SHAP explainer based on model type
                if isinstance(best_model, CustomLossRandomForest):
                    # For Random Forest, use TreeExplainer
                    explainer = shap.TreeExplainer(best_model.trees[0])  # Use first tree as representative
                else:
                    # For logistic regression and other models, use KernelExplainer
                    explainer = shap.KernelExplainer(best_model.predict_proba, X[:100])  # Use small sample for background
                
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
                plt.title(f'SHAP Summary Plot - {genus} (Ecoregion Level)', fontsize=16, fontweight='bold')
                
                # Save SHAP summary plot in genus folder
                shap_summary_path = os.path.join(output_dir, "shap_summary.png")
                plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    SHAP summary plot saved to: {shap_summary_path}")
                
                # Create SHAP bar plot (feature importance)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type="bar", show=False)
                plt.title(f'SHAP Feature Importance - {genus} (Ecoregion Level)', fontsize=16, fontweight='bold')
                
                # Save SHAP bar plot in genus folder
                shap_bar_path = os.path.join(output_dir, "shap_importance.png")
                plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    SHAP bar plot saved to: {shap_bar_path}")
                
                # Calculate and save feature importance scores
                feature_importance = np.abs(shap_values).mean(0)
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                importance_path = os.path.join(output_dir, "shap_importance_scores.csv")
                importance_df.to_csv(importance_path, index=False)
                print(f"    Feature importance scores saved to: {importance_path}")
                
                # Print top 10 features
                print(f"    Top 10 most important features:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    print(f"      {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
                
            except Exception as e:
                print(f"    Error in SHAP analysis: {str(e)}")
                print(f"    Skipping SHAP analysis for {genus}")
            
            print(f"  Feature importance analysis completed for {genus}")
                    
        except Exception as e:
            print(f"Error processing {genus}: {str(e)}")
            continue
    
    print(f"\nSHAP analysis for ecoregion-level genus models completed!")


# Example usage at the end of the file
if __name__ == "__main__":
    # Example: Generate SHAP analysis for ecoregion-level models
    species_list = ["Dalbergia sissoo","Syzygium cumini"]
    # perform_feature_importance_for_ecoregion_species(species_list)
    
    genus_list = ["Memecylon","Macaranga"]
    # perform_feature_importance_for_ecoregion_genera(genus_list)
    
    # Test ecoregion-level modeling
    # genus_list = ["Erythrina","Macaranga"]
    for genus in genus_list:
        # comprehensive_genus_with_precomputed_features(genus, features_csv_path="data/presence_points_with_features.csv")
        all_ecoregion_level_genus_with_precomputed_features(genus, features_csv_path="data/presence_points_with_features.csv")
  


