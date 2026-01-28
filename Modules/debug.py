import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report

def dice_loss(y_true, y_pred, smooth=1):
    """Dice loss for binary segmentation"""
    y_true = y_true.astype(np.float32)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Clip predictions
    
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return 1 - (2 * intersection + smooth) / (union + smooth)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Standard focal loss implementation
    ce_loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    focal_weight = (1 - p_t) ** gamma
    alpha_weight = np.where(y_true == 1, alpha, 1 - alpha)
    
    return np.mean(alpha_weight * focal_weight * ce_loss)

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1):
    """
    Tversky loss - corrected with standard parameters
    alpha=0.3, beta=0.7 emphasizes recall (penalizes false negatives more)
    """
    y_true = y_true.astype(np.float32)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    TP = np.sum(y_true * y_pred)
    FP = np.sum((1 - y_true) * y_pred)
    FN = np.sum(y_true * (1 - y_pred))
    
    tversky_index = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tversky_index

class CustomLossRandomForest:
    def __init__(self, n_estimators=100, loss_type='dice', loss_params=None):
        self.n_estimators = n_estimators
        self.loss_type = loss_type
        self.loss_params = loss_params or {}
        self.trees = []
        self.weights = None
        self.feature_importances_ = None
        
    def _loss_function(self, y_true, y_pred):
        """Get the appropriate loss function"""
        loss_functions = {
            'dice': dice_loss,
            'focal': focal_loss,
            'tversky': tversky_loss
        }
        return loss_functions[self.loss_type](y_true, y_pred, **self.loss_params)
    
    def fit(self, X, y, sample_weights=None, max_iter=200, restarts=15, verbose=True):
        """
        Fit the custom loss random forest
        
        Key fixes:
        1. Better initialization strategies
        2. More robust optimization
        3. Improved convergence checking
        4. Better handling of edge cases
        """
        if verbose:
            print(f"Training Custom Loss Random Forest with {self.loss_type} loss...")
        
        # Train base random forest
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=42,
            max_depth=10,  # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf.fit(X, y, sample_weight=sample_weights)
        
        # Get predictions from each tree
        tree_predictions = np.array([tree.predict_proba(X)[:, 1] for tree in rf.estimators_]).T
        
        if verbose:
            print(f"Tree predictions shape: {tree_predictions.shape}")
            print(f"Tree prediction ranges: {tree_predictions.min():.4f} to {tree_predictions.max():.4f}")
        
        # Optimize ensemble weights
        best_loss = np.inf
        best_weights = None
        n_trees = tree_predictions.shape[1]
        
        # Try different initialization strategies
        initialization_strategies = [
            np.ones(n_trees) / n_trees,  # Uniform weights
            np.random.dirichlet(np.ones(n_trees)),  # Random Dirichlet
            np.random.dirichlet(np.ones(n_trees) * 2),  # More concentrated Dirichlet
        ]
        
        # Add some random restarts
        for _ in range(restarts - len(initialization_strategies)):
            initialization_strategies.append(np.random.dirichlet(np.ones(n_trees)))
        
        for restart, init_weights in enumerate(initialization_strategies):
            try:
                def objective(weights):
                    # Ensure weights are positive and sum to 1
                    w = np.abs(weights)
                    w = w / (np.sum(w) + 1e-8)  # Add small epsilon to prevent division by zero
                    
                    # Compute ensemble prediction
                    ensemble_pred = tree_predictions.dot(w)
                    
                    # Clip predictions to valid range
                    ensemble_pred = np.clip(ensemble_pred, 1e-7, 1 - 1e-7)
                    
                    # Compute loss
                    loss_val = self._loss_function(y, ensemble_pred)
                    
                    # Add small regularization to prefer simpler solutions
                    l2_reg = 0.001 * np.sum(w ** 2)
                    
                    return loss_val + l2_reg
                
                # Multiple optimization attempts with different methods
                methods = ['L-BFGS-B', 'SLSQP']
                
                for method in methods:
                    try:
                        result = minimize(
                            objective, 
                            init_weights,
                            method=method,
                            bounds=[(1e-6, 1.0)] * n_trees,
                            options={'maxiter': max_iter, 'ftol': 1e-9}
                        )
                        
                        if result.success or result.fun < best_loss:
                            # Normalize weights
                            candidate_weights = np.abs(result.x)
                            candidate_weights = candidate_weights / np.sum(candidate_weights)
                            
                            # Evaluate candidate
                            ensemble_pred = tree_predictions.dot(candidate_weights)
                            ensemble_pred = np.clip(ensemble_pred, 1e-7, 1 - 1e-7)
                            candidate_loss = self._loss_function(y, ensemble_pred)
                            
                            if candidate_loss < best_loss:
                                best_loss = candidate_loss
                                best_weights = candidate_weights
                                if verbose:
                                    print(f"Restart {restart}, Method {method}: Loss = {candidate_loss:.6f}")
                    
                    except Exception as e:
                        if verbose:
                            print(f"Optimization failed for restart {restart}, method {method}: {e}")
                        continue
                        
            except Exception as e:
                if verbose:
                    print(f"Restart {restart} failed: {e}")
                continue
        
        # Fallback to uniform weights if optimization failed
        if best_weights is None:
            print("Warning: Optimization failed, using uniform weights")
            best_weights = np.ones(n_trees) / n_trees
            ensemble_pred = tree_predictions.dot(best_weights)
            best_loss = self._loss_function(y, ensemble_pred)
        
        self.weights = best_weights
        self.trees = rf.estimators_
        self.feature_importances_ = rf.feature_importances_
        
        if verbose:
            print(f"Final optimized loss: {best_loss:.6f}")
            print(f"Weight distribution - Min: {self.weights.min():.4f}, Max: {self.weights.max():.4f}")
            print(f"Number of dominant trees (weight > 0.1): {np.sum(self.weights > 0.1)}")
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.trees or self.weights is None:
            raise ValueError("Model not fitted yet")
        
        # Get predictions from each tree
        tree_predictions = np.array([tree.predict_proba(X)[:, 1] for tree in self.trees]).T
        
        # Weighted ensemble prediction
        ensemble_pred = tree_predictions.dot(self.weights)
        ensemble_pred = np.clip(ensemble_pred, 1e-7, 1 - 1e-7)
        
        # Return probabilities for both classes
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def predict(self, X):
        """Predict binary classes"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

# Testing and comparison function
def test_custom_loss_models():
    """Test the custom loss models and compare performance"""
    print("Testing Custom Loss Random Forest Models")
    print("=" * 50)
    
    # Generate imbalanced binary classification dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.7, 0.3],  # Imbalanced classes
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
    
    # Test different loss functions
    loss_configs = {
        'dice': {'loss_type': 'dice', 'loss_params': {}},
        'focal': {'loss_type': 'focal', 'loss_params': {'alpha': 0.25, 'gamma': 2.0}},
        'tversky': {'loss_type': 'tversky', 'loss_params': {'alpha': 0.3, 'beta': 0.7}}
    }
    
    results = {}
    
    for loss_name, config in loss_configs.items():
        print(f"\n--- Training with {loss_name.upper()} loss ---")
        
        # Train model
        model = CustomLossRandomForest(
            n_estimators=50,  # Reduced for faster testing
            **config
        )
        
        model.fit(X_train, y_train, verbose=True)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        test_loss = model._loss_function(y_test, y_pred_proba)
        
        results[loss_name] = {
            'accuracy': accuracy,
            'test_loss': test_loss,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test {loss_name} Loss: {test_loss:.6f}")
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    
    for loss_name, result in results.items():
        print(f"{loss_name.upper():8} - Accuracy: {result['accuracy']:.4f}, Loss: {result['test_loss']:.6f}")
    
    # Find best performing model
    best_accuracy_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_loss_model = min(results.items(), key=lambda x: x[1]['test_loss'])
    
    print(f"\nBest Accuracy: {best_accuracy_model[0].upper()} ({best_accuracy_model[1]['accuracy']:.4f})")
    print(f"Best Loss: {best_loss_model[0].upper()} ({best_loss_model[1]['test_loss']:.6f})")
    
    # Detailed classification reports
    print(f"\n--- Detailed Classification Reports ---")
    for loss_name, result in results.items():
        print(f"\n{loss_name.upper()} Loss Model:")
        print(classification_report(y_test, result['predictions'], target_names=['Class 0', 'Class 1']))
    
    return results

# Additional debugging function
def debug_loss_behavior():
    """Debug why dice loss might not be performing as expected"""
    print("\n" + "=" * 50)
    print("DEBUGGING LOSS FUNCTION BEHAVIOR")
    print("=" * 50)
    
    # Create sample predictions with different characteristics
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
    
    # Different prediction scenarios
    scenarios = {
        'Perfect': y_true.astype(float),
        'Good (0.9 correlation)': y_true * 0.9 + np.random.random(n_samples) * 0.1,
        'Moderate (0.7 correlation)': y_true * 0.7 + np.random.random(n_samples) * 0.3,
        'Random': np.random.random(n_samples),
        'Anti-correlated': 1 - y_true.astype(float)
    }
    
    print(f"Sample size: {n_samples}, Positive class ratio: {np.mean(y_true):.2f}")
    print()
    
    # Test each scenario
    for scenario_name, y_pred in scenarios.items():
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Ensure valid range
        
        dice = dice_loss(y_true, y_pred)
        focal = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)
        tversky = tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7)
        
        # Calculate accuracy for reference
        accuracy = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        
        print(f"{scenario_name:20} - Acc: {accuracy:.3f}, Dice: {dice:.4f}, Focal: {focal:.4f}, Tversky: {tversky:.4f}")

# Improved version with better optimization
class ImprovedCustomLossRandomForest:
    def __init__(self, n_estimators=100, loss_type='dice', loss_params=None):
        self.n_estimators = n_estimators
        self.loss_type = loss_type
        self.loss_params = loss_params or {}
        self.trees = []
        self.weights = None
        
    def _loss_function(self, y_true, y_pred):
        """Get the appropriate loss function with error handling"""
        loss_functions = {
            'dice': dice_loss,
            'focal': focal_loss,
            'tversky': tversky_loss
        }
        
        # Ensure inputs are valid
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        try:
            return loss_functions[self.loss_type](y_true, y_pred, **self.loss_params)
        except Exception as e:
            print(f"Loss computation failed: {e}")
            return np.inf
    
    def fit(self, X, y, sample_weights=None, max_iter=300, restarts=20, verbose=True):
        """Improved fitting with better optimization strategies"""
        
        # Train base random forest with better parameters
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            max_depth=None,  # Allow deeper trees
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            oob_score=True
        )
        
        rf.fit(X, y, sample_weight=sample_weights)
        
        if verbose:
            print(f"Base RF OOB Score: {rf.oob_score_:.4f}")
        
        # Get tree predictions
        tree_predictions = np.array([tree.predict_proba(X)[:, 1] for tree in rf.estimators_]).T
        
        if verbose:
            print(f"Tree predictions - Mean: {tree_predictions.mean():.4f}, Std: {tree_predictions.std():.4f}")
        
        # Optimize weights with multiple strategies
        best_loss = np.inf
        best_weights = None
        n_trees = tree_predictions.shape[1]
        
        def objective_with_constraints(weights):
            """Objective function with numerical stability"""
            # Softmax normalization for better numerical stability
            exp_weights = np.exp(weights - np.max(weights))
            normalized_weights = exp_weights / np.sum(exp_weights)
            
            # Compute ensemble prediction
            ensemble_pred = tree_predictions.dot(normalized_weights)
            ensemble_pred = np.clip(ensemble_pred, 1e-7, 1 - 1e-7)
            
            # Compute loss
            loss_val = self._loss_function(y, ensemble_pred)
            
            # Add entropy regularization to prevent overfitting
            entropy_reg = 0.01 * np.sum(normalized_weights * np.log(normalized_weights + 1e-8))
            
            return loss_val + entropy_reg
        
        # Try multiple initialization strategies
        for restart in range(restarts):
            try:
                # Different initialization strategies
                if restart == 0:
                    init_weights = np.zeros(n_trees)  # Start with uniform (after softmax)
                elif restart == 1:
                    init_weights = np.random.normal(0, 0.1, n_trees)  # Small random
                else:
                    init_weights = np.random.normal(0, 1, n_trees)  # Large random
                
                # Optimize using different methods
                methods = ['L-BFGS-B', 'Powell', 'Nelder-Mead']
                
                for method in methods[:2]:  # Try first two methods
                    try:
                        result = minimize(
                            objective_with_constraints,
                            init_weights,
                            method=method,
                            options={'maxiter': max_iter, 'ftol': 1e-12}
                        )
                        
                        if result.success or result.fun < best_loss:
                            # Convert back to normalized weights
                            exp_weights = np.exp(result.x - np.max(result.x))
                            candidate_weights = exp_weights / np.sum(exp_weights)
                            
                            # Validate the candidate
                            ensemble_pred = tree_predictions.dot(candidate_weights)
                            ensemble_pred = np.clip(ensemble_pred, 1e-7, 1 - 1e-7)
                            candidate_loss = self._loss_function(y, ensemble_pred)
                            
                            if candidate_loss < best_loss:
                                best_loss = candidate_loss
                                best_weights = candidate_weights
                                
                                if verbose and restart % 5 == 0:
                                    print(f"Restart {restart}: {method} - Loss = {candidate_loss:.6f}")
                    
                    except Exception as e:
                        continue
                        
            except Exception as e:
                continue
        
        # Final fallback
        if best_weights is None:
            print("Warning: All optimizations failed, using uniform weights")
            best_weights = np.ones(n_trees) / n_trees
            ensemble_pred = tree_predictions.dot(best_weights)
            best_loss = self._loss_function(y, ensemble_pred)
        
        self.weights = best_weights
        self.trees = rf.estimators_
        self.feature_importances_ = rf.feature_importances_
        
        if verbose:
            print(f"Final {self.loss_type} loss: {best_loss:.6f}")
            print(f"Weight entropy: {-np.sum(self.weights * np.log(self.weights + 1e-8)):.4f}")
            print(f"Effective number of trees: {1/np.sum(self.weights**2):.1f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.trees or self.weights is None:
            raise ValueError("Model not fitted yet")
        
        tree_predictions = np.array([tree.predict_proba(X)[:, 1] for tree in self.trees]).T
        ensemble_pred = tree_predictions.dot(self.weights)
        ensemble_pred = np.clip(ensemble_pred, 1e-7, 1 - 1e-7)
        
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def predict(self, X):
        """Predict binary classes"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

def comprehensive_test():
    """Comprehensive test comparing all models"""
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 60)
    
    # Test with multiple datasets
    datasets = [
        ('Balanced', make_classification(n_samples=800, n_features=20, weights=[0.5, 0.5], random_state=42)),
        ('Imbalanced', make_classification(n_samples=800, n_features=20, weights=[0.8, 0.2], random_state=42)),
        ('Highly Imbalanced', make_classification(n_samples=800, n_features=20, weights=[0.9, 0.1], random_state=42))
    ]
    
    for dataset_name, (X, y) in datasets:
        print(f"\n{'='*20} {dataset_name} Dataset {'='*20}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        print(f"Class distribution: {np.bincount(y_train)}")
        
        # Test improved models
        loss_configs = {
            'Dice': {'loss_type': 'dice', 'loss_params': {}},
            'Focal': {'loss_type': 'focal', 'loss_params': {'alpha': 0.25, 'gamma': 2.0}},
            'Tversky': {'loss_type': 'tversky', 'loss_params': {'alpha': 0.3, 'beta': 0.7}}
        }
        
        results = {}
        
        for loss_name, config in loss_configs.items():
            model = ImprovedCustomLossRandomForest(n_estimators=30, **config)
            model.fit(X_train, y_train, verbose=False)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[loss_name] = accuracy
            print(f"{loss_name:8} Loss - Accuracy: {accuracy:.4f}")
        
        # Show best performer
        best_model = max(results.items(), key=lambda x: x[1])
        print(f"Best model: {best_model[0]} (Accuracy: {best_model[1]:.4f})")

if __name__ == "__main__":
    # Run debugging first
    debug_loss_behavior()
    
    # Test original implementation
    print("\n" + "="*60)
    print("TESTING ORIGINAL IMPLEMENTATION")
    test_custom_loss_models()
    
    # Test improved implementation
    print("\n" + "="*60)
    print("TESTING IMPROVED IMPLEMENTATION")
    comprehensive_test()