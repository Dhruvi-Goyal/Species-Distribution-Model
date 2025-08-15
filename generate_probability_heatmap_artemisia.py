import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import joblib
import matplotlib.pyplot as plt

# --- CONFIG ---
CACHE_CSV = 'presence_absence_cache.csv'  # Update path as needed
MODEL_PATH = 'artemisia_best_model.pkl'   # Update path as needed
HEATMAP_NPY = 'artemisia_probability_heatmap.npy'
HEATMAP_PNG = 'artemisia_probability_heatmap.png'
GRID_RES = 100  # meters
GRID_SIZE = 10000  # 10km

# --- 1. Load presence/absence data ---
df = pd.read_csv(CACHE_CSV)

# --- 2. Find closest pair (presence/absence) and midpoint ---
presence = df[df['label'] == 1][['x', 'y']].values
absence = df[df['label'] == 0][['x', 'y']].values

tree = cKDTree(absence)
dists, idxs = tree.query(presence, k=1)
min_idx = np.argmin(dists)
closest_presence = presence[min_idx]
closest_absence = absence[idxs[min_idx]]
midpoint = (closest_presence + closest_absence) / 2
print(f"Closest presence: {closest_presence}, absence: {closest_absence}, midpoint: {midpoint}")

# --- 3. Generate 10km x 10km grid (100m resolution) centered at midpoint ---
grid_half = GRID_SIZE // 2
grid_x = np.arange(midpoint[0] - grid_half, midpoint[0] + grid_half, GRID_RES)
grid_y = np.arange(midpoint[1] - grid_half, midpoint[1] + grid_half, GRID_RES)
xx, yy = np.meshgrid(grid_x, grid_y)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])
print(f"Grid shape: {xx.shape}, total points: {grid_points.shape[0]}")

# --- 4. Load trained model ---
model = joblib.load(MODEL_PATH)

# --- 5. Extract features for each grid point ---
def extract_features_for_points(points):
    # Placeholder: Replace with your actual feature extraction logic
    # For example, sample rasters at these points, or use precomputed layers
    # Here, we just use dummy features (all zeros)
    # Return shape: (n_points, n_features)
    n_points = points.shape[0]
    n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 10
    return np.zeros((n_points, n_features))

features = extract_features_for_points(grid_points)

# --- 6. Predict probability for each grid point ---
if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(features)[:, 1]
else:
    proba = model.predict(features)  # fallback

heatmap = proba.reshape(xx.shape)

# --- 7. Save heatmap ---
np.save(HEATMAP_NPY, heatmap)
plt.figure(figsize=(8, 6))
plt.imshow(heatmap, origin='lower', extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], cmap='viridis')
plt.colorbar(label='Probability')
plt.title('Artemisia Probability Heatmap (10km x 10km, 100m res)')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.tight_layout()
plt.savefig(HEATMAP_PNG, dpi=300)
plt.close()
print(f"Saved heatmap as {HEATMAP_NPY} and {HEATMAP_PNG}") 