import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import joblib
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Modules'))
from Modules.features_extractor import Feature_Extractor
from Modules.custom_loss_trainers import CustomLossRandomForest, CustomLossLogisticRegression, CustomLossWeightedLogisticRegression


# --- CONFIG ---
# Set either SPECIES or GENUS and set USE_GENUS accordingly
SPECIES = 'None'  # e.g., 'Artemisia nilagirica'
GENUS = 'Artemisia'  # e.g., 'Artemisia'
USE_GENUS = True  # Set to True to use genus-level modeling

if USE_GENUS:
    target_name = GENUS
    target_safe = GENUS.replace(' ', '_')
else:
    target_name = SPECIES
    target_safe = SPECIES.replace(' ', '_')

PRESENCE_CSV = f'data/presence_features_{target_safe}.csv'
ABSENCE_CSV = f'outputs/absence_reliability_{target_safe}.csv'
MODEL_PATH = f'outputs/best_model_{target_safe}.joblib'
HEATMAP_NPY = f'outputs/{target_safe}_probability_heatmap.npy'
HEATMAP_PNG = f'outputs/{target_safe}_probability_heatmap.png'
GRID_RES_METERS = 100  # meters
GRID_SIZE_METERS = 10000  # 10km

# --- 1. Load cached presence/absence data ---
pres_df = pd.read_csv(PRESENCE_CSV)
abs_df = pd.read_csv(ABSENCE_CSV)

# --- 2. Find closest pair (presence/absence) and midpoint ---
pres_coords = pres_df[['longitude', 'latitude']].values
abs_coords = abs_df[['decimalLongitude', 'decimalLatitude']].values
tree = cKDTree(abs_coords)
dists, idxs = tree.query(pres_coords, k=1)
min_idx = np.argmin(dists)
closest_presence = pres_coords[min_idx]
closest_absence = abs_coords[idxs[min_idx]]
midpoint = (closest_presence + closest_absence) / 2
print(f"Closest presence: {closest_presence}, absence: {closest_absence}, midpoint: {midpoint}")
print(f"Grid center (midpoint): longitude={midpoint[0]}, latitude={midpoint[1]}")

# --- 3. Generate 10km x 10km grid (100m resolution) centered at midpoint, using degrees ---
meters_per_degree_lat = 111320
meters_per_degree_lon = 111320 * np.cos(np.deg2rad(midpoint[1]))
grid_res_deg_lat = GRID_RES_METERS / meters_per_degree_lat
grid_res_deg_lon = GRID_RES_METERS / meters_per_degree_lon
grid_half_deg_lat = (GRID_SIZE_METERS / 2) / meters_per_degree_lat
grid_half_deg_lon = (GRID_SIZE_METERS / 2) / meters_per_degree_lon

print(f"Grid step: {grid_res_deg_lon:.6f} deg lon, {grid_res_deg_lat:.6f} deg lat")
print(f"Grid bounds: lon {midpoint[0] - grid_half_deg_lon:.6f} to {midpoint[0] + grid_half_deg_lon:.6f}, "
      f"lat {midpoint[1] - grid_half_deg_lat:.6f} to {midpoint[1] + grid_half_deg_lat:.6f}")

grid_x = np.arange(midpoint[0] - grid_half_deg_lon, midpoint[0] + grid_half_deg_lon, grid_res_deg_lon)
grid_y = np.arange(midpoint[1] - grid_half_deg_lat, midpoint[1] + grid_half_deg_lat, grid_res_deg_lat)
xx, yy = np.meshgrid(grid_x, grid_y)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])
print(f"Grid shape: {xx.shape}, total points: {grid_points.shape[0]}")

# --- 4. Load trained model ---
model = joblib.load(MODEL_PATH)

# --- 5. Extract features for each grid point using Feature_Extractor ---
# Use the same feature columns as in the cached files (excluding longitude, latitude)
# For presence, use longitude/latitude; for absence, use decimalLongitude/decimalLatitude
feature_cols = [col for col in pres_df.columns if col not in ['longitude', 'latitude']]

# Initialize Earth Engine and Feature_Extractor
try:
    import ee
    ee.Initialize()
except Exception as e:
    print("Earth Engine initialization failed. Please ensure you have authenticated.")
    raise e

fe = Feature_Extractor(ee)

grid_df = pd.DataFrame(grid_points, columns=['longitude', 'latitude'])
print("Extracting features for grid points (this may take a while)...")

# Cache file for grid features
cache_grid_features = f"outputs/grid_features_cache_{target_safe}_midpoint_{midpoint[0]:.4f}_{midpoint[1]:.4f}.csv"

if os.path.exists(cache_grid_features):
    print(f"Loading grid features from cache: {cache_grid_features}")
    grid_features_df = pd.read_csv(cache_grid_features)
else:
    grid_features_df = fe.add_features(grid_df)
    grid_features_df.to_csv(cache_grid_features, index=False)
    print(f"Saved grid features to cache: {cache_grid_features}")

# Ensure the order of features matches the model's expectation
features = grid_features_df[feature_cols].values

# --- Elevation Heatmap ---
if 'elevation' in grid_features_df.columns:
    elevation = grid_features_df['elevation'].values.reshape(xx.shape)
    ELEVATION_HEATMAP_PNG = f'outputs/{target_safe}_elevation_heatmap.png'
    plt.figure(figsize=(8, 6))
    plt.imshow(elevation, origin='lower', extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title(f'{target_name} Elevation Heatmap (10km x 10km, 100m res)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.yticks(np.linspace(grid_y[0], grid_y[-1], num=6))
    plt.xticks(np.linspace(grid_x[0], grid_x[-1], num=6))
    plt.tight_layout()
    plt.savefig(ELEVATION_HEATMAP_PNG, dpi=300)
    plt.close()
    print(f"Saved elevation heatmap as {ELEVATION_HEATMAP_PNG}")
else:
    print("Elevation column not found in grid features. Skipping elevation heatmap.")

# --- 6. Predict probability for each grid point ---
if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(features)[:, 1]
else:
    proba = model.predict(features)  # fallback

heatmap = proba.reshape(xx.shape)

# --- 7. Save heatmap ---
os.makedirs('outputs', exist_ok=True)
np.save(HEATMAP_NPY, heatmap)
plt.figure(figsize=(8, 6))
plt.imshow(heatmap, origin='lower', extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], cmap='viridis')
plt.colorbar(label='Probability')
plt.title(f'{target_name} Probability Heatmap (10km x 10km, 100m res)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# Ensure y-ticks (latitude) are visible and not overlapped
plt.yticks(np.linspace(grid_y[0], grid_y[-1], num=6))
plt.xticks(np.linspace(grid_x[0], grid_x[-1], num=6))
plt.tight_layout()
plt.savefig(HEATMAP_PNG, dpi=300)
plt.close()
print(f"Saved heatmap as {HEATMAP_NPY} and {HEATMAP_PNG}") 