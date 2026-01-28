import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from joblib import load
import ee
from .features_extractor import Feature_Extractor
from .custom_loss_trainers import CustomLossRandomForest, CustomLossLogisticRegression, CustomLossWeightedLogisticRegression
import rasterio
from rasterio.transform import from_origin
import os

# Define your lists
species_list = ["Mangifera_indica","Syzygium cumini","Dalbergia sissoo","Cocos nucifera","Artocarpus heterophyllus"]
genus_list = ["Primula","Memecylon","Macaranga","Artemisia"]

# === Step 1: Generate Grid Inside India from Shapefile ===

# Load India boundary
india = gpd.read_file("Inputs/India_Country_Boundary.shp").to_crs("EPSG:4326")
resolution = 0.05  # ~1 km


# Define grid resolution

# === Step 2: Extract Features Using Earth Engine with Caching ===
cache_path = "Outputs/grid_features_cache_5km.csv"
if os.path.exists(cache_path):
    print(f"Loading features from cache: {cache_path}")
    features_df = pd.read_csv(cache_path)
else:
    
    minx, miny, maxx, maxy = india.total_bounds

    x_coords = np.arange(minx, maxx, resolution)

    y_coords = np.arange(miny, maxy, resolution)

    grid_points = [Point(x, y) for y in y_coords for x in x_coords]

    # Filter points inside India
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")

    grid_in_india = grid_gdf[grid_gdf.within(india.geometry.union_all())].copy()

    grid_in_india["longitude"] = grid_in_india.geometry.x

    grid_in_india["latitude"] = grid_in_india.geometry.y

    grid_df = grid_in_india[["longitude", "latitude"]].reset_index(drop=True)

    # Print number of points in grid
    print(f"Number of points in grid: {len(grid_df)}")

    print("On step 1")
    ee.Initialize()  # replace with your actual EE project ID
    print("passed step 1")
    fe = Feature_Extractor(ee)
    print("passed step 2")
    features_df = fe.add_features(grid_df)  # returns DataFrame with features + lat/lon
    print("passed step 3")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    features_df.to_csv(cache_path, index=False)
    print(f"Features saved to cache: {cache_path}")

# Extract feature columns
feature_cols = [col for col in features_df.columns if col not in ['latitude', 'longitude']]
X_all = features_df[feature_cols].values.astype(float)

# === Step 3: For each species and genus, load model and generate maps ===
for name in species_list + genus_list:
    safe_name = name.replace(' ', '_')
    print(f"\n=== Processing: {name} ===")
    model_path = f"Outputs/best_model_{safe_name}.joblib"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Skipping.")
        continue
    clf = load(model_path)
    # Predict probabilities
    features_df[f'probability_{safe_name}'] = clf.predict_proba(X_all)[:, 1]
    # Export to CSV
    csv_path = f"Outputs/{safe_name}_probability_map.csv"
    features_df[['longitude', 'latitude', f'probability_{safe_name}']].rename(columns={f'probability_{safe_name}': 'probability'}).to_csv(csv_path, index=False)
    print(f"✅ CSV exported: {csv_path}")
    # Prepare raster grid
    longs = sorted(features_df['longitude'].unique())
    lats = sorted(features_df['latitude'].unique())
    pixel_size = resolution
    transform = from_origin(min(longs), max(lats), pixel_size, pixel_size)
    width = len(longs)
    height = len(lats)
    # Pivot to 2D matrix (lat × lon)
    prob_grid = features_df.pivot(index='latitude', columns='longitude', values=f'probability_{safe_name}')
    raster = prob_grid.to_numpy()[::-1]  # flip vertically for rasterio
    # Save GeoTIFF
    tiff_path = f"Outputs/{safe_name}_probability_map.tif"
    with rasterio.open(
        tiff_path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs=india.crs.to_wkt(),
        transform=transform
    ) as dst:
        dst.write(raster, 1)
    print(f"✅ GeoTIFF exported: {tiff_path}")
