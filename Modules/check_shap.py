import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# ====== PARAMETERS ======
raster_path = "elevation.tif"              # Path to elevation raster (e.g., SRTM, GMTED, etc.)
occ_csv = "occurrences_with_shap.csv"      # CSV must have: latitude, longitude, shap_elevation
lat_col, lon_col = "latitude", "longitude"
shap_col = "shap_elevation"                # Column name for SHAP values of elevation

# ====== 1. Load occurrence data ======
occ = pd.read_csv(occ_csv)
gdf = gpd.GeoDataFrame(
    occ,
    geometry=gpd.points_from_xy(occ[lon_col], occ[lat_col]),
    crs="EPSG:4326"
)

# ====== 2. Sample elevation values from raster ======
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    # Reproject points if needed
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    sampled_vals = [val[0] if val is not None else np.nan for val in src.sample(coords)]

gdf["elevation_value"] = sampled_vals

# ====== 3. Quick check ======
print(gdf[["elevation_value", shap_col]].describe())
print("Pearson corr:", gdf["elevation_value"].corr(gdf[shap_col]))

# ====== 4. Plot maps ======
fig, ax = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

# left: raster + points colored by elevation
with rasterio.open(raster_path) as src:
    show(src, ax=ax[0], title="Elevation map with points")
    gdf.plot(column="elevation_value", ax=ax[0], markersize=25,
             cmap="terrain", legend=True, alpha=0.7)

# right: raster + points colored by SHAP value
with rasterio.open(raster_path) as src:
    show(src, ax=ax[1], title="Points colored by SHAP (elevation)")
    gdf.plot(column=shap_col, ax=ax[1], markersize=25,
             cmap="RdYlBu_r", legend=True, alpha=0.7)

plt.show()

# ====== 5. Scatter plot elevation vs SHAP ======
plt.figure(figsize=(7,6))
plt.scatter(gdf["elevation_value"], gdf[shap_col], alpha=0.6, c="blue")
plt.xlabel("Elevation value")
plt.ylabel("SHAP value (impact on model output)")
plt.title("Elevation vs SHAP for Syzygium cumini")
plt.grid(True)
plt.show()
