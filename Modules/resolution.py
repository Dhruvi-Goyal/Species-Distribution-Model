"""
Analyze rainfall variation within a 100 km × 100 km area
using WorldClim BIO12 (annual precipitation).
"""

import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic


class LocalRainfallAnalyzer:
    """Analyze rainfall variation in a small region (e.g., 100 × 100 km)."""

    def __init__(self, project_id):
        print("🌍 Initializing Earth Engine...")
        ee.Initialize(project=project_id)

    def make_square_region(self, center_lat, center_lon, side_km=100):
        """Return an ee.Geometry.Rectangle for a square region of side_km km."""
        # Convert half side from km → degrees (approx 1° ≈ 111 km)
        delta_deg = (side_km / 2) / 111.0
        region = ee.Geometry.Rectangle([
            center_lon - delta_deg,
            center_lat - delta_deg,
            center_lon + delta_deg,
            center_lat + delta_deg
        ])
        print(f"🗺️ Created {side_km} km × {side_km} km square centered at ({center_lat}, {center_lon})")
        return region

    def generate_random_points(self, region, n_points=300):
        """Generate random sample points in region."""
        points = ee.FeatureCollection.randomPoints(region=region, points=n_points, seed=42)
        coords_list = points.aggregate_array('.geo').getInfo()
        lat_lon_pairs = [(c['coordinates'][1], c['coordinates'][0]) for c in coords_list]
        print(f"✅ Generated {len(lat_lon_pairs)} random points")
        return lat_lon_pairs

    def extract_rainfall(self, rainfall_image, points, feature_name='bio12'):
        """Extract rainfall values at given coordinates."""
        ee_points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lon, lat])) for lat, lon in points])
        sampled = rainfall_image.sampleRegions(collection=ee_points, scale=1000, geometries=True)
        sampled_list = sampled.getInfo()['features']

        data = []
        for f in sampled_list:
            coords = f['geometry']['coordinates']
            props = f['properties']
            value = props.get(feature_name, list(props.values())[0] if props else None)
            data.append({'latitude': coords[1], 'longitude': coords[0], feature_name: value})

        df = pd.DataFrame(data).dropna(subset=[feature_name])
        print(f"✅ Extracted {len(df)} rainfall values (BIO12)")
        return df

    def compute_variogram(self, df, feature_name, n_bins=15, max_distance_km=100):
        """Compute local semi-variogram."""
        coords = df[['latitude', 'longitude']].values
        values = df[feature_name].values
        distances, semivariances = [], []

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = geodesic(coords[i], coords[j]).km
                if d <= max_distance_km:
                    distances.append(d)
                    semivariances.append(0.5 * (values[i] - values[j]) ** 2)

        distances, semivariances = np.array(distances), np.array(semivariances)
        bins = np.linspace(0, max_distance_km, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_semivar = [np.mean(semivariances[(distances >= bins[k]) & (distances < bins[k + 1])])
                       if np.any((distances >= bins[k]) & (distances < bins[k + 1])) else np.nan
                       for k in range(n_bins)]

        sill = np.nanmax(bin_semivar)
        threshold = 0.95 * sill
        stable_idx = np.where(bin_semivar >= threshold)[0]
        eff_range = bin_centers[stable_idx[0]] if len(stable_idx) > 0 else None

        return bin_centers, bin_semivar, eff_range

    def plot_variogram(self, distances, semivar, eff_range=None):
        plt.figure(figsize=(9, 6))
        plt.plot(distances, semivar, 'o-', linewidth=2)
        if eff_range:
            plt.axvline(eff_range, color='red', linestyle='--', label=f'Effective range = {eff_range:.1f} km')
        plt.xlabel("Distance (km)")
        plt.ylabel("Semivariance")
        plt.title("Local Rainfall Semi-Variogram (BIO12)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    analyzer = LocalRainfallAnalyzer(project_id='ee-mtpictd')

    # Define ~100 km × 100 km region (you can change coordinates)
    # Example: central India near Nagpur
    region = analyzer.make_square_region(center_lat=21.15, center_lon=79.08, side_km=100)

    # Load rainfall dataset (WorldClim v1 BIO12 = annual precipitation)
    rainfall = ee.Image("WORLDCLIM/V1/BIO").select('bio12')

    # Generate sample points & extract rainfall values
    points = analyzer.generate_random_points(region, n_points=3000)
    df = analyzer.extract_rainfall(rainfall, points)

    # Compute variogram (local variation)
    distances, semivar, eff_range = analyzer.compute_variogram(df, 'bio12', max_distance_km=100)
    analyzer.plot_variogram(distances, semivar, eff_range)

    if eff_range:
        print(f"🌦️ Rainfall effective variation scale ≈ {eff_range:.1f} km")
    else:
        print("⚠️ Could not determine effective range (rainfall may vary uniformly).")


if __name__ == "__main__":
    # ee.Authenticate()  # run once manually if needed
    main()
