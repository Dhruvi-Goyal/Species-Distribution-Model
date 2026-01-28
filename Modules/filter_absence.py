# filter_absence.py

import pandas as pd
import numpy as np
import sys

# Haversine formula: distance in kilometers between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def filter_by_distance(species_name):
         # ← hard‑coded species
    all_csv    = "data/testing_SDM/all_presence_point.csv"
    feat_csv   = "data/presence_points_with_features.csv"
    out_csv    = "data/filtered_presence_points_" + species_name + ".csv"
    min_km     = 15.0

    # 1. Load data
    df_all  = pd.read_csv(all_csv)
    df_feat = pd.read_csv(feat_csv)

    # 2. Select only this species
    base_pts = df_all.loc[df_all['species'] == species_name, 
                          ['decimalLatitude', 'decimalLongitude']].to_numpy()
    cand_pts = df_feat.loc[df_feat['species'] != species_name].copy()

    if base_pts.size == 0:
        print(f"No presence points for '{species_name}' in {all_csv}")
        sys.exit(1)

    # 3. Compute minimum distance to any base point
    def min_dist_to_base(row):
        lat, lon = row['decimalLatitude'], row['decimalLongitude']
        dists = haversine(lat, lon, base_pts[:,0], base_pts[:,1])
        return dists.min()

    cand_pts['min_dist_km'] = cand_pts.apply(min_dist_to_base, axis=1)

    # 4. Filter out any point within min_km
    filtered = cand_pts[cand_pts['min_dist_km'] > min_km].copy()

    # 5. Save results
    filtered.drop(columns=['min_dist_km'], inplace=True)
    filtered.to_csv(out_csv, index=False)
    print(f"✅ {len(filtered)} points ≥ {min_km} km from existing presences written to {out_csv}")

if __name__ == "__main__":
    species_list = ["Mangifera indica","Syzygium cumini","Artocarpus heterophyllus","Cocos nucifera","Moringa oleifera","Dalbergia sissoo"]
    for species in species_list:
        filter_by_distance(species)
