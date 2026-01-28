import pandas as pd
import os
from .custom_loss_trainers import comprehensive_species_with_precomputed_features, comprehensive_genus_with_precomputed_features
import folium
from geopy.distance import geodesic
from .features_extractor import Feature_Extractor
import ee

species_list = ["Mangifera indica","Syzygium cumini","Dalbergia sissoo","Artocarpus heterophyllus","Cocos nucifera"]
# genus_list = ["Artemisia", "Macaranga", "Memecylon"]

features_csv_path = "data/presence_points_with_features.csv"
presence_csv = "data/testing_SDM/all_presence_point.csv"

# Load all presence points
all_presence = pd.read_csv(presence_csv)

# Helper to plot presence and absence on India map
def plot_presence_absence(presence_df, absence_df, name, output_dir="Outputs"):
    os.makedirs(output_dir, exist_ok=True)
    m = folium.Map(location=[22, 78], zoom_start=5, tiles="OpenStreetMap")
    # Plot presence (blue)
    for _, row in presence_df.iterrows():
        folium.CircleMarker(
            location=[row['decimalLatitude'], row['decimalLongitude']],
            radius=2, color='blue', fill=True, fillColor='blue', fill_opacity=0.7
        ).add_to(m)
    # Plot absence (red)
    for _, row in absence_df.iterrows():
        folium.CircleMarker(
            location=[row['decimalLatitude'], row['decimalLongitude']],
            radius=2, color='red', fill=True, fillColor='red', fill_opacity=0.7
        ).add_to(m)
    map_path = os.path.join(output_dir, f"{name}_presence_absence_map.html")
    m.save(map_path)
    print(f"Map saved: {map_path}")

# Check for close presence-absence pairs and print details
def check_close_pairs(pres_points, absence_selected, feature_cols, threshold_km=10):
    print(f"Checking for presence-absence pairs closer than {threshold_km} km...")
    found = False
    fe = None
    for a_idx, a_row in absence_selected.iterrows():
        a_coord = (a_row['decimalLatitude'], a_row['decimalLongitude'])
        for p_idx, p_row in pres_points.iterrows():
            p_coord = (p_row['decimalLatitude'], p_row['decimalLongitude'])
            dist_km = geodesic(a_coord, p_coord).km
            if dist_km < threshold_km:
                print(f"\n[WARNING] Close pair found (<{threshold_km} km):")
                print(f"  Absence point index: {a_idx}, Presence point index: {p_idx}")
                print(f"  Distance: {dist_km:.2f} km")
                print(f"  Absence reliability: {a_row.get('reliability_score', 'N/A')}")
                print(f"  Absence features:")
                print(a_row[feature_cols])
                # Extract and print presence features using Feature_Extractor
                if fe is None:
                    try:
                        ee.Initialize()
                    except Exception:
                        pass
                    fe = Feature_Extractor(ee)
                pres_feat_df = fe.add_features(pd.DataFrame({
                    'longitude': [p_row['decimalLongitude']],
                    'latitude': [p_row['decimalLatitude']]
                }))
                pres_feat_row = pres_feat_df.iloc[0]
                pres_feat_cols = [col for col in feature_cols if col in pres_feat_row.index]
                print(f"  Presence features (extracted):")
                print(pres_feat_row[pres_feat_cols])
                found = True
                break
        if found:
            break
    mean_reliability = absence_selected['reliability_score'].mean() if 'reliability_score' in absence_selected.columns else 'N/A'
    print(f"Mean reliability score of all absences: {mean_reliability}")

# For each species
def process_species(species):
    print(f"\n=== Processing species: {species} ===")
    result = comprehensive_species_with_precomputed_features(
        species, features_csv_path=features_csv_path, reliability_threshold=0.15
    )
    if result is None:
        print(f"No data for species: {species}")
        return
    _, _, _, absence_selected, pres_clean, _ = result
    # Get original presence points for this species
    pres_points = all_presence[all_presence['species'] == species]
    # Plot
    plot_presence_absence(pres_points, absence_selected, species.replace(' ', '_'))
    # Save absence points
    absence_selected.to_csv(f"Outputs/{species.replace(' ', '_')}_absence_points.csv", index=False)
    print(f"Absence points saved: Outputs/{species.replace(' ', '_')}_absence_points.csv")
    # Check for close pairs
    # feature_cols = [col for col in absence_selected.columns if col not in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'decimalLongitude', 'decimalLatitude', 'reliability_score']]
    # check_close_pairs(pres_points, absence_selected, feature_cols)

# For each genus
def process_genus(genus):
    print(f"\n=== Processing genus: {genus} ===")
    result = comprehensive_genus_with_precomputed_features(
        genus, features_csv_path=features_csv_path, reliability_threshold=0.07
    )
    if result is None:
        print(f"No data for genus: {genus}")
        return
    _, _, _, absence_selected, pres_clean, _ = result
    # Get original presence points for this genus
    pres_points = all_presence[all_presence['genus'] == genus]
    # Plot
    plot_presence_absence(pres_points, absence_selected, genus.replace(' ', '_'))
    # Save absence points
    absence_selected.to_csv(f"Outputs/{genus.replace(' ', '_')}_absence_points.csv", index=False)
    print(f"Absence points saved: Outputs/{genus.replace(' ', '_')}_absence_points.csv")
    # Check for close pairs
    # feature_cols = [col for col in absence_selected.columns if col not in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'decimalLongitude', 'decimalLatitude', 'reliability_score']]
    # check_close_pairs(pres_points, absence_selected, feature_cols)

if __name__ == "__main__":
    for species in species_list:
        process_species(species)
    # for genus in genus_list:
    #     process_genus(genus) 