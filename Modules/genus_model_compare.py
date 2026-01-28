import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist
import ee
from Modules.features_extractor import Feature_Extractor

# Import model/loss configs from custom_loss_trainers
from Modules.custom_loss_trainers import model_names, loss_cfgs, train_funcs, calculate_feature_based_reliability

def calculate_reliability(presence_points_df, absence_point_dict, feature_cols):
    # Select only the feature columns, in the correct order, for presence and absence
    presence_features = presence_points_df[feature_cols].values
    try:
        absence_point_features = [float(absence_point_dict[col]) for col in feature_cols]
    except KeyError:
        # If any feature is missing, return None or np.nan
        return None
    if any(pd.isnull(absence_point_features)):
        return None
    distances = cdist([absence_point_features], presence_features, metric='euclidean')
    similarities = np.exp(-distances**2 / (2 * presence_features.shape[1]))
    mean_similarity = np.nanmean(similarities)
    reliability_value = 1 - mean_similarity
    return reliability_value

def main():
    ee.Initialize()  # Ensure Earth Engine is initialized before any EE operations
    genus = "Dalbergia"
    # Load presence points for the genus from all_presence_point.csv (no features)
    all_presence = pd.read_csv('data/testing_SDM/all_presence_point.csv')
    presence_df = all_presence[all_presence['genus'] == genus].copy()

    # Load candidate absence points (with features)
    absence_candidates = pd.read_csv('data/presence_points_with_features.csv')

    # Prepare feature columns (as per your list)
    requested_feature_cols = [
        'annual_mean_temperature', 'mean_diurnal_range', 'isothermality', 'temperature_seasonality',
        'max_temperature_warmest_month', 'min_temperature_coldest_month', 'temperature_annual_range',
        'mean_temperature_wettest_quarter', 'mean_temperature_driest_quarter', 'mean_temperature_warmest_quarter',
        'mean_temperature_coldest_quarter', 'annual_precipitation', 'precipitation_wettest_month',
        'precipitation_driest_month', 'precipitation_seasonality', 'precipitation_wettest_quarter',
        'precipitation_driest_quarter', 'precipitation_warmest_quarter', 'precipitation_coldest_quarter',
        'aridity_index', 'topsoil_ph', 'subsoil_ph', 'topsoil_texture', 'subsoil_texture', 'elevation'
    ]
    # Only use features present in both presence_df and absence_candidates
    feature_cols = [col for col in requested_feature_cols if col in presence_df.columns and col in absence_candidates.columns]
    missing_features = [col for col in requested_feature_cols if col not in feature_cols]
    if missing_features:
        print(f"Warning: The following features are missing in the data and will be skipped: {missing_features}")

    # Add caching for absence points
    absence_cache_file = f"data/filtered_absence_points_{genus.replace(' ', '_')}.csv"
    if os.path.exists(absence_cache_file):
        absence_df = pd.read_csv(absence_cache_file)
        print(f"Loaded cached filtered absence points from {absence_cache_file}")
    else:
        # Filter absence candidates to those not in the presence set (by coordinates)
        presence_coords = set(zip(presence_df['decimalLongitude'], presence_df['decimalLatitude']))
        absence_candidates = absence_candidates[
            ~absence_candidates.apply(lambda row: (row['decimalLongitude'], row['decimalLatitude']) in presence_coords, axis=1)
        ]

        # Calculate reliability for each candidate absence point
        reliable_absences = []
        for idx, row in absence_candidates.iterrows():
            absence_point_dict = {col: row[col] for col in feature_cols}
            reliability_value = calculate_reliability(presence_df, absence_point_dict, feature_cols)
            if reliability_value is not None and reliability_value > 0.07:
                reliable_absences.append(row)

        # Convert to DataFrame and assign label 0
        absence_df = pd.DataFrame(reliable_absences)
        absence_df['label'] = 0

        # Optionally, sample to match the number of presence points
        absence_df = absence_df.sample(n=len(presence_df), random_state=42) if len(absence_df) > len(presence_df) else absence_df
        
        absence_df.to_csv(absence_cache_file, index=False)
        print(f"Generated and cached filtered absence points to {absence_cache_file}")

    # Prepare presence_df for modeling (add label 1 and dummy reliability)
    presence_df['label'] = 1
    presence_df['reliability'] = 1.0

    # Now continue with the rest of your pipeline (concatenation, feature selection, etc.)

    # Exclude taxonomic, coordinate, and reliability columns
    exclude_cols = [
        'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species',
        'decimalLongitude', 'decimalLatitude', 'longitude', 'latitude', 'label', 'reliability'
    ]
    feature_cols = [col for col in all_df.columns if col not in exclude_cols]

    print("Feature columns:", feature_cols)
    print("First few rows of all_df:")
    print(all_df.head())

    # Check for overlap between presence and absence points
    pres_coords = set(zip(presence_df['longitude'], presence_df['latitude']))
    abs_coords = set(zip(absence_df['longitude'], absence_df['latitude']))
    overlap_pres_abs = pres_coords & abs_coords
    print(f"Number of overlapping points between presence and absence: {len(overlap_pres_abs)}")

    # Visualize feature distributions for presence vs. absence
    plot_dir = 'outputs/feature_distributions'
    os.makedirs(plot_dir, exist_ok=True)
    for col in feature_cols:
        plt.figure(figsize=(6,4))
        plt.hist(presence_df[col].dropna(), bins=20, alpha=0.5, label='Presence', color='blue', density=True)
        plt.hist(absence_df[col].dropna(), bins=20, alpha=0.5, label='Absence', color='red', density=True)
        plt.title(f'Feature: {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{col}_hist.png'))
        plt.close()
    print(f'Feature distribution plots saved to {plot_dir}')

    # Prepare X, y
    X = all_df[feature_cols].apply(pd.to_numeric, errors='coerce').values
    y = all_df["label"].values

    # Remove rows with NaN in features
    valid_mask = ~np.isnan(X).any(axis=1)
    print("Number of valid rows after NaN filtering:", np.sum(valid_mask))
    X = X[valid_mask]
    y = y[valid_mask]
    valid_df = all_df.loc[valid_mask].reset_index(drop=True)

    # Prepare sample weights (as in custom_loss_trainers.py)
    reliability = valid_df['reliability'].values
    sample_weights = np.ones_like(y, dtype=float)
    absence_mask = (y == 0)
    absence_reliability = reliability[absence_mask]
    power_transform = 1.0  # Change this if you want to use a different power
    if power_transform != 1.0:
        absence_reliability = np.power(absence_reliability, power_transform)
    if len(absence_reliability) > 0:
        min_w = np.min(absence_reliability)
        max_w = np.max(absence_reliability)
        if max_w != min_w:
            absence_reliability = (absence_reliability - min_w) / (max_w - min_w)
        else:
            absence_reliability = np.ones_like(absence_reliability)
    sample_weights[absence_mask] = absence_reliability

    # Split
    X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42, stratify=y)
    train_indices = set(valid_df.index[:len(X_tr)])
    test_indices = set(valid_df.index[len(X_tr):])
    train_coords = set(zip(valid_df.loc[list(train_indices), 'longitude'], valid_df.loc[list(train_indices), 'latitude']))
    test_coords = set(zip(valid_df.loc[list(test_indices), 'longitude'], valid_df.loc[list(test_indices), 'latitude']))
    overlap_train_test = train_coords & test_coords
    print(f"Number of overlapping points between train and test: {len(overlap_train_test)}")

    # --- Model Results with True Labels ---
    print(f"\nGenus: {genus}")
    print(f"Presence points: {np.sum(y)} | Absence points: {len(y) - np.sum(y)} | Features: {len(feature_cols)}")
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")
    print("\nModel Results (True Labels):")
    print("{:<25} {:<12} {:<8} {:<8} {:<8} {:<8}".format('Model+Loss', 'Accuracy', 'F1', 'Prec', 'Recall', 'Support'))

    for model_name in model_names:
        for loss_name, loss_type, loss_params in loss_cfgs:
            # Only pass sample weights to models that support it (RF, WLR)
            if model_name in ['RF', 'WLR']:
                clf = train_funcs[model_name](X_tr, y_tr, sw_tr, loss_type, loss_params)
            else:
                clf = train_funcs[model_name](X_tr, y_tr, None, loss_type, loss_params)
            y_pred = clf.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred)
            rec = recall_score(y_te, y_pred)
            support = np.sum(y_te)
            print("{:<25} {:<12.3f} {:<8.3f} {:<8.3f} {:<8.3f} {:<8}".format(
                f"{model_name}+{loss_name}", acc, f1, prec, rec, support))

    # --- Sanity Check: Shuffle Labels ---
    print("\nSanity check: Shuffling labels and retraining...")
    y_shuffled = y.copy()
    np.random.shuffle(y_shuffled)
    X_tr_s, X_te_s, y_tr_s, y_te_s, sw_tr_s, sw_te_s = train_test_split(X, y_shuffled, sample_weights, test_size=0.2, random_state=42, stratify=y_shuffled)
    print(f"\nModel Results (Shuffled Labels):")
    print("{:<25} {:<12} {:<8} {:<8} {:<8} {:<8}".format('Model+Loss', 'Accuracy', 'F1', 'Prec', 'Recall', 'Support'))
    for model_name in model_names:
        for loss_name, loss_type, loss_params in loss_cfgs:
            if model_name in ['RF', 'WLR']:
                clf = train_funcs[model_name](X_tr_s, y_tr_s, sw_tr_s, loss_type, loss_params)
            else:
                clf = train_funcs[model_name](X_tr_s, y_tr_s, None, loss_type, loss_params)
            y_pred = clf.predict(X_te_s)
            acc = accuracy_score(y_te_s, y_pred)
            f1 = f1_score(y_te_s, y_pred)
            prec = precision_score(y_te_s, y_pred)
            rec = recall_score(y_te_s, y_pred)
            support = np.sum(y_te_s)
            print("{:<25} {:<12.3f} {:<8.3f} {:<8.3f} {:<8.3f} {:<8}".format(
                f"{model_name}+{loss_name}", acc, f1, prec, rec, support))

if __name__ == "__main__":
    main() 