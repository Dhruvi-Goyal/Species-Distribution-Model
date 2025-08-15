#!/usr/bin/env python3
"""
Script to select absence points that are contrasting to presence points in key environmental features.
"""

import pandas as pd
import numpy as np
import os

def select_contrasting_absence_points(
    species_name, 
    features_csv_path="data/presence_points_with_features.csv",
    target_absence_count=None,
    key_features=['elevation', 'annual_precipitation', 'subsoil_ph', 'annual_mean_temperature'],
    contrast_method='extreme_values'
):
    """
    Select absence points that contrast with presence points in key environmental features.
    """
    print(f"\n{'='*80}")
    print(f"SELECTING CONTRASTING ABSENCE POINTS FOR: {species_name}")
    print(f"{'='*80}")
    
    try:
        # Load data
        print("Loading data...")
        features_df = pd.read_csv(features_csv_path, low_memory=False)
        presence_df = pd.read_csv('data/testing_SDM/all_presence_point.csv', low_memory=False)
        
        # Get presence points for the species
        pres = presence_df[presence_df['species'] == species_name].copy()
        if len(pres) == 0:
            print(f"❌ No presence points found for species: {species_name}")
            return None
        
        print(f"Found {len(pres)} presence points for {species_name}")
        
        # Get the order of the target species
        target_order = pres['order'].iloc[0]
        
        # Get all points from different orders (potential absence points)
        different_order_df = features_df[features_df['order'] != target_order].copy()
        
        # Remove any points that are the same as presence points
        presence_coords = set(zip(pres['decimalLongitude'], pres['decimalLatitude']))
        different_order_df = different_order_df[~different_order_df.apply(
            lambda r: (r['decimalLongitude'], r['decimalLatitude']) in presence_coords, axis=1
        )]
        
        print(f"Found {len(different_order_df)} potential absence points from different orders")
        
        # Check which key features are available
        available_features = []
        for feature in key_features:
            if feature in different_order_df.columns:
                available_features.append(feature)
                print(f"✅ Found key feature: {feature}")
            else:
                print(f"❌ Missing key feature: {feature}")
        
        if not available_features:
            print("❌ None of the key features found in the dataset!")
            return None
        
        print(f"Using {len(available_features)} key features for contrast selection")
        
        # Clean data - remove rows with NaN in key features
        print("Cleaning data...")
        different_order_clean = different_order_df[~different_order_df[available_features].isna().any(axis=1)].copy()
        
        # Convert to numeric
        for feature in available_features:
            different_order_clean[feature] = pd.to_numeric(different_order_clean[feature], errors='coerce')
        
        # Remove any remaining NaN
        different_order_clean = different_order_clean.dropna(subset=available_features)
        
        print(f"Valid absence candidates: {len(different_order_clean)}")
        
        if len(different_order_clean) == 0:
            print("❌ No valid absence candidates after cleaning!")
            return None
        
        # Load presence features for comparison
        species_name_safe = species_name.replace(" ", "_")
        presence_features_file = f"data/presence_features_{species_name_safe}.csv"
        
        if not os.path.exists(presence_features_file):
            print("❌ Presence features file not found. Please run comprehensive analysis first.")
            return None
        
        pres_with_features = pd.read_csv(presence_features_file, low_memory=False)
        pres_clean = pres_with_features[~pres_with_features[available_features].isna().any(axis=1)].copy()
        
        for feature in available_features:
            pres_clean[feature] = pd.to_numeric(pres_clean[feature], errors='coerce')
        pres_clean = pres_clean.dropna(subset=available_features)
        
        print(f"Valid presence points: {len(pres_clean)}")
        
        # Set target absence count
        if target_absence_count is None:
            target_absence_count = len(pres_clean)
        
        print(f"Target absence points: {target_absence_count}")
        
        # Select contrasting absence points based on method
        if contrast_method == 'extreme_values':
            selected_absences = select_by_extreme_values(different_order_clean, pres_clean, available_features, target_absence_count)
        elif contrast_method == 'different_ranges':
            selected_absences = select_by_different_ranges(different_order_clean, pres_clean, available_features, target_absence_count)
        elif contrast_method == 'opposite_quartiles':
            selected_absences = select_by_opposite_quartiles(different_order_clean, pres_clean, available_features, target_absence_count)
        else:
            print(f"❌ Unknown contrast method: {contrast_method}")
            return None
        
        # Analyze the contrast of selected points
        print(f"\n{'='*60}")
        print(f"CONTRAST ANALYSIS OF SELECTED ABSENCE POINTS")
        print(f"{'='*60}")
        
        analyze_contrast(selected_absences, pres_clean, available_features)
        
        return selected_absences
        
    except Exception as e:
        print(f"❌ Error selecting contrasting absence points: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def select_by_extreme_values(absence_df, presence_df, features, target_count):
    """
    Select absence points that have extreme values compared to presence points.
    """
    print(f"Using extreme values method to select contrasting points...")
    
    # Calculate presence point statistics for each feature
    presence_stats = {}
    for feature in features:
        presence_stats[feature] = {
            'mean': presence_df[feature].mean(),
            'std': presence_df[feature].std(),
            'min': presence_df[feature].min(),
            'max': presence_df[feature].max(),
            'q25': presence_df[feature].quantile(0.25),
            'q75': presence_df[feature].quantile(0.75)
        }
    
    # Create masks for extreme values (outside 2 standard deviations from presence mean)
    extreme_masks = []
    for feature in features:
        mean_val = presence_stats[feature]['mean']
        std_val = presence_stats[feature]['std']
        
        # Points that are very different from presence points
        low_extreme = absence_df[feature] < (mean_val - 2 * std_val)
        high_extreme = absence_df[feature] > (mean_val + 2 * std_val)
        extreme_mask = low_extreme | high_extreme
        extreme_masks.append(extreme_mask)
    
    # Combine masks - select points that are extreme in at least one feature
    combined_mask = extreme_masks[0]
    for mask in extreme_masks[1:]:
        combined_mask = combined_mask | mask
    
    extreme_candidates = absence_df[combined_mask].copy()
    
    print(f"Found {len(extreme_candidates)} extreme candidates")
    
    if len(extreme_candidates) >= target_count:
        selected_points = extreme_candidates.sample(n=target_count, random_state=42)
    else:
        # If not enough extreme points, add some moderate contrast points
        remaining_needed = target_count - len(extreme_candidates)
        
        # Select points that are moderately different (outside 1 standard deviation)
        moderate_masks = []
        for feature in features:
            mean_val = presence_stats[feature]['mean']
            std_val = presence_stats[feature]['std']
            
            low_moderate = absence_df[feature] < (mean_val - std_val)
            high_moderate = absence_df[feature] > (mean_val + std_val)
            moderate_mask = (low_moderate | high_moderate) & ~combined_mask
            moderate_masks.append(moderate_mask)
        
        combined_moderate_mask = moderate_masks[0]
        for mask in moderate_masks[1:]:
            combined_moderate_mask = combined_moderate_mask | mask
        
        moderate_candidates = absence_df[combined_moderate_mask].copy()
        
        if len(moderate_candidates) >= remaining_needed:
            additional_points = moderate_candidates.sample(n=remaining_needed, random_state=42)
        else:
            additional_points = moderate_candidates
        
        selected_points = pd.concat([extreme_candidates, additional_points])
    
    print(f"Selected {len(selected_points)} points using extreme values method")
    return selected_points.head(target_count)

def analyze_contrast(absence_df, presence_df, features):
    """
    Analyze the contrast between selected absence points and presence points.
    """
    print(f"\nContrast analysis:")
    print(f"Absence points: {len(absence_df)}")
    print(f"Presence points: {len(presence_df)}")
    
    for feature in features:
        print(f"\n{feature}:")
        
        # Calculate ranges
        pres_range = presence_df[feature].max() - presence_df[feature].min()
        abs_range = absence_df[feature].max() - absence_df[feature].min()
        
        # Calculate overlap
        overlap_min = max(presence_df[feature].min(), absence_df[feature].min())
        overlap_max = min(presence_df[feature].max(), absence_df[feature].max())
        overlap_range = max(0, overlap_max - overlap_min)
        total_range = max(presence_df[feature].max(), absence_df[feature].max()) - min(presence_df[feature].min(), absence_df[feature].min())
        overlap_ratio = overlap_range / total_range if total_range > 0 else 0
        
        # Calculate means
        pres_mean = presence_df[feature].mean()
        abs_mean = absence_df[feature].mean()
        mean_diff = abs(abs_mean - pres_mean)
        
        print(f"  Presence range: {pres_range:.4f}")
        print(f"  Absence range: {abs_range:.4f}")
        print(f"  Overlap ratio: {overlap_ratio:.1%}")
        print(f"  Mean difference: {mean_diff:.4f}")
        
        if overlap_ratio > 0.9:
            print(f"  ⚠️  HIGH OVERLAP: {overlap_ratio:.1%}")
        elif overlap_ratio > 0.7:
            print(f"  ⚠️  MODERATE OVERLAP: {overlap_ratio:.1%}")
        else:
            print(f"  ✅ GOOD SEPARATION: {overlap_ratio:.1%}")

if __name__ == "__main__":
    # Example usage
    species_to_analyze = "Syzygium cumini"
    
    print("Contrasting Absence Point Selection Tool")
    print("=" * 50)
    
    # Select contrasting absence points
    contrasting_absences = select_contrasting_absence_points(
        species_name=species_to_analyze,
        contrast_method='extreme_values',
        key_features=['elevation', 'annual_precipitation', 'subsoil_ph', 'annual_mean_temperature']
    )
    
    if contrasting_absences is not None:
        print(f"\n✅ Successfully selected {len(contrasting_absences)} contrasting absence points")
    else:
        print(f"\n❌ Failed to select contrasting absence points") 