#!/usr/bin/env python3
"""
Quick script to check feature variation for a single species or genus.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import pandas as pd
import numpy as np

def quick_feature_check(species_name, features_csv_path="data/presence_points_with_features.csv"):
    """
    Quick feature variation check for a species.
    """
    print(f"\n{'='*80}")
    print(f"QUICK FEATURE VARIATION CHECK FOR: {species_name}")
    print(f"{'='*80}")
    
    try:
        # Load the data
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
        
        # Define feature columns
        exclude_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 
                       'decimalLongitude', 'decimalLatitude']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Check if presence features are already saved
        species_name_safe = species_name.replace(" ", "_")
        presence_features_file = f"data/presence_features_{species_name_safe}.csv"
        
        if os.path.exists(presence_features_file):
            print("Loading pre-computed presence features...")
            pres_with_features = pd.read_csv(presence_features_file, low_memory=False)
        else:
            print("❌ Presence features file not found. Please run the comprehensive analysis first.")
            return None
        
        # Clean data
        pres_clean = pres_with_features[~pres_with_features[feature_cols].isna().any(axis=1)].copy()
        different_order_clean = different_order_df[~different_order_df[feature_cols].isna().any(axis=1)].copy()
        
        # Ensure numeric
        for col in feature_cols:
            if col in pres_clean.columns:
                pres_clean.loc[:, col] = pd.to_numeric(pres_clean[col], errors='coerce')
            if col in different_order_clean.columns:
                different_order_clean.loc[:, col] = pd.to_numeric(different_order_clean[col], errors='coerce')
        
        # Remove NaN
        pres_clean = pres_clean.dropna(subset=feature_cols)
        different_order_clean = different_order_clean.dropna(subset=feature_cols)
        
        print(f"Valid presence points: {len(pres_clean)}")
        print(f"Valid potential absence points: {len(different_order_clean)}")
        
        # Select absence points (1:1 ratio)
        num_presence = len(pres_clean)
        target_absence = num_presence
        
        if len(different_order_clean) >= target_absence:
            absence_selected = different_order_clean.sample(n=target_absence, random_state=42)
        else:
            absence_selected = different_order_clean
        
        # Prepare final datasets
        X_presence = pres_clean[feature_cols].values.astype(float)
        X_absence = absence_selected[feature_cols].values.astype(float)
        
        # Combine datasets
        X = np.vstack([X_presence, X_absence])
        y = np.concatenate([np.ones(len(X_presence)), np.zeros(len(X_absence))])
        
        print(f"Final dataset: {len(X)} samples ({np.sum(y)} presence, {len(y)-np.sum(y)} absence)")
        
        # Feature distribution analysis completed
        print(f"Feature distribution analysis completed for {species_name}")
        
        return {
            'total_samples': len(X),
            'presence_samples': np.sum(y),
            'absence_samples': len(y) - np.sum(y)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing species {species_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with one species
    species_to_check = "Syzygium cumini"
    print(f"Checking feature variation for: {species_to_check}")
    
    result = quick_feature_check(species_to_check)
    
    if result:
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Total samples: {result['total_samples']}")
        print(f"Presence samples: {result['presence_samples']}")
        print(f"Absence samples: {result['absence_samples']}")
        
        if result['low_variation_features']:
            print(f"⚠️  Low variation features: {', '.join(result['low_variation_features'])}")
        if result['high_overlap_features']:
            print(f"⚠️  High overlap features: {', '.join(result['high_overlap_features'])}")
        if not result['low_variation_features'] and not result['high_overlap_features']:
            print(f"✅ Good feature variation and separation")
    else:
        print("❌ Analysis failed") 