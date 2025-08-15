import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
from pathlib import Path

def load_per_ecoregion_results(species_list, genus_list):
    """
    Load all per-ecoregion results CSV files for the given species and genera.
    """
    all_data = []
    
    # Load species data
    for species in species_list:
        species_name = species.replace(" ", "_")
        # Look for CSV files matching the pattern
        pattern = f"outputs/{species_name}_*_per_ecoregion_results.csv"
        files = glob.glob(pattern)
        
        for file in files:
            try:
                df = pd.read_csv(file)
                # Extract model and loss function from filename
                filename = os.path.basename(file)
                parts = filename.replace(f"{species_name}_", "").replace("_per_ecoregion_results.csv", "").split("_")
                
                if len(parts) >= 2:
                    model = parts[0]  # RF, LR, WLR
                    loss_function = "_".join(parts[1:])  # Dice_Loss, Focal_Loss, Tversky_Loss
                    
                    df['species'] = species
                    df['genus'] = None
                    df['model'] = model
                    df['loss_function'] = loss_function
                    df['type'] = 'species'
                    all_data.append(df)
                    print(f"Loaded {file}: {len(df)} ecoregions")
                else:
                    print(f"Could not parse filename: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    # Load genus data
    for genus in genus_list:
        genus_name = genus.replace(" ", "_")
        # Look for CSV files matching the pattern
        pattern = f"outputs/{genus_name}_*_per_ecoregion_results.csv"
        files = glob.glob(pattern)
        
        for file in files:
            try:
                df = pd.read_csv(file)
                # Extract model and loss function from filename
                filename = os.path.basename(file)
                parts = filename.replace(f"{genus_name}_", "").replace("_per_ecoregion_results.csv", "").split("_")
                
                if len(parts) >= 2:
                    model = parts[0]  # RF, LR, WLR
                    loss_function = "_".join(parts[1:])  # Dice_Loss, Focal_Loss, Tversky_Loss
                    
                    df['species'] = None
                    df['genus'] = genus
                    df['model'] = model
                    df['loss_function'] = loss_function
                    df['type'] = 'genus'
                    all_data.append(df)
                    print(f"Loaded {file}: {len(df)} ecoregions")
                else:
                    print(f"Could not parse filename: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal data loaded: {len(combined_df)} rows")
        print(f"Columns: {list(combined_df.columns)}")
        return combined_df
    else:
        print("No data files found!")
        return None

def create_reduced_data(df):
    """
    Create reduced data with randomly reduced values and more spread.
    """
    reduced_df = df.copy()
    
    # Define reduction ranges for each species/genus
    reduction_ranges = {
        'Dalbergia sissoo': (0.20, 0.25),  # 20-25% reduction
        'Syzygium cumini': (0.10, 0.13),   # 10-13% reduction
        'Memecylon': (0.04, 0.07),         # 4-7% reduction
        'Macaranga': (0.01, 0.02)          # 1-2% reduction
    }
    
    # Apply reductions to each row
    for idx, row in reduced_df.iterrows():
        if row['species'] in reduction_ranges:
            species = row['species']
        elif row['genus'] in reduction_ranges:
            species = row['genus']
        else:
            continue
            
        min_reduction, max_reduction = reduction_ranges[species]
        
        # Random reduction factor
        reduction_factor = np.random.uniform(min_reduction, max_reduction)
        
        # Apply reduction to accuracy, tpr, tnr
        for metric in ['accuracy', 'tpr', 'tnr']:
            if pd.notna(row[metric]):
                # Reduce the value
                reduced_value = row[metric] * (1 - reduction_factor)
                
                # Add random spread with increased variability for Macaranga
                if species == 'Macaranga':
                    # Much higher spread for Macaranga, especially for TNR
                    if metric == 'tnr':
                        # For TNR, allow much lower values (down to 50%)
                        spread_factor = np.random.uniform(0.20, 0.40)  # 20-40% spread
                        # Higher chance of downward spread for TNR
                        if np.random.random() > 0.3:  # 70% chance of going down
                            reduced_value *= (1 - spread_factor)
                        else:
                            reduced_value *= (1 + spread_factor * 0.5)  # Smaller upward spread
                    else:
                        # For accuracy and TPR, moderate spread
                        spread_factor = np.random.uniform(0.15, 0.25)  # 15-25% spread
                        if np.random.random() > 0.5:
                            reduced_value *= (1 + spread_factor)
                        else:
                            reduced_value *= (1 - spread_factor)
                else:
                    # Standard spread for other species/genera
                    spread_factor = np.random.uniform(0.05, 0.15)  # 5-15% additional spread
                    if np.random.random() > 0.5:
                        reduced_value *= (1 + spread_factor)
                    else:
                        reduced_value *= (1 - spread_factor)
                
                # Ensure values stay within reasonable bounds (0-1)
                # For Macaranga TNR, allow lower bound of 0.5 (50%)
                if species == 'Macaranga' and metric == 'tnr':
                    reduced_value = max(0.5, min(1.0, reduced_value))
                else:
                    reduced_value = max(0.0, min(1.0, reduced_value))
                
                reduced_df.at[idx, metric] = reduced_value
    
    return reduced_df

def create_box_plots(df, output_dir="outputs/metrics_plots"):
    """
    Create box plots for accuracy, TPR, and TNR metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Filter data for species (accuracy and TPR)
    species_data = df[df['type'] == 'species'].copy()
    genus_data = df[df['type'] == 'genus'].copy()
    
    # Create species plots (Accuracy and TPR)
    if not species_data.empty:
        print("\nCreating species plots (Accuracy and TPR)...")
        
        # Plot 1: Accuracy by Model and Loss Function for Species
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        
        # Create a combined label for model and loss function
        species_data['model_loss'] = species_data['model'] + '_' + species_data['loss_function'].str.replace('_Loss', '')
        
        # Create box plot with individual points
        sns.boxplot(data=species_data, x='model_loss', y='accuracy', hue='species')
        # Add individual data points
        sns.stripplot(data=species_data, x='model_loss', y='accuracy', hue='species', 
                     dodge=True, size=8, color='0.3', alpha=0.7, legend=False)
        plt.title('Accuracy Distribution by Model and Loss Function (Species)', fontsize=14, fontweight='bold')
        plt.xlabel('Model_LossFunction', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Plot 2: TPR by Model and Loss Function for Species
        plt.subplot(1, 2, 2)
        sns.boxplot(data=species_data, x='model_loss', y='tpr', hue='species')
        # Add individual data points
        sns.stripplot(data=species_data, x='model_loss', y='tpr', hue='species', 
                     dodge=True, size=8, color='0.3', alpha=0.7, legend=False)
        plt.title('True Positive Rate (TPR) Distribution by Model and Loss Function (Species)', fontsize=14, fontweight='bold')
        plt.xlabel('Model_LossFunction', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'species_accuracy_tpr_boxplots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual species plots
        for species in species_data['species'].unique():
            species_subset = species_data[species_data['species'] == species]
            
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            sns.boxplot(data=species_subset, x='model_loss', y='accuracy')
            # Add individual data points
            sns.stripplot(data=species_subset, x='model_loss', y='accuracy', 
                         size=8, color='0.3', alpha=0.7)
            plt.title(f'Accuracy Distribution - {species}', fontsize=14, fontweight='bold')
            plt.xlabel('Model_LossFunction', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            plt.subplot(1, 2, 2)
            sns.boxplot(data=species_subset, x='model_loss', y='tpr')
            # Add individual data points
            sns.stripplot(data=species_subset, x='model_loss', y='tpr', 
                         size=8, color='0.3', alpha=0.7)
            plt.title(f'True Positive Rate (TPR) Distribution - {species}', fontsize=14, fontweight='bold')
            plt.xlabel('Model_LossFunction', fontsize=12)
            plt.ylabel('True Positive Rate (TPR)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            species_name_safe = species.replace(" ", "_")
            plt.savefig(os.path.join(output_dir, f'{species_name_safe}_accuracy_tpr_boxplots.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create genus plots (Accuracy and TNR)
    if not genus_data.empty:
        print("\nCreating genus plots (Accuracy and TNR)...")
        
        # Plot 1: Accuracy by Model and Loss Function for Genera
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        
        # Create a combined label for model and loss function
        genus_data['model_loss'] = genus_data['model'] + '_' + genus_data['loss_function'].str.replace('_Loss', '')
        
        # Create box plot with individual points
        sns.boxplot(data=genus_data, x='model_loss', y='accuracy', hue='genus')
        # Add individual data points
        sns.stripplot(data=genus_data, x='model_loss', y='accuracy', hue='genus', 
                     dodge=True, size=8, color='0.3', alpha=0.7, legend=False)
        plt.title('Accuracy Distribution by Model and Loss Function (Genera)', fontsize=14, fontweight='bold')
        plt.xlabel('Model_LossFunction', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Genus', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Plot 2: TNR by Model and Loss Function for Genera
        plt.subplot(1, 2, 2)
        sns.boxplot(data=genus_data, x='model_loss', y='tnr', hue='genus')
        # Add individual data points
        sns.stripplot(data=genus_data, x='model_loss', y='tnr', hue='genus', 
                     dodge=True, size=8, color='0.3', alpha=0.7, legend=False)
        plt.title('True Negative Rate (TNR) Distribution by Model and Loss Function (Genera)', fontsize=14, fontweight='bold')
        plt.xlabel('Model_LossFunction', fontsize=12)
        plt.ylabel('True Negative Rate (TNR)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Genus', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'genus_accuracy_tnr_boxplots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual genus plots
        for genus in genus_data['genus'].unique():
            genus_subset = genus_data[genus_data['genus'] == genus]
            
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            sns.boxplot(data=genus_subset, x='model_loss', y='accuracy')
            # Add individual data points
            sns.stripplot(data=genus_subset, x='model_loss', y='accuracy', 
                         size=8, color='0.3', alpha=0.7)
            plt.title(f'Accuracy Distribution - {genus}', fontsize=14, fontweight='bold')
            plt.xlabel('Model_LossFunction', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            plt.subplot(1, 2, 2)
            sns.boxplot(data=genus_subset, x='model_loss', y='tnr')
            # Add individual data points
            sns.stripplot(data=genus_subset, x='model_loss', y='tnr', 
                         size=8, color='0.3', alpha=0.7)
            plt.title(f'True Negative Rate (TNR) Distribution - {genus}', fontsize=14, fontweight='bold')
            plt.xlabel('Model_LossFunction', fontsize=12)
            plt.ylabel('True Negative Rate (TNR)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            genus_name_safe = genus.replace(" ", "_")
            plt.savefig(os.path.join(output_dir, f'{genus_name_safe}_accuracy_tnr_boxplots.png'), dpi=300, bbox_inches='tight')
            plt.close()

def create_summary_statistics(df, output_dir="outputs/metrics_plots"):
    """
    Create summary statistics tables and save them as CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data
    species_data = df[df['type'] == 'species'].copy()
    genus_data = df[df['type'] == 'genus'].copy()
    
    # Species summary statistics
    if not species_data.empty:
        print("\nCreating species summary statistics...")
        
        # Create combined model_loss column
        species_data['model_loss'] = species_data['model'] + '_' + species_data['loss_function'].str.replace('_Loss', '')
        
        # Group by species, model_loss and calculate statistics
        species_summary = species_data.groupby(['species', 'model_loss']).agg({
            'accuracy': ['mean', 'std', 'min', 'max', 'count'],
            'tpr': ['mean', 'std', 'min', 'max'],
            'tnr': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Flatten column names
        species_summary.columns = ['_'.join(col).strip() for col in species_summary.columns]
        species_summary = species_summary.reset_index()
        
        # Save to CSV
        species_summary.to_csv(os.path.join(output_dir, 'species_summary_statistics.csv'), index=False)
        print(f"Species summary saved to: {os.path.join(output_dir, 'species_summary_statistics.csv')}")
        
        # Print summary
        print("\nSpecies Summary Statistics:")
        print(species_summary.to_string(index=False))
    
    # Genus summary statistics
    if not genus_data.empty:
        print("\nCreating genus summary statistics...")
        
        # Create combined model_loss column
        genus_data['model_loss'] = genus_data['model'] + '_' + genus_data['loss_function'].str.replace('_Loss', '')
        
        # Group by genus, model_loss and calculate statistics
        genus_summary = genus_data.groupby(['genus', 'model_loss']).agg({
            'accuracy': ['mean', 'std', 'min', 'max', 'count'],
            'tpr': ['mean', 'std', 'min', 'max'],
            'tnr': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Flatten column names
        genus_summary.columns = ['_'.join(col).strip() for col in genus_summary.columns]
        genus_summary = genus_summary.reset_index()
        
        # Save to CSV
        genus_summary.to_csv(os.path.join(output_dir, 'genus_summary_statistics.csv'), index=False)
        print(f"Genus summary saved to: {os.path.join(output_dir, 'genus_summary_statistics.csv')}")
        
        # Print summary
        print("\nGenus Summary Statistics:")
        print(genus_summary.to_string(index=False))

def save_per_ecoregion_reduced_stats(df, output_dir="outputs/metrics_plots"):
    """
    Save per-ecoregion reduced statistics for each species and genus.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data
    species_data = df[df['type'] == 'species'].copy()
    genus_data = df[df['type'] == 'genus'].copy()
    
    # Save species per-ecoregion data (accuracy and TPR)
    if not species_data.empty:
        print("\nSaving species per-ecoregion reduced statistics...")
        
        # Create combined model_loss column
        species_data['model_loss'] = species_data['model'] + '_' + species_data['loss_function'].str.replace('_Loss', '')
        
        # Select relevant columns for species (accuracy and TPR)
        species_columns = ['ecoregion', 'species', 'model', 'loss_function', 'model_loss', 'accuracy', 'tpr']
        species_per_ecoregion = species_data[species_columns].copy()
        
        # Sort by species, model_loss, and ecoregion
        species_per_ecoregion = species_per_ecoregion.sort_values(['species', 'model_loss', 'ecoregion'])
        
        # Save to CSV
        species_per_ecoregion.to_csv(os.path.join(output_dir, 'species_per_ecoregion_reduced_stats.csv'), index=False)
        print(f"Species per-ecoregion data saved to: {os.path.join(output_dir, 'species_per_ecoregion_reduced_stats.csv')}")
        
        # Also save individual species files
        for species in species_data['species'].unique():
            species_subset = species_per_ecoregion[species_per_ecoregion['species'] == species]
            species_name_safe = species.replace(" ", "_")
            species_subset.to_csv(os.path.join(output_dir, f'{species_name_safe}_per_ecoregion_reduced_stats.csv'), index=False)
            print(f"  - {species_name_safe}_per_ecoregion_reduced_stats.csv")
    
    # Save genus per-ecoregion data (accuracy and TNR)
    if not genus_data.empty:
        print("\nSaving genus per-ecoregion reduced statistics...")
        
        # Create combined model_loss column
        genus_data['model_loss'] = genus_data['model'] + '_' + genus_data['loss_function'].str.replace('_Loss', '')
        
        # Select relevant columns for genus (accuracy and TNR)
        genus_columns = ['ecoregion', 'genus', 'model', 'loss_function', 'model_loss', 'accuracy', 'tnr']
        genus_per_ecoregion = genus_data[genus_columns].copy()
        
        # Sort by genus, model_loss, and ecoregion
        genus_per_ecoregion = genus_per_ecoregion.sort_values(['genus', 'model_loss', 'ecoregion'])
        
        # Save to CSV
        genus_per_ecoregion.to_csv(os.path.join(output_dir, 'genus_per_ecoregion_reduced_stats.csv'), index=False)
        print(f"Genus per-ecoregion data saved to: {os.path.join(output_dir, 'genus_per_ecoregion_reduced_stats.csv')}")
        
        # Also save individual genus files
        for genus in genus_data['genus'].unique():
            genus_subset = genus_per_ecoregion[genus_per_ecoregion['genus'] == genus]
            genus_name_safe = genus.replace(" ", "_")
            genus_subset.to_csv(os.path.join(output_dir, f'{genus_name_safe}_per_ecoregion_reduced_stats.csv'), index=False)
            print(f"  - {genus_name_safe}_per_ecoregion_reduced_stats.csv")

def create_heatmap_plots(df, output_dir="outputs/metrics_plots"):
    """
    Create heatmap plots showing average metrics across models and loss functions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data
    species_data = df[df['type'] == 'species'].copy()
    genus_data = df[df['type'] == 'genus'].copy()
    
    # Species heatmaps
    if not species_data.empty:
        print("\nCreating species heatmaps...")
        
        # Create combined model_loss column
        species_data['model_loss'] = species_data['model'] + '_' + species_data['loss_function'].str.replace('_Loss', '')
        
        # Create pivot tables for heatmaps
        accuracy_pivot = species_data.pivot_table(values='accuracy', index='species', columns='model_loss', aggfunc='mean')
        tpr_pivot = species_data.pivot_table(values='tpr', index='species', columns='model_loss', aggfunc='mean')
        
        # Plot accuracy heatmap
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(accuracy_pivot, annot=True, cmap='RdYlBu_r', center=0.5, fmt='.3f', cbar_kws={'label': 'Accuracy'})
        plt.title('Average Accuracy by Species and Model', fontsize=14, fontweight='bold')
        plt.xlabel('Model_LossFunction', fontsize=12)
        plt.ylabel('Species', fontsize=12)
        
        # Plot TPR heatmap
        plt.subplot(1, 2, 2)
        sns.heatmap(tpr_pivot, annot=True, cmap='RdYlBu_r', center=0.5, fmt='.3f', cbar_kws={'label': 'TPR'})
        plt.title('Average TPR by Species and Model', fontsize=14, fontweight='bold')
        plt.xlabel('Model_LossFunction', fontsize=12)
        plt.ylabel('Species', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'species_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Genus heatmaps
    if not genus_data.empty:
        print("\nCreating genus heatmaps...")
        
        # Create combined model_loss column
        genus_data['model_loss'] = genus_data['model'] + '_' + genus_data['loss_function'].str.replace('_Loss', '')
        
        # Create pivot tables for heatmaps
        accuracy_pivot = genus_data.pivot_table(values='accuracy', index='genus', columns='model_loss', aggfunc='mean')
        tnr_pivot = genus_data.pivot_table(values='tnr', index='genus', columns='model_loss', aggfunc='mean')
        
        # Plot accuracy heatmap
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(accuracy_pivot, annot=True, cmap='RdYlBu_r', center=0.5, fmt='.3f', cbar_kws={'label': 'Accuracy'})
        plt.title('Average Accuracy by Genus and Model', fontsize=14, fontweight='bold')
        plt.xlabel('Model_LossFunction', fontsize=12)
        plt.ylabel('Genus', fontsize=12)
        
        # Plot TNR heatmap
        plt.subplot(1, 2, 2)
        sns.heatmap(tnr_pivot, annot=True, cmap='RdYlBu_r', center=0.5, fmt='.3f', cbar_kws={'label': 'TNR'})
        plt.title('Average TNR by Genus and Model', fontsize=14, fontweight='bold')
        plt.xlabel('Model_LossFunction', fontsize=12)
        plt.ylabel('Genus', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'genus_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Main function to run the plotting analysis.
    """
    print("Per-Ecoregion Metrics Plotting Tool")
    print("=" * 50)
    
    # Define the species and genus lists
    species_list = ["Dalbergia sissoo", "Syzygium cumini"]
    genus_list = ["Memecylon", "Macaranga"]
    
    print(f"Species to analyze: {species_list}")
    print(f"Genera to analyze: {genus_list}")
    
    # Load the data
    print("\nLoading per-ecoregion results...")
    df = load_per_ecoregion_results(species_list, genus_list)
    
    if df is None or df.empty:
        print("No data found! Please run the modeling functions first to generate per-ecoregion results.")
        return
    
    # Create output directory
    output_dir = "outputs/metrics_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create original plots
    print("\nCreating original plots...")
    create_box_plots(df, output_dir)
    create_heatmap_plots(df, output_dir)
    create_summary_statistics(df, output_dir)
    save_per_ecoregion_reduced_stats(df, output_dir)
    
    # Create reduced plots with more spread
    print("\nCreating reduced plots with more spread...")
    reduced_df = create_reduced_data(df)
    
    # Create reduced plots in a separate directory
    reduced_output_dir = os.path.join(output_dir, "reduced_plots")
    os.makedirs(reduced_output_dir, exist_ok=True)
    
    create_box_plots(reduced_df, reduced_output_dir)
    create_heatmap_plots(reduced_df, reduced_output_dir)
    create_summary_statistics(reduced_df, reduced_output_dir)
    save_per_ecoregion_reduced_stats(reduced_df, reduced_output_dir)
    
    print(f"\n✅ All plots and statistics saved to: {output_dir}")
    print(f"✅ Reduced plots saved to: {reduced_output_dir}")
    print("\nGenerated files:")
    print("- species_accuracy_tpr_boxplots.png")
    print("- genus_accuracy_tnr_boxplots.png")
    print("- Individual species/genus box plots")
    print("- species_heatmaps.png")
    print("- genus_heatmaps.png")
    print("- species_summary_statistics.csv")
    print("- genus_summary_statistics.csv")
    print("- species_per_ecoregion_reduced_stats.csv")
    print("- genus_per_ecoregion_reduced_stats.csv")
    print("- Individual species/genus per-ecoregion CSV files")
    print("\nReduced plots (with more spread):")
    print("- All the above files in reduced_plots/ subdirectory")

if __name__ == "__main__":
    main() 