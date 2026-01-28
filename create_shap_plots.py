import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_shap_plots(csv_path, output_dir="shap_plots"):
    """
    Create various SHAP plots from importance scores CSV file
    
    Args:
        csv_path: Path to the SHAP importance scores CSV file
        output_dir: Directory to save the plots
    """
    # Read the SHAP importance scores
    df = pd.read_csv(csv_path)
    
    # Sort by importance (descending)
    df = df.sort_values('SHAP_Importance', ascending=True)
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Horizontal Bar Plot (most common SHAP visualization)
    plt.figure(figsize=(12, 10))
    bars = plt.barh(df['Feature'], df['SHAP_Importance'])
    
    # Color bars based on importance
    colors = plt.cm.viridis(df['SHAP_Importance'] / df['SHAP_Importance'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('SHAP Importance')
    plt.title('SHAP Feature Importance Scores')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_importance_barplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Top 10 features bar plot
    top_10 = df.tail(10)  # Get top 10 features
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_10['Feature'], top_10['SHAP_Importance'])
    
    # Color bars
    colors = plt.cm.viridis(top_10['SHAP_Importance'] / top_10['SHAP_Importance'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('SHAP Importance')
    plt.title('Top 10 SHAP Feature Importance Scores')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_top10_barplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Pie chart for top 10 features
    plt.figure(figsize=(12, 8))
    top_10_pie = top_10.copy()
    # Add "Others" category for remaining features
    others_importance = df.head(len(df)-10)['SHAP_Importance'].sum()
    if others_importance > 0:
        top_10_pie = pd.concat([top_10_pie, 
                               pd.DataFrame({'Feature': ['Others'], 
                                           'SHAP_Importance': [others_importance]})])
    
    plt.pie(top_10_pie['SHAP_Importance'], labels=top_10_pie['Feature'], 
            autopct='%1.1f%%', startangle=90)
    plt.title('SHAP Feature Importance Distribution')
    plt.axis('equal')
    plt.savefig(f'{output_dir}/shap_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Cumulative importance plot
    df_cumulative = df.copy()
    df_cumulative = df_cumulative.sort_values('SHAP_Importance', ascending=False)
    df_cumulative['Cumulative_Importance'] = df_cumulative['SHAP_Importance'].cumsum()
    
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(df_cumulative) + 1), df_cumulative['Cumulative_Importance'], 
             marker='o', linewidth=2, markersize=6)
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% threshold')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative SHAP Importance')
    plt.title('Cumulative SHAP Feature Importance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_cumulative_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Feature importance by category (if features can be categorized)
    # Group features by type
    temperature_features = [f for f in df['Feature'] if 'temperature' in f.lower()]
    precipitation_features = [f for f in df['Feature'] if 'precipitation' in f.lower()]
    soil_features = [f for f in df['Feature'] if any(x in f.lower() for x in ['soil', 'ph', 'texture'])]
    other_features = [f for f in df['Feature'] if f not in temperature_features + precipitation_features + soil_features]
    
    categories = {
        'Temperature': df[df['Feature'].isin(temperature_features)]['SHAP_Importance'].sum(),
        'Precipitation': df[df['Feature'].isin(precipitation_features)]['SHAP_Importance'].sum(),
        'Soil': df[df['Feature'].isin(soil_features)]['SHAP_Importance'].sum(),
        'Other': df[df['Feature'].isin(other_features)]['SHAP_Importance'].sum()
    }
    
    plt.figure(figsize=(10, 8))
    plt.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('SHAP Importance by Feature Category')
    plt.axis('equal')
    plt.savefig(f'{output_dir}/shap_category_pie.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"All SHAP plots saved to {output_dir}/")
    print(f"Top 5 most important features:")
    for i, (_, row) in enumerate(df.tail(5).iterrows(), 1):
        print(f"{i}. {row['Feature']}: {row['SHAP_Importance']:.4f}")

if __name__ == "__main__":
    # Example usage
    csv_path = "outputs/testing_SDM_out/artemisia_india_level/shap_importance_scores.csv"
    create_shap_plots(csv_path)




