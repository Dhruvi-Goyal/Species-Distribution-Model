import pandas as pd
import folium
from folium import plugins
import numpy as np

def plot_syzygium_presence_absence():
    """
    Create a Folium map showing syzygium presence and absence points
    """
    # Read the data
    presence_df = pd.read_csv('outputs/extracted_features/syzygium_ecoregion_presence_features.csv')
    absence_df = pd.read_csv('outputs/extracted_features/syzygium_ecoregion_absence_features.csv')
    
    # Clean the data - remove rows with missing coordinates
    presence_df = presence_df.dropna(subset=['longitude', 'latitude'])
    absence_df = absence_df.dropna(subset=['longitude', 'latitude'])
    
    print(f"Number of presence points: {len(presence_df)}")
    print(f"Number of absence points: {len(absence_df)}")
    
    # Calculate center of the map (average of all points)
    all_lats = np.concatenate([presence_df['latitude'].values, absence_df['latitude'].values])
    all_lons = np.concatenate([presence_df['longitude'].values, absence_df['longitude'].values])
    
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    # Create the map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add different tile layers
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
    
    # Create feature groups for different point types
    presence_group = folium.FeatureGroup(name='Presence Points', overlay=True)
    absence_group = folium.FeatureGroup(name='Absence Points', overlay=True)
    
    # Add presence points (green markers)
    for idx, row in presence_df.iterrows():
        if not pd.isna(row['longitude']) and not pd.isna(row['latitude']):
            # Create popup with information
            popup_text = f"""
            <b>syzygium Presence Point</b><br>
            <b>Coordinates:</b> {row['latitude']:.4f}, {row['longitude']:.4f}<br>
            <b>Elevation:</b> {row.get('elevation', 'N/A'):.2f}<br>
            <b>Annual Mean Temperature:</b> {row.get('annual_mean_temperature', 'N/A'):.3f}<br>
            <b>Annual Precipitation:</b> {row.get('annual_precipitation', 'N/A'):.3f}<br>
            <b>Topsoil pH:</b> {row.get('topsoil_ph', 'N/A'):.3f}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                popup=folium.Popup(popup_text, max_width=300),
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.7,
                weight=1
            ).add_to(presence_group)
    
    # Add absence points (red markers)
    for idx, row in absence_df.iterrows():
        if not pd.isna(row['longitude']) and not pd.isna(row['latitude']):
            # Create popup with information
            popup_text = f"""
            <b>syzygium Absence Point</b><br>
            <b>Coordinates:</b> {row['latitude']:.4f}, {row['longitude']:.4f}<br>
            <b>Elevation:</b> {row.get('elevation', 'N/A'):.2f}<br>
            <b>Annual Mean Temperature:</b> {row.get('annual_mean_temperature', 'N/A'):.3f}<br>
            <b>Annual Precipitation:</b> {row.get('annual_precipitation', 'N/A'):.3f}<br>
            <b>Topsoil pH:</b> {row.get('topsoil_ph', 'N/A'):.3f}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                popup=folium.Popup(popup_text, max_width=300),
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.5,
                weight=1
            ).add_to(absence_group)
    
    # Add feature groups to map
    presence_group.add_to(m)
    absence_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>syzygium Distribution</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> Presence Points</p>
    <p><i class="fa fa-circle" style="color:red"></i> Absence Points</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add some statistics as a text overlay
    stats_html = f'''
    <div style="position: fixed; 
                top: 50px; right: 50px; width: 250px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <p><b>syzygium Dataset Statistics</b></p>
    <p>Total Presence Points: {len(presence_df)}</p>
    <p>Total Absence Points: {len(absence_df)}</p>
    <p>Total Points: {len(presence_df) + len(absence_df)}</p>
    <p>Latitude Range: {all_lats.min():.2f}° to {all_lats.max():.2f}°</p>
    <p>Longitude Range: {all_lons.min():.2f}° to {all_lons.max():.2f}°</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Add heatmap option
    # Create heatmap data
    presence_heatmap_data = [[row['latitude'], row['longitude']] for idx, row in presence_df.iterrows() 
                            if not pd.isna(row['longitude']) and not pd.isna(row['latitude'])]
    absence_heatmap_data = [[row['latitude'], row['longitude']] for idx, row in absence_df.iterrows() 
                           if not pd.isna(row['longitude']) and not pd.isna(row['latitude'])]
    
    # Add heatmap layers
    if presence_heatmap_data:
        plugins.HeatMap(
            presence_heatmap_data,
            name='Presence Heatmap',
            radius=15,
            blur=10,
            max_zoom=13,
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
        ).add_to(m)
    
    if absence_heatmap_data:
        plugins.HeatMap(
            absence_heatmap_data,
            name='Absence Heatmap',
            radius=15,
            blur=10,
            max_zoom=13,
            gradient={0.4: 'purple', 0.65: 'orange', 1: 'red'}
        ).add_to(m)
    
    # Add fullscreen option
    plugins.Fullscreen().add_to(m)
    
    # Add minimap
    minimap = plugins.MiniMap(toggle_display=True)
    m.add_child(minimap)
    
    # Save the map
    output_file = 'syzygium_presence_absence_map.html'
    m.save(output_file)
    print(f"Map saved as: {output_file}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Presence points: {len(presence_df)}")
    print(f"Absence points: {len(absence_df)}")
    print(f"Total points: {len(presence_df) + len(absence_df)}")
    print(f"Latitude range: {all_lats.min():.4f}° to {all_lats.max():.4f}°")
    print(f"Longitude range: {all_lons.min():.4f}° to {all_lons.max():.4f}°")
    
    # Check for environmental data availability
    env_cols = ['annual_mean_temperature', 'annual_precipitation', 'elevation', 'topsoil_ph']
    print("\nEnvironmental Data Availability:")
    for col in env_cols:
        if col in presence_df.columns:
            presence_available = presence_df[col].notna().sum()
            absence_available = absence_df[col].notna().sum()
            print(f"{col}: {presence_available} presence, {absence_available} absence points")
    
    return m

if __name__ == "__main__":
    # Create the map
    map_obj = plot_syzygium_presence_absence()
    print("\nMap created successfully! Open 'syzygium_presence_absence_map.html' in your browser to view it.")




