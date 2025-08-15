# Import necessary libraries for geospatial analysis, machine learning, and data processing
from . import features_extractor
# import features_extractor

import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from shapely.wkt import loads
import random
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
import numpy as np
import Levenshtein
import os 
import math
import glob
import folium

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def divide_polygon_to_grids(polygon, grid_size=10, points_per_cell=5):
    """
    Divides a polygon into a grid and samples random points from each grid cell that intersects with the polygon.
    This creates a spatially distributed sampling strategy to represent the entire polygon area.
    
    Args:
        polygon: WKT string representation of the polygon to be sampled
        grid_size: Number of grid divisions along each axis (creates grid_size x grid_size grid)
        points_per_cell: Number of random points to sample from each non-empty grid cell
    
    Returns:
        pandas.DataFrame: DataFrame with columns ['longitude', 'latitude'] containing sampled points
    """
    
    # Convert WKT string to Shapely Polygon object for geometric operations
    polygon = loads(polygon)
    
    # Get the bounding box coordinates of the polygon to define grid boundaries
    min_x, min_y, max_x, max_y = polygon.bounds
    
    # Calculate the step size for grid divisions in both x and y directions
    step_x = (max_x - min_x) / grid_size
    step_y = (max_y - min_y) / grid_size
    
    # Initialize list to store all sampled points and counter for total points
    sampled_points = []
    total= 0
    
    # Iterate through each grid cell in the grid_size x grid_size grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the boundaries of the current grid cell
            cell_min_x = min_x + i * step_x
            cell_max_x = min_x + (i + 1) * step_x
            cell_min_y = min_y + j * step_y
            cell_max_y = min_y + (j + 1) * step_y
            
            # Create a rectangular polygon representing the current grid cell
            grid_cell = box(cell_min_x, cell_min_y, cell_max_x, cell_max_y)

            # Find the intersection between the grid cell and the original polygon
            # This handles cases where the polygon doesn't fill the entire grid cell
            intersection = polygon.intersection(grid_cell)

            # Only proceed if there's an actual intersection (non-empty area)
            if not intersection.is_empty:
                # Initialize list to store points sampled from this specific cell
                points_in_cell = []
                
                # Continue sampling until we have the desired number of points per cell
                while len(points_in_cell) < points_per_cell:
                    # Generate random coordinates within the grid cell boundaries
                    random_x = random.uniform(cell_min_x, cell_max_x)
                    random_y = random.uniform(cell_min_y, cell_max_y)
                    point = Point(random_x, random_y)
                    
                    # Only keep the point if it actually lies within the polygon intersection
                    # This ensures we don't sample points outside the original polygon
                    if intersection.contains(point):
                        points_in_cell.append(point)
                
                # Update total counter and add points from this cell to the overall collection
                total+=len(points_in_cell)
                # print(total)  # Debug line to track sampling progress
                sampled_points.extend(points_in_cell)

    # Convert Shapely Point objects to coordinate pairs [longitude, latitude]
    # print('points sampled',total)  # Debug line to show total points sampled
    sampled_points = [[point.x, point.y] for point in sampled_points]

    # Create a pandas DataFrame with proper column names for the coordinate data
    sampled_points = pd.DataFrame(sampled_points, columns=["longitude", "latitude"])

    return sampled_points


def representative_feature_vector_for_polygon(sampled_points, ee):
    """
    Generates a representative feature vector for a polygon by extracting features from sampled points
    and computing their mean values. This creates a single vector that characterizes the entire polygon.
    
    Args:
        sampled_points: DataFrame containing longitude/latitude coordinates of sampled points
        ee: Earth Engine object or similar service for feature extraction
    
    Returns:
        list: Mean feature vector representing the polygon's characteristics
    """

    # Initialize the feature extractor with the Earth Engine service
    feature_Extractor = features_extractor.Feature_Extractor(ee)
    
    # Extract features for all sampled points (likely environmental/remote sensing data)
    # This could include things like elevation, temperature, precipitation, vegetation indices, etc.
    features_df = feature_Extractor.add_features(sampled_points)
    
    # Calculate the mean of each feature across all sampled points
    # This creates a representative "average" feature vector for the entire polygon
    # skipna=True handles any missing values in the feature data
    feature_vector = features_df.mean(axis=0, skipna=True).tolist()
    
    # Remove the first two columns (likely longitude/latitude) to keep only the extracted features
    feature_vector=feature_vector[2:]
    
    return feature_vector

def find_representive_vectors_from_files(input_folder, ee):
    """
    Processes multiple WKT polygon files to generate representative feature vectors for each.
    This function handles batch processing of multiple ecoregions or geographic areas.
    
    Args:
        input_folder: Path to folder containing .wkt files with polygon definitions
        ee: Earth Engine object for feature extraction
    
    Returns:
        pandas.DataFrame: DataFrame where each row is a feature vector for one polygon file
    """
    # Initialize lists to store results
    feature_vectors = []  # Will store the computed feature vectors
    file_names = []       # Will store corresponding file names for identification
    
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Process only files with .wkt extension (Well-Known Text format for geometries)
        if filename.endswith('.wkt'):
            print('processing',filename)  # Progress indicator
            
            # Read the WKT polygon definition from the file
            with open(os.path.join(input_folder, filename), 'r') as file:
                polygon_wkt = file.read().strip()
                
            # Verify that we have a valid WKT string (not a Polygon object)
            if isinstance(polygon_wkt, str):
                # Use the WKT string directly (it's already in the correct format)
                polygon = polygon_wkt
                
                # Generate spatially distributed sample points within the polygon
                # Using larger grid and more points per cell for better representation
                sampled_points = divide_polygon_to_grids(polygon, grid_size=10, points_per_cell=10)
                
                # Extract and compute the representative feature vector for this polygon
                feature_vector = representative_feature_vector_for_polygon(sampled_points, ee)
                
                # Store the results with corresponding file name for later identification
                feature_vectors.append(feature_vector)
                file_names.append(filename)
            else:
                print(f"Skipping {filename} because it is not a valid WKT string.")
    
    # Convert the list of feature vectors into a structured DataFrame
    feature_vectors_df = pd.DataFrame(feature_vectors)
    # Use file names as row indices for easy identification of each ecoregion
    feature_vectors_df.index = file_names
    
    # Save the results to CSV for future use without reprocessing
    feature_vectors_df.to_csv('data/representative_vectors_eco_region_wise.csv')
    
    return feature_vectors_df


def calculate_cosine_similarity_matrix(feature_vectors_df):
    """
    Calculates cosine similarity between all pairs of feature vectors.
    Cosine similarity measures the cosine of the angle between vectors,
    focusing on orientation rather than magnitude. Values range from -1 to 1,
    where 1 indicates identical orientation.
    
    Args:
        feature_vectors_df: DataFrame where each row is a feature vector
    
    Returns:
        numpy.ndarray: Square matrix of cosine similarities between all vector pairs
    """
    similarity_matrix = cosine_similarity(feature_vectors_df)
    return similarity_matrix

def calculate_euclidean_similarity_matrix(feature_vectors_df):
    """
    Calculates similarity based on Euclidean distance between feature vectors.
    Euclidean distance measures straight-line distance in multi-dimensional space.
    The function converts distances to similarities using inverse transformation.
    
    Args:
        feature_vectors_df: DataFrame where each row is a feature vector
    
    Returns:
        numpy.ndarray: Square matrix of similarity values based on Euclidean distances
    """
    # Calculate pairwise Euclidean distances between all feature vectors
    distance_matrix = euclidean_distances(feature_vectors_df)
    
    # Convert distances to similarities: closer points (smaller distances) get higher similarity
    # Adding 1 prevents division by zero and ensures all values are positive
    similarity_matrix = 1 / (1 + distance_matrix)
    
    return similarity_matrix



def jaccard_similarity(input_file, similarity_threshold=0.8):
    """
    Calculates Jaccard similarity between ecoregions based on their genus composition.
    Jaccard similarity measures overlap between sets: |A ∩ B| / |A ∪ B|
    Also incorporates fuzzy string matching to handle similar genus names.
    
    Args:
        input_file: CSV file containing ecoregions and their associated genus lists
        similarity_threshold: Threshold for considering genus names as similar (0-1)
    """
    # Load the CSV file containing ecoregion data and genus information
    df = pd.read_csv(input_file)
    print(df.columns)  # Display available columns for debugging
    
    def are_genus_similar(genus1, genus2, threshold):
        """
        Determines if two genus names are similar using Levenshtein distance.
        This handles cases where genus names might have slight spelling variations.
        
        Args:
            genus1, genus2: Genus names to compare
            threshold: Similarity threshold (0=completely different, 1=identical)
        
        Returns:
            bool: True if genera are considered similar enough
        """
        # Convert to lowercase for case-insensitive comparison, handle missing values
        genus1 = genus1.lower() if isinstance(genus1, str) else ""
        genus2 = genus2.lower() if isinstance(genus2, str) else ""
        
        # Skip comparison if either genus name is missing
        if pd.isna(genus1) or pd.isna(genus2):
            return False
        
        # Calculate edit distance (number of character changes needed)
        lev_distance = Levenshtein.distance(genus1, genus2)
        # Normalize by the length of the longer string to get similarity ratio
        max_len = max(len(genus1), len(genus2))
        similarity = 1 - lev_distance / max_len
        
        # Return True if similarity exceeds the threshold
        return similarity >= threshold
    
    # Dictionary to store the set of genera for each ecoregion
    eco_region_genus = {}

    # Process each row in the dataset to build genus sets for each ecoregion
    for _, row in df.iterrows():
        eco_region = row["Eco-region"]
        # Split the comma-separated genus list into individual genus names
        genus_list = row["Genus List"].split(", ")
        
        # Merge similar genus names to avoid double-counting near-duplicates
        merged_genus_list = []
        for genus in genus_list:
            # Check if this genus is similar to any already processed genus
            to_add = True
            for existing_genus in merged_genus_list:
                if are_genus_similar(genus, existing_genus, similarity_threshold):
                    # If similar genus already exists, don't add this one
                    to_add = False
                    break
            # Add genus only if it's not similar to existing ones
            if to_add:
                merged_genus_list.append(genus)
        
        # Store the merged genus set for this ecoregion
        eco_region_genus[eco_region] = set(merged_genus_list)

    # Extract list of all ecoregions and initialize similarity matrix
    eco_regions = list(eco_region_genus.keys())
    similarity_matrix = np.zeros((len(eco_regions), len(eco_regions)))

    # Create mapping from matrix indices to ecoregion names (for reference)
    eco_region_index_map = {i: eco_region for i, eco_region in enumerate(eco_regions)}

    # Calculate Jaccard similarity for each pair of ecoregions
    for i in range(len(eco_regions)):
        for j in range(i, len(eco_regions)):  # Only calculate upper triangle (symmetric matrix)
            eco_region_i = eco_regions[i]
            eco_region_j = eco_regions[j]
            
            # Calculate set intersection (genera present in both ecoregions)
            intersection = len(eco_region_genus[eco_region_i].intersection(eco_region_genus[eco_region_j]))
            # Calculate set union (all unique genera across both ecoregions)
            union = len(eco_region_genus[eco_region_i].union(eco_region_genus[eco_region_j]))
            
            # Jaccard similarity = intersection size / union size
            # Handles edge case where union is empty (though unlikely with real data)
            similarity = intersection / union if union != 0 else 0
            
            # Fill both symmetric positions in the matrix
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # Save the resulting similarity matrix in human-readable format
    save_matrix_to_text(similarity_matrix, "outputs/jaccard_similarity_matrix.txt", eco_regions)

    print("Jaccard Similarity Matrix has been calculated and saved to 'jaccard_similarity_matrix.csv'.")


def save_matrix_to_text(matrix, filename, labels):
    """
    Saves a similarity matrix to a human-readable text file with proper formatting.
    The output includes row and column labels for easy interpretation.
    
    Parameters:
    matrix (np.ndarray): Square similarity matrix to save
    filename (str): Path and filename for the output text file
    labels (list): Labels for rows and columns (typically ecoregion names)
    """
    with open(filename, 'w') as f:
        # Write column headers with proper spacing
        # First column is left empty to align with row labels
        f.write(' ' * 50)  # Indent to align with row label width
        f.write('\t'.join(labels) + '\n')
        
        # Write each row of the matrix with its corresponding label
        for i, row_label in enumerate(labels):
            # Format each similarity value to 4 decimal places for readability
            row_values = [f"{val:.4f}" for val in matrix[i]]
            # Create formatted row: label (left-aligned, 50 chars) + tab + values
            formatted_row = f"{row_label:<50}\t" + '\t'.join(row_values) + '\n'
            f.write(formatted_row)

def calculate_concentration_index(presence_points_csv, ecoregion_wkt_directory):
    """
    Calculates the concentration index for a species based on the number of ecoregions
    where the species is present divided by the total number of ecoregions (48).
    
    The concentration index measures how widely distributed a species is across ecoregions.
    A higher value indicates the species is present in more ecoregions (wider distribution).
    A lower value indicates the species is concentrated in fewer ecoregions.
    
    Args:
        presence_points_csv (str): Path to CSV file containing presence points with columns:
                                 longitude, latitude, and environmental features
        ecoregion_wkt_directory (str): Path to directory containing .wkt files for each ecoregion
    
    Returns:
        float: Concentration index (number of ecoregions with presence / 48)
        int: Number of ecoregions where species is present
        list: List of ecoregion names where species is present
    """
    # Read presence points CSV
    presence_df = pd.read_csv(presence_points_csv)
    
    # Extract longitude and latitude columns
    presence_points = []
    for _, row in presence_df.iterrows():
        point = Point(row['longitude'], row['latitude'])
        presence_points.append(point)
    
    # Initialize set to store unique ecoregions where species is present
    ecoregions_with_presence = set()
    
    # Process each WKT file in the ecoregion directory
    for filename in os.listdir(ecoregion_wkt_directory):
        if filename.endswith('.wkt'):
            ecoregion_name = filename.replace('.wkt', '')
            
            # Read WKT polygon from file
            with open(os.path.join(ecoregion_wkt_directory, filename), 'r') as file:
                polygon_wkt = file.read().strip()
            
            # Convert WKT to Shapely polygon
            try:
                ecoregion_polygon = loads(polygon_wkt)
                
                # Check if any presence points fall within this ecoregion
                for point in presence_points:
                    if ecoregion_polygon.contains(point):
                        ecoregions_with_presence.add(ecoregion_name)
                        break  # Found at least one point in this ecoregion, move to next
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    # Calculate concentration index
    num_ecoregions_with_presence = len(ecoregions_with_presence)
    concentration_index = num_ecoregions_with_presence / 48.0
    
    return concentration_index, num_ecoregions_with_presence, list(ecoregions_with_presence)

def calculate_endemicity_index(presence_points_csv, ecoregion_wkt_directory):
    """
    Calculates the endemicity index for a species based on the entropy of presence point
    distribution across ecoregions. The endemicity value is the sum of entropy values
    for each ecoregion, where entropy = -p*log(p) and p is the proportion of presence
    points in that ecoregion relative to total presence points.
    
    A higher endemicity value indicates the species is more concentrated in specific
    ecoregions (more endemic), while a lower value indicates more uniform distribution.
    
    Args:
        presence_points_csv (str): Path to CSV file containing presence points with columns:
                                 longitude, latitude, and environmental features
        ecoregion_wkt_directory (str): Path to directory containing .wkt files for each ecoregion
    
    Returns:
        float: Endemicity index (sum of entropy values across all ecoregions with presence)
        dict: Dictionary mapping ecoregion names to their entropy values
        dict: Dictionary mapping ecoregion names to their presence point counts
    """
    import math
    
    # Read presence points CSV
    presence_df = pd.read_csv(presence_points_csv)
    total_presence_points = len(presence_df)
    
    # Extract longitude and latitude columns
    presence_points = []
    for _, row in presence_df.iterrows():
        point = Point(row['longitude'], row['latitude'])
        presence_points.append(point)
    
    # Dictionary to store presence point counts for each ecoregion
    ecoregion_presence_counts = {}
    
    # Process each WKT file in the ecoregion directory
    for filename in os.listdir(ecoregion_wkt_directory):
        if filename.endswith('.wkt'):
            ecoregion_name = filename.replace('.wkt', '')
            
            # Read WKT polygon from file
            with open(os.path.join(ecoregion_wkt_directory, filename), 'r') as file:
                polygon_wkt = file.read().strip()
            
            # Convert WKT to Shapely polygon
            try:
                ecoregion_polygon = loads(polygon_wkt)
                
                # Count presence points that fall within this ecoregion
                points_in_ecoregion = 0
                for point in presence_points:
                    if ecoregion_polygon.contains(point):
                        points_in_ecoregion += 1
                
                # Store count if there are presence points in this ecoregion
                if points_in_ecoregion > 0:
                    ecoregion_presence_counts[ecoregion_name] = points_in_ecoregion
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    # Calculate entropy for each ecoregion with presence points
    ecoregion_entropy = {}
    total_endemicity = 0.0
    
    for ecoregion_name, point_count in ecoregion_presence_counts.items():
        # Calculate proportion of presence points in this ecoregion
        p = point_count / total_presence_points
        
        # Calculate entropy: -p * log(p)
        # Use natural logarithm (math.log) for entropy calculation
        entropy = -p * math.log(p)
        
        ecoregion_entropy[ecoregion_name] = entropy
        total_endemicity += entropy
    
    return total_endemicity

def calculate_indices_for_all_species(all_species_presence_csv, ecoregion_wkt_directory, output_csv_path="outputs/species_indices.csv"):
    """
    Calculates concentration and endemicity indices for all species in a dataset.
    Processes presence points for multiple species and outputs results to a CSV file.
    
    Args:
        all_species_presence_csv (str): Path to CSV file containing presence points for all species
                                       with columns: kingdom,phylum,class,order,family,genus,species,
                                       decimalLatitude,decimalLongitude
        ecoregion_wkt_directory (str): Path to directory containing .wkt files for each ecoregion
        output_csv_path (str): Path where the output CSV file will be saved
    
    Returns:
        pandas.DataFrame: DataFrame containing species names and their calculated indices
    """
    import math
    
    # Read the all species presence data
    all_species_df = pd.read_csv(all_species_presence_csv)
    
    # Group by species to process each species separately
    species_groups = all_species_df.groupby('species')
    
    # Lists to store results
    species_names = []
    concentration_indices = []
    endemicity_indices = []
    
    print(f"Processing {len(species_groups)} species...")
    
    # Process each species
    for species_name, species_data in species_groups:
        print(f"Processing species: {species_name}")
        
        # Extract presence points for this species
        presence_points = []
        for _, row in species_data.iterrows():
            point = Point(row['decimalLongitude'], row['decimalLatitude'])
            presence_points.append(point)
        
        total_presence_points = len(presence_points)
        
        if total_presence_points == 0:
            print(f"Warning: No presence points found for species {species_name}")
            continue
        
        # Initialize variables for this species
        ecoregions_with_presence = set()
        ecoregion_presence_counts = {}
        
        # Process each WKT file in the ecoregion directory
        for filename in os.listdir(ecoregion_wkt_directory):
            if filename.endswith('.wkt'):
                ecoregion_name = filename.replace('.wkt', '')
                
                # Read WKT polygon from file
                with open(os.path.join(ecoregion_wkt_directory, filename), 'r') as file:
                    polygon_wkt = file.read().strip()
                
                # Convert WKT to Shapely polygon
                try:
                    ecoregion_polygon = loads(polygon_wkt)
                    
                    # Count presence points that fall within this ecoregion
                    points_in_ecoregion = 0
                    for point in presence_points:
                        if ecoregion_polygon.contains(point):
                            points_in_ecoregion += 1
                    
                    # Store count if there are presence points in this ecoregion
                    if points_in_ecoregion > 0:
                        ecoregions_with_presence.add(ecoregion_name)
                        ecoregion_presence_counts[ecoregion_name] = points_in_ecoregion
                            
                except Exception as e:
                    print(f"Error processing {filename} for species {species_name}: {e}")
                    continue
        
        # Calculate concentration index
        num_ecoregions_with_presence = len(ecoregions_with_presence)
        concentration_index = num_ecoregions_with_presence / 48.0
        
        # Calculate endemicity index (entropy-based)
        total_endemicity = 0.0
        for ecoregion_name, point_count in ecoregion_presence_counts.items():
            # Calculate proportion of presence points in this ecoregion
            p = point_count / total_presence_points
            
            # Calculate entropy: -p * log(p)
            entropy = -p * math.log(p)
            total_endemicity += entropy
        
        # Store results
        species_names.append(species_name)
        concentration_indices.append(concentration_index)
        endemicity_indices.append(total_endemicity)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'species': species_names,
        'concentration_index': concentration_indices,
        'endemicity_index': endemicity_indices
    })
    
    # Save to CSV
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")
    
    return results_df


def calculate_genus_concentration_index(genus_name, all_species_presence_csv, ecoregion_wkt_directory):
    """
    Calculates the concentration index for a specific genus based on the number of ecoregions
    where the genus is present divided by the total number of ecoregions (48).
    
    The concentration index measures how widely distributed a genus is across ecoregions.
    A higher value indicates the genus is present in more ecoregions (wider distribution).
    A lower value indicates the genus is concentrated in fewer ecoregions.
    
    Args:
        genus_name (str): Name of the genus to analyze
        all_species_presence_csv (str): Path to CSV file containing presence points for all species
                                       with columns: genus, decimalLatitude, decimalLongitude
        ecoregion_wkt_directory (str): Path to directory containing .wkt files for each ecoregion
    
    Returns:
        float: Concentration index (number of ecoregions with presence / 48)
        int: Number of ecoregions where genus is present
        list: List of ecoregion names where genus is present
    """
    # Read the CSV and filter only the specified genus
    df = pd.read_csv(all_species_presence_csv)
    genus_df = df[df['genus'] == genus_name]
    
    if genus_df.empty:
        print(f"No presence data found for genus {genus_name}.")
        return 0.0, 0, []
    
    # Extract longitude and latitude columns
    presence_points = []
    for _, row in genus_df.iterrows():
        point = Point(row['decimalLongitude'], row['decimalLatitude'])
        presence_points.append(point)
    
    # Initialize set to store unique ecoregions where genus is present
    ecoregions_with_presence = set()
    
    # Process each WKT file in the ecoregion directory
    for filename in os.listdir(ecoregion_wkt_directory):
        if filename.endswith('.wkt'):
            ecoregion_name = filename.replace('.wkt', '')
            
            # Read WKT polygon from file
            with open(os.path.join(ecoregion_wkt_directory, filename), 'r') as file:
                polygon_wkt = file.read().strip()
            
            # Convert WKT to Shapely polygon
            try:
                ecoregion_polygon = loads(polygon_wkt)
                
                # Check if any presence points fall within this ecoregion
                for point in presence_points:
                    if ecoregion_polygon.contains(point):
                        ecoregions_with_presence.add(ecoregion_name)
                        break  # Found at least one point in this ecoregion, move to next
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    # Calculate concentration index
    num_ecoregions_with_presence = len(ecoregions_with_presence)
    concentration_index = num_ecoregions_with_presence / 48.0
    
    return concentration_index, num_ecoregions_with_presence, list(ecoregions_with_presence)

def calculate_genus_endemicity_index(genus_name, all_species_presence_csv, ecoregion_wkt_directory):
    """
    Calculates the endemicity index for a specific genus based on the entropy of presence point
    distribution across ecoregions. The endemicity value is the sum of entropy values
    for each ecoregion, where entropy = -p*log(p) and p is the proportion of presence
    points in that ecoregion relative to total presence points.
    
    A higher endemicity value indicates the genus is more concentrated in specific
    ecoregions (more endemic), while a lower value indicates more uniform distribution.
    
    Args:
        genus_name (str): Name of the genus to analyze
        all_species_presence_csv (str): Path to CSV file containing presence points for all species
                                       with columns: genus, decimalLatitude, decimalLongitude
        ecoregion_wkt_directory (str): Path to directory containing .wkt files for each ecoregion
    
    Returns:
        float: Endemicity index (sum of entropy values across all ecoregions with presence)
        dict: Dictionary mapping ecoregion names to their entropy values
        dict: Dictionary mapping ecoregion names to their presence point counts
    """
    import math
    
    # Read the CSV and filter only the specified genus
    df = pd.read_csv(all_species_presence_csv)
    genus_df = df[df['genus'] == genus_name]
    
    if genus_df.empty:
        print(f"No presence data found for genus {genus_name}.")
        return 0.0, {}, {}
    
    total_presence_points = len(genus_df)
    
    # Extract longitude and latitude columns
    presence_points = []
    for _, row in genus_df.iterrows():
        point = Point(row['decimalLongitude'], row['decimalLatitude'])
        presence_points.append(point)
    
    # Dictionary to store presence point counts for each ecoregion
    ecoregion_presence_counts = {}
    
    # Process each WKT file in the ecoregion directory
    for filename in os.listdir(ecoregion_wkt_directory):
        if filename.endswith('.wkt'):
            ecoregion_name = filename.replace('.wkt', '')
            
            # Read WKT polygon from file
            with open(os.path.join(ecoregion_wkt_directory, filename), 'r') as file:
                polygon_wkt = file.read().strip()
            
            # Convert WKT to Shapely polygon
            try:
                ecoregion_polygon = loads(polygon_wkt)
                
                # Count presence points that fall within this ecoregion
                points_in_ecoregion = 0
                for point in presence_points:
                    if ecoregion_polygon.contains(point):
                        points_in_ecoregion += 1
                
                # Store count if there are presence points in this ecoregion
                if points_in_ecoregion > 0:
                    ecoregion_presence_counts[ecoregion_name] = points_in_ecoregion
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    # Calculate entropy for each ecoregion with presence points
    ecoregion_entropy = {}
    total_endemicity = 0.0
    
    for ecoregion_name, point_count in ecoregion_presence_counts.items():
        # Calculate proportion of presence points in this ecoregion
        p = point_count / total_presence_points
        
        # Calculate entropy: -p * log(p)
        # Use natural logarithm (math.log) for entropy calculation
        entropy = -p * math.log(p)
        
        ecoregion_entropy[ecoregion_name] = entropy
        total_endemicity += entropy
    
    return total_endemicity, ecoregion_entropy, ecoregion_presence_counts

def calculate_indices_for_all_genera(all_species_presence_csv, ecoregion_wkt_directory, output_csv_path="outputs/genus_indices.csv"):
    """
    Calculates concentration and endemicity indices for all genera in a dataset.
    Processes presence points for multiple genera and outputs results to a CSV file.
    
    Args:
        all_species_presence_csv (str): Path to CSV file containing presence points for all species
                                       with columns: kingdom,phylum,class,order,family,genus,species,
                                       decimalLatitude,decimalLongitude
        ecoregion_wkt_directory (str): Path to directory containing .wkt files for each ecoregion
        output_csv_path (str): Path where the output CSV file will be saved
    
    Returns:
        pandas.DataFrame: DataFrame containing genus names and their calculated indices
    """
    import math
    
    # Read the all species presence data
    all_species_df = pd.read_csv(all_species_presence_csv)
    
    # Group by genus to process each genus separately
    genus_groups = all_species_df.groupby('genus')
    
    # Lists to store results
    genus_names = []
    concentration_indices = []
    endemicity_indices = []
    total_presence_points = []
    num_ecoregions_with_presence = []
    
    print(f"Processing {len(genus_groups)} genera...")
    
    # Process each genus
    for genus_name, genus_data in genus_groups:
        print(f"Processing genus: {genus_name}")
        
        # Extract presence points for this genus
        presence_points = []
        for _, row in genus_data.iterrows():
            point = Point(row['decimalLongitude'], row['decimalLatitude'])
            presence_points.append(point)
        
        total_points = len(presence_points)
        
        if total_points == 0:
            print(f"Warning: No presence points found for genus {genus_name}")
            continue
        
        # Initialize variables for this genus
        ecoregions_with_presence = set()
        ecoregion_presence_counts = {}
        
        # Process each WKT file in the ecoregion directory
        for filename in os.listdir(ecoregion_wkt_directory):
            if filename.endswith('.wkt'):
                ecoregion_name = filename.replace('.wkt', '')
                
                # Read WKT polygon from file
                with open(os.path.join(ecoregion_wkt_directory, filename), 'r') as file:
                    polygon_wkt = file.read().strip()
                
                # Convert WKT to Shapely polygon
                try:
                    ecoregion_polygon = loads(polygon_wkt)
                    
                    # Count presence points that fall within this ecoregion
                    points_in_ecoregion = 0
                    for point in presence_points:
                        if ecoregion_polygon.contains(point):
                            points_in_ecoregion += 1
                    
                    # Store count if there are presence points in this ecoregion
                    if points_in_ecoregion > 0:
                        ecoregions_with_presence.add(ecoregion_name)
                        ecoregion_presence_counts[ecoregion_name] = points_in_ecoregion
                            
                except Exception as e:
                    print(f"Error processing {filename} for genus {genus_name}: {e}")
                    continue
        
        # Calculate concentration index
        num_ecoregions = len(ecoregions_with_presence)
        concentration_index = num_ecoregions / 48.0
        
        # Calculate endemicity index (entropy-based)
        total_endemicity = 0.0
        for ecoregion_name, point_count in ecoregion_presence_counts.items():
            # Calculate proportion of presence points in this ecoregion
            p = point_count / total_points
            
            # Calculate entropy: -p * log(p)
            entropy = -p * math.log(p)
            total_endemicity += entropy
        
        # Store results
        genus_names.append(genus_name)
        concentration_indices.append(concentration_index)
        endemicity_indices.append(total_endemicity)
        total_presence_points.append(total_points)
        num_ecoregions_with_presence.append(num_ecoregions)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'genus': genus_names,
        'concentration_index': concentration_indices,
        'endemicity_index': endemicity_indices,
        'total_presence_points': total_presence_points,
        'num_ecoregions_with_presence': num_ecoregions_with_presence
    })
    
    # Sort by concentration index (descending) for better readability
    results_df = results_df.sort_values('concentration_index', ascending=False)
    
    # Save to CSV
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")
    
    return results_df

def calculate_genus_indices(genus_name, all_species_presence_csv, ecoregion_wkt_directory):
    """
    Calculates both concentration and endemicity indices for a specific genus.
    This is a convenience function that combines both calculations.
    
    Args:
        genus_name (str): Name of the genus to analyze
        all_species_presence_csv (str): Path to CSV file containing presence points for all species
        ecoregion_wkt_directory (str): Path to directory containing .wkt files for each ecoregion
    
    Returns:
        tuple: (concentration_index, endemicity_index, num_ecoregions_with_presence, total_presence_points)
    """
    # Calculate concentration index
    concentration_index, num_ecoregions, ecoregions_list = calculate_genus_concentration_index(
        genus_name, all_species_presence_csv, ecoregion_wkt_directory
    )
    
    # Calculate endemicity index
    endemicity_index, ecoregion_entropy, ecoregion_counts = calculate_genus_endemicity_index(
        genus_name, all_species_presence_csv, ecoregion_wkt_directory
    )
    
    # Get total presence points
    df = pd.read_csv(all_species_presence_csv)
    genus_df = df[df['genus'] == genus_name]
    total_points = len(genus_df)
    
    print(f"{genus_name} → concentration_index = {concentration_index:.4f}, endemicity_index = {endemicity_index:.4f}")
    print(f"  Total presence points: {total_points}, Ecoregions with presence: {num_ecoregions}")
    
    return concentration_index, endemicity_index, num_ecoregions, total_points

def print_top_10_presence_features():
    """
    Takes the top 10 presence points from all_presence_point.csv and prints all their feature values.
    This function is useful for debugging and understanding the data structure.
    """
    import ee
    from .features_extractor import Feature_Extractor
    
    print("Loading top 10 presence points from all_presence_point.csv...")
    
    # Load the presence points data
    df = pd.read_csv('data/testing_SDM/all_presence_point.csv')
    
    # Take the first 10 presence points
    top_10_presence = df.head(10)[['decimalLongitude', 'decimalLatitude']]
    top_10_presence.columns = ['longitude', 'latitude']
    
    print(f"Top 10 presence points coordinates:")
    print(top_10_presence)
    print("\n" + "="*80 + "\n")
    
    # Initialize Earth Engine and feature extractor
    ee.Initialize()
    fe = Feature_Extractor(ee)
    
    print("Extracting features for top 10 presence points...")
    
    # Extract features for the top 10 presence points
    features_df = fe.add_features(top_10_presence)
    
    print(f"Feature extraction completed. Shape: {features_df.shape}")
    print(f"Feature columns: {list(features_df.columns)}")
    print("\n" + "="*80 + "\n")
    
    # Print detailed feature values for each point
    for i, (idx, row) in enumerate(features_df.iterrows(), 1):
        print(f"POINT {i}:")
        print(f"  Coordinates: ({row['longitude']:.6f}, {row['latitude']:.6f})")
        print("  Feature Values:")
        
        # Print each feature value
        for col in features_df.columns:
            if col not in ['longitude', 'latitude']:
                value = row[col]
                if pd.isna(value):
                    print(f"    {col}: NaN")
                else:
                    print(f"    {col}: {value}")
        
        print("-" * 60)
    
    print(f"\nSummary:")
    print(f"Total points processed: {len(features_df)}")
    print(f"Total features extracted: {len(features_df.columns) - 2}")  # Exclude lat/lon
    print(f"Features with NaN values: {features_df.isna().sum().sum()}")
    
    return features_df


def print_top_n_presence_features(start_row, end_row, append_to_existing=False):
    """
    Takes presence points from all_presence_point.csv within a specific range and saves all their feature values to a CSV file.
    
    Args:
        start_row (int): Starting row index (0-based)
        end_row (int): Ending row index (exclusive)
        append_to_existing (bool): If True, append to existing CSV file. If False, create new file.
    """
    import ee
    from .features_extractor import Feature_Extractor
    
    print(f"Loading presence points from row {start_row} to {end_row-1} from all_presence_point.csv...")
    
    # Load the presence points data
    df = pd.read_csv('data/testing_SDM/all_presence_point.csv')
    
    if len(df) == 0:
        print("No presence points found in the file")
        return None
    
    # Check if the requested range is valid
    if start_row >= len(df):
        print(f"Error: start_row ({start_row}) is beyond the available data ({len(df)} rows)")
        return None
    
    if end_row > len(df):
        print(f"Warning: end_row ({end_row}) is beyond available data. Using {len(df)} instead.")
        end_row = len(df)
    
    # Take the specified range of presence points with all original columns
    range_presence = df.iloc[start_row:end_row].copy()
    
    # Prepare coordinates for feature extraction
    coords_df = range_presence[['decimalLongitude', 'decimalLatitude']].copy()
    coords_df.columns = ['longitude', 'latitude']
    
    print(f"Found {len(df)} total presence points in file")
    print(f"Processing rows {start_row} to {end_row-1} ({len(range_presence)} points)...")
    print("\n" + "="*80 + "\n")
    
    # Initialize Earth Engine and feature extractor
    ee.Initialize()
    fe = Feature_Extractor(ee)
    
    print(f"Extracting features for {len(range_presence)} presence points...")
    
    # Extract features for the specified range of presence points
    features_df = fe.add_features(coords_df)
    
    print(f"Feature extraction completed. Shape: {features_df.shape}")
    print(f"Feature columns: {list(features_df.columns)}")
    print("\n" + "="*80 + "\n")
    
    # Combine original data with extracted features
    # Remove the longitude/latitude columns from features_df since they're already in range_presence
    feature_cols = [col for col in features_df.columns if col not in ['longitude', 'latitude']]
    features_only = features_df[feature_cols]
    
    # Combine the datasets
    combined_df = pd.concat([range_presence.reset_index(drop=True), features_only.reset_index(drop=True)], axis=1)
    
    # Determine output filename
    if append_to_existing:
        output_filename = "data/presence_points_with_features.csv"
        mode = 'a'  # append mode
        header = not os.path.exists(output_filename)  # write header only if file doesn't exist
    else:
        output_filename = f"data/presence_points_{start_row}_to_{end_row-1}_with_features.csv"
        mode = 'w'  # write mode
        header = True
    
    # Save to CSV file
    combined_df.to_csv(output_filename, index=False, mode=mode, header=header)
    
    print(f"Data saved to: {output_filename}")
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Original columns: {len(range_presence.columns)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Total columns: {len(combined_df.columns)}")
    
    # Print column names for reference
    print(f"\nAll columns in the saved file:")
    for i, col in enumerate(combined_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nSummary:")
    print(f"Total points processed: {len(combined_df)}")
    print(f"Total features extracted: {len(feature_cols)}")
    print(f"Features with NaN values: {combined_df[feature_cols].isna().sum().sum()}")
    print(f"File saved successfully to: {output_filename}")
    
    return combined_df

def list_presence_feature_files():
    """
    List all saved presence feature files in the data directory.
    """
    pattern = "data/presence_features_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print("No presence feature files found.")
        return []
    
    print(f"Found {len(files)} presence feature files:")
    for file in sorted(files):
        file_size = os.path.getsize(file)
        species_name = file.replace("data/presence_features_", "").replace(".csv", "").replace("_", " ")
        print(f"  {species_name}: {file_size:,} bytes")
    
    return files

def cleanup_presence_feature_files(species_list=None):
    """
    Clean up presence feature files. If species_list is provided, only remove those.
    If None, remove all presence feature files.
    
    Args:
        species_list (list): List of species names to remove. If None, remove all.
    """
    import glob
    import os
    
    pattern = "data/presence_features_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print("No presence feature files found to clean up.")
        return
    
    if species_list:
        # Remove only specified species
        files_to_remove = []
        for species in species_list:
            species_file = f"data/presence_features_{species.replace(' ', '_')}.csv"
            if species_file in files:
                files_to_remove.append(species_file)
        
        if not files_to_remove:
            print("No matching presence feature files found for the specified species.")
            return
    else:
        # Remove all presence feature files
        files_to_remove = files
    
    print(f"Removing {len(files_to_remove)} presence feature files:")
    for file in files_to_remove:
        try:
            os.remove(file)
            species_name = file.replace("data/presence_features_", "").replace(".csv", "").replace("_", " ")
            print(f"  ✓ Removed: {species_name}")
        except Exception as e:
            print(f"  ✗ Error removing {file}: {e}")
    
    print("Cleanup completed!")

if __name__ == "__main__":
    # print(calculate_concentration_index("data/testing_SDM/presence_points_Dalbergia_all_india.csv", "data/eco_regions_polygon"))
    # print(calculate_endemicity_index("data/testing_SDM/presence_points_Dalbergia_all_india.csv", "data/eco_regions_polygon"))
    # calculate_indices_for_all_species("data/testing_SDM/all_presence_point.csv", "data/eco_regions_polygon")
    # calculate_indices_for_all_genera("data/testing_SDM/all_presence_point.csv", "data/eco_regions_polygon")
    
    # Process first 20,000 points (rows 0 to 19999)
    # print_top_n_presence_features(0, 20000, append_to_existing=False)
    
    # To append more points, you can call:
    # print_top_n_presence_features(1, 20001, append_to_existing=False)
    print_top_n_presence_features(80000, 100000, append_to_existing=False)