import ee 
import os 
from shapely.wkt import loads
import pandas as pd
import numpy as np
import signal
from contextlib import contextmanager
from Modules import presence_dataloader, features_extractor, LULC_filter, pseudo_absence_generator, models, Generate_Prob, utility
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import spearmanr
import concurrent.futures
from functools import partial
import time
import traceback
import threading
import queue
from sklearn.model_selection import train_test_split
from Modules.feature_sensitivity_analysis import FeatureSensitivityAnalyzer
from Modules.models import perform_feature_importance_for_all_species
ee.Authenticate()
ee.Initialize(project='sigma-bay-425614-a6')





# The following code sets up the environment and imports necessary modules for the project.
# It authenticates and initializes the Google Earth Engine API.
# The context manager 'timeout' is defined below to handle timeouts for code blocks.
# There are also commented-out functions for testing models on ecoregions, which can be enabled as needed.

@contextmanager
def timeout(time):
    """Raise TimeoutError if the block takes longer than 'time' seconds."""
    def raise_timeout(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)
    try:
        yield
    finally:
        signal.alarm(0)

# def test_model_on_all_ecoregions(clf, Features_extractor, modelss, output_file='data/avg_prob.txt'):
#     polygon_dir = 'data/eco_regions_polygon'

#     # Write the header for the output file if the file doesn't exist
#     if not os.path.exists(output_file):
#         with open(output_file, 'w') as out_file:
#             out_file.write('Ecoregion,Average_Probability\n')

#     cnt = 0

#     for filename in os.listdir(polygon_dir):
#         print(f'Starting for ecoregion {cnt + 1}')
#         if filename.endswith('.wkt'):  # Process only .wkt files
#             ecoregion_name = os.path.splitext(filename)[0]  # Get ecoregion name without extension
#             polygon_path = os.path.join(polygon_dir, filename)

#             try:
#                 with timeout(15):  # Set a 2-minute timeout
#                     # Read the polygon WKT
#                     with open(polygon_path, 'r') as file:
#                         polygon_wkt = file.read().strip()

#                     # Generate test data for the current ecoregion
#                     X_dissimilar = Features_extractor.add_features(
#                         utility.divide_polygon_to_grids(polygon_wkt, grid_size=1, points_per_cell=20)
#                     )
#                     test_presence_path = 'data/test_presence.csv'
#                     pd.DataFrame(X_dissimilar).to_csv(test_presence_path, index=False)

#                     X_test, y_test, _, _, _ = modelss.load_data(
#                         presence_path=test_presence_path,
#                         absence_path='data/test_absence.csv'
#                     )

#                     # Remove NaN and infinite values from test set
#                     X_test = np.array(X_test, dtype=float)
#                     mask = np.isfinite(X_test).all(axis=1)
#                     X_test = X_test[mask]

#                     if X_test.shape[0] == 0:  # If no valid samples remain
#                         print(f'No valid samples for {ecoregion_name}. Setting average probability to 0.')
#                         avg_probability = 0
#                     else:
#                         # Make predictions
#                         y_proba = clf.predict_proba(X_test)[:, 1]

#                         # Calculate the average probability
#                         avg_probability = y_proba.mean()
#             except TimeoutError:
#                 print(f'Timeout for {ecoregion_name}. Setting average probability to 0.')
#                 avg_probability = 0

#             # Write the result to the output file
#             with open(output_file, 'a') as out_file:
#                 out_file.write(f'{ecoregion_name},{avg_probability}\n')
#             cnt += 1
#             print(f'Done for ecoregion {cnt}')

#     print(f'Average probabilities saved to {output_file}')


#add issues on why some eco-regions were taking too long to find avg prob on.


# Function to process ecoregions and calculate their average probability
# Parameters:
#   - filename: Name of the WKT file containing the ecoregion polygons
#   - polygon_dir: Directory containing the WKT files
#   - clf: Trained classifier model
#   - Features_extractor: Object for extracting features from data
#   - modelss: Object containing data loading utilities
# Returns:
#   - float: Average probability of presence for the ecoregion
#   - Returns 0.0 if processing fails or times out




def test_model_on_all_ecoregions(clf, Features_extractor, modelss, output_file='data/avg_prob.txt', num_workers=16):
    polygon_dir = 'data/eco_regions_polygon'
    
    # Write the header for the output file if the file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as out_file:
            out_file.write('Ecoregion,Average_Probability\n')
    
    # Get all .wkt files
    wkt_files = [f for f in os.listdir(polygon_dir) if f.endswith('.wkt')]
    total_files = len(wkt_files)
    
    print(f'Starting processing of {total_files} ecoregions with {num_workers} workers')
    
    # Create a queue of files to process
    file_queue = queue.Queue()
    for filename in wkt_files:
        file_queue.put(filename)
    
    # Create a lock for file writing
    file_lock = threading.Lock()
    
    # Create a shared counter for completed files
    completed = [0]
    
    def worker():
        while not file_queue.empty():
            try:
                # Get a file from the queue with a timeout
                try:
                    filename = file_queue.get(timeout=1)
                except queue.Empty:
                    break
                
                # Process the file
                ecoregion_name = os.path.splitext(filename)[0]
                try:
                    avg_probability = process_single_ecoregion(
                        filename, polygon_dir, clf, Features_extractor, modelss
                    )
                except Exception as e:
                    print(f'Error processing {ecoregion_name}: {str(e)}')
                    avg_probability = 0.0
                
                # Write the result to the output file
                with file_lock:
                    with open(output_file, 'a') as out_file:
                        out_file.write(f'{ecoregion_name},{avg_probability}\n')
                    completed[0] += 1
                    print(f'Completed {completed[0]}/{total_files}: {ecoregion_name}')
                
                # Mark the task as done
                file_queue.task_done()
            except Exception as e:
                print(f'Worker error: {str(e)}')
    
    # Start worker threads
    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Wait for all files to be processed or timeout
    timeout = 60 * 60  # 1 hour timeout
    start_time = time.time()
    while completed[0] < total_files and time.time() - start_time < timeout:
        time.sleep(1)
    
    # Check if all files were processed
    if completed[0] < total_files:
        print(f'WARNING: Only {completed[0]}/{total_files} files were processed before timeout')
        
        # Process any remaining files sequentially
        remaining_files = []
        while not file_queue.empty():
            try:
                remaining_files.append(file_queue.get(timeout=1))
            except queue.Empty:
                break
        
        print(f'Processing remaining {len(remaining_files)} files sequentially')
        for filename in remaining_files:
            ecoregion_name = os.path.splitext(filename)[0]
            try:
                avg_probability = process_single_ecoregion(
                    filename, polygon_dir, clf, Features_extractor, modelss
                )
            except Exception as e:
                print(f'Error processing {ecoregion_name}: {str(e)}')
                avg_probability = 0.0
            
            with open(output_file, 'a') as out_file:
                out_file.write(f'{ecoregion_name},{avg_probability}\n')
            completed[0] += 1
            print(f'Completed {completed[0]}/{total_files}: {ecoregion_name}')
    
    print(f'All ecoregions processed. Average probabilities saved to {output_file}')




# Function to process a single ecoregion file and calculate average probability
# Parameters:
#   filename: Name of the ecoregion file
#   polygon_dir: Directory containing polygon files
#   clf: Classifier model
#   Features_extractor: Feature extraction utility
#   modelss: Model utilities for data loading
# Returns:
#   avg_probability: Average probability score for the ecoregion


def process_single_ecoregion(filename, polygon_dir, clf, Features_extractor, modelss):
    """Process a single ecoregion file and return the average probability."""
    ecoregion_name = os.path.splitext(filename)[0]
    polygon_path = os.path.join(polygon_dir, filename)
    
    # Read the polygon WKT
    with open(polygon_path, 'r') as file:
        polygon_wkt = file.read().strip()
    
    # Generate test data for the current ecoregion
    X_dissimilar = Features_extractor.add_features(
        utility.divide_polygon_to_grids(polygon_wkt, grid_size=1, points_per_cell=20)
    )
    
    # Create a unique temporary file for this thread
    thread_id = threading.get_ident()
    timestamp = int(time.time() * 1000)
    temp_file = f'data/temp_presence_{thread_id}_{timestamp}.csv'
    pd.DataFrame(X_dissimilar).to_csv(temp_file, index=False)
    
    try:
        X_test, y_test, _, _, _,bias_weights = modelss.load_data(
            presence_path=temp_file,
            absence_path='data/test_absence.csv'
        )
        
        # Remove NaN and infinite values from test set
        X_test = np.array(X_test, dtype=float)
        mask = np.isfinite(X_test).all(axis=1)
        X_test = X_test[mask]
        
        if X_test.shape[0] == 0:  # If no valid samples remain
            print(f'No valid samples for {ecoregion_name}. Setting average probability to 0.')
            avg_probability = 0
        else:
            # Make predictions
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            # Calculate the average probability
            avg_probability = y_proba.mean()
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
    
    return avg_probability







def perform_feature_sensitivity_analysis(model, X, feature_names):
    """
    Perform feature sensitivity analysis on the trained model.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
    """
    print("\nPerforming feature sensitivity analysis...")
    
    # Define feature ranges (min, max) for each feature
    feature_ranges = {
        'annual_mean_temperature': (-185, 293),  # in Celsius
        'mean_diurnal_range': (49, 163),         # in Celsius
        'isothermality': (19, 69),               # percentage
        'temperature_seasonality': (431, 11303),  # standard deviation
        'max_temperature_warmest_month': (-51, 434),  # in Celsius
        'min_temperature_coldest_month': (-369, 246),  # in Celsius
        'temperature_annual_range': (74, 425),    # in Celsius
        'mean_temperature_wettest_quarter': (-143, 339),  # in Celsius
        'mean_temperature_driest_quarter': (-275, 309),   # in Celsius
        'mean_temperature_warmest_quarter': (-97, 351),  # in Celsius
        'mean_temperature_coldest_quarter': (-300, 275),  # in Celsius
        'annual_precipitation': (51, 11401),      # in mm
        'precipitation_wettest_month': (7, 2949),  # in mm
        'precipitation_driest_month': (0, 81),     # in mm
        'precipitation_seasonality': (27, 172),    # coefficient of variation
        'precipitation_wettest_quarter': (18, 8019),  # in mm
        'precipitation_driest_quarter': (0, 282),     # in mm
        'precipitation_warmest_quarter': (10, 6090),  # in mm
        'precipitation_coldest_quarter': (0, 5162),   # in mm
        'aridity_index': (403, 65535),            # dimensionless
        'topsoil_ph': (0, 8.3),                   # pH units
        'subsoil_ph': (0, 8.3),                   # pH units
        'topsoil_texture': (0, 3),                # texture class
        'subsoil_texture': (0, 13),               # texture class
        'elevation': (-54, 7548)                  # in meters
    }
    
    # Initialize analyzer with feature ranges
    analyzer = FeatureSensitivityAnalyzer(model, feature_names, feature_ranges)
    
    # Find a high probability point
    try:
        base_point, base_prob = analyzer.find_high_probability_point(X, threshold=0.9)
        print(f"Found point with probability: {base_prob:.4f}")
    except ValueError as e:
        print(f"Warning: {e}. Using point with highest probability instead.")
        probs = model.predict_proba(X)[:, 1]
        best_idx = np.argmax(probs)
        base_point = X[best_idx]
        base_prob = probs[best_idx]
        print(f"Using point with probability: {base_prob:.4f}")
    
    # Analyze all features
    results = analyzer.analyze_all_features(base_point, X)
    
    # Plot sensitivity
    analyzer.plot_feature_sensitivity(
        results,
        save_path='outputs/testing_SDM_out/feature_sensitivity.png'
    )
    
    # Calculate and print feature importance
    importance_scores = analyzer.get_feature_importance(results)
    print("\nFeature Importance Scores:")
    for feature, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {score:.4f}")

def compute_rank_correlation(prob_file, similarity_file, similarity_col, output_csv):
    """
    Compute Spearman rank correlation between similarity and average probability for ecoregions.
    Args:
        prob_file: CSV file with columns ['Ecoregion', 'Average_Probability']
        similarity_file: CSV file with columns ['Ecoregion', similarity_col]
        similarity_col: Name of the similarity column to use (e.g., 'Euclidean_Similarity')
        output_csv: Path to save merged results with correlation value
    Returns:
        corr: Spearman rank correlation coefficient
        pval: p-value
    """
    prob_df = pd.read_csv(prob_file)
    sim_df = pd.read_csv(similarity_file)
    merged = pd.merge(prob_df, sim_df, on='Ecoregion')
    corr, pval = spearmanr(merged[similarity_col], merged['Average_Probability'])
    print(f"Spearman rank correlation: {corr:.4f} (p={pval:.4g})")
    merged['Rank_Correlation'] = corr
    merged.to_csv(output_csv, index=False)
    return corr, pval

def main():
    # Feature importance analysis is now run from Modules/models.py
    # No need to call perform_feature_importance_for_all_species from main.py
    pass

if __name__ == "__main__":
    main()

