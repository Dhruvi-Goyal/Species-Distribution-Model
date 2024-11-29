import ee 
from shapely.wkt import loads
import pandas as pd
import numpy as np
from Modules import presence_dataloader, features_extractor, LULC_filter, pseudo_absence_generator, models, Generate_Prob, utility
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
ee.Authenticate()
ee.Initialize(project='sigma-bay-425614-a6')

def main():
  
    Presence_dataloader = presence_dataloader.Presence_dataloader()
    Features_extractor = features_extractor.Feature_Extractor(ee)
    LULC_Filter = LULC_filter.LULC_Filter(ee)
    Pseudo_absence = pseudo_absence_generator.PseudoAbsences(ee)
    modelss = models.Models()
    # generate_prob = Generate_Prob.Generate_Prob(ee)
    
    
    # # raw_occurrences = Presence_dataloader.load_raw_presence_data()   #uncomment if want to use gbif api to generate presence points
    
    # unique_presences = Presence_dataloader.load_unique_lon_lats()
    # presences_filtered_LULC = LULC_Filter.filter_by_lulc(unique_presences)
    # presence_data_with_features  = Features_extractor.add_features(presences_filtered_LULC)
    # presence_data_with_features.to_csv('data/presence.csv',index=False,mode='w')

    # pseudo_absence_points_with_features = Pseudo_absence.generate_pseudo_absences(presence_data_with_features)
    # print('training model')
    # X,y,_,_,_ = modelss.load_data()
    # clf, X_test, y_test, y_pred, y_proba = modelss.RandomForest(X,y)
    # print('done training')
    

    # y_pred = clf.predict(X_test)
    # metrics = {
    #         'accuracy': accuracy_score(y_test, y_pred),
    #         'confusion_matrix': confusion_matrix(y_test, y_pred),
    #         'classification_report': classification_report(y_test, y_pred)
    #     }
        
    # # Print the results

    # print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print("\nConfusion Matrix:")
    # print(metrics['confusion_matrix'])
    # print("\nClassification Report:")
    # print(metrics['classification_report'])
    # print('done predicting')

    # prob_map, transform = generate_prob.predict_eco_region(clf)
   
   
    # print(pseudo_absence_points_with_features.head(5))
    # pseudo_absence_points_with_features.to_csv('data/pseudo_absence.csv', index=False)

    feature_vectors_df = utility.find_representive_vectors_from_files('data/eco_regions_polygon', ee)
    
    # Step 2: Calculate similarity matrices
    feature_vectors_df = pd.read_csv('data/representative_vectors_eco_region_wise.csv', index_col=0)
    cosine_similarity_matrix = utility.calculate_cosine_similarity_matrix(feature_vectors_df)
    euclidean_similarity_matrix = utility.calculate_euclidean_similarity_matrix(feature_vectors_df)
    
    row_labels = feature_vectors_df.index.tolist()
    
    # # Print results
    # print("Cosine Similarity Matrix:")
    # cosine_df = pd.DataFrame(
    #     cosine_similarity_matrix, 
    #     index=row_labels, 
    #     columns=row_labels
    # )
    # print(cosine_df)
    
    # print("\nEuclidean Similarity Matrix:")
    # euclidean_df = pd.DataFrame(
    #     euclidean_similarity_matrix, 
    #     index=row_labels, 
    #     columns=row_labels
    # )
    # print(euclidean_df)
    
    # # Save matrices to text files
    utility.save_matrix_to_text(
        cosine_similarity_matrix, 
        'data/cosine_similarity_matrix.txt', 
        row_labels
    )
    utility.save_matrix_to_text(
        euclidean_similarity_matrix, 
        'data/euclidean_similarity_matrix.txt', 
        row_labels


    )

    # Example usage:
    # input_file = "data/eco_region_wise_genus.csv"  # Replace with your cleaned input file path
    # utility.jaccard_similarity(input_file)
    # with open('data/eco_regions_polygon/Terai_Duar_savanna_and_grasslands.wkt', 'r') as file:
    #     polygon_wkt1 = file.read().strip()
    #     # print(polygon_wkt)
    
    # with open('data/eco_regions_polygon/Northwestern_Himalayan_alpine_shrub_and_meadows.wkt', 'r') as file:
    #     polygon_wkt2 = file.read().strip()

    # X_dissimilar = Features_extractor.add_features(utility.divide_polygon_to_grids(polygon_wkt1,grid_size=1,points_per_cell=20))
    # pd.DataFrame.to_csv(X_dissimilar,'data/test_presence.csv')
    # X_test,y_test,_,_,_ = modelss.load_data(presence_path='data/test_presence.csv',absence_path='data/test_absence.csv')

    # print('predicting for a dissimilar reogionnn')
    # y_pred = clf.predict(X_test)
    # y_proba = clf.predict_proba(X_test)[:, 1]

    # print(f"Accuracy_RFC: {accuracy_score(y_test, y_pred):.4f}")
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))


    # print("\nProbabilities on the test set:")
    # for i, prob in enumerate(y_proba):
    #     print(f"Sample {i}: {prob:.4f}")


    # X_similar = Features_extractor.add_features(utility.divide_polygon_to_grids(polygon_wkt2,grid_size=1,points_per_cell=20))
    # print(X_dissimilar)
    # print(X_similar)



    return 

    
   



if __name__ == "__main__":
    main()

