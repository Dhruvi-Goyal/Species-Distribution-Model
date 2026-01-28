import pandas as pd 

class LULC_Filter:
    def __init__(self, ee):
        self.ee = ee
        # Load the dominant (mode) LULC image for the Malabar region
        self.modeLULC = self.load_modeLULC()

    def load_modeLULC(self):
        ee = self.ee

        # Load the RESOLVE ecoregions dataset
        ecoregions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')

        # Filter for the India boundary
        india = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
                  .filter(ee.Filter.eq('country_co', 'IN'))

        # Select only the "Malabar Coast moist forests" ecoregion
        malabar_ecoregion = ecoregions \
            .filterBounds(india) \
            .filter(ee.Filter.eq('ECO_NAME', 'Malabar Coast moist forests')) \
            .first()

        # Load Dynamic World LULC image collection and filter to this ecoregion
        lulc = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
            .filterBounds(malabar_ecoregion.geometry()) \
            .select('label')  # Use only the 'label' band

        # Compute the most frequent LULC class (mode) for each pixel across time
        modeLULC = lulc.mode().clip(malabar_ecoregion.geometry())

        return modeLULC

    def filter_by_lulc(self, df):
        """
        Filters input dataframe by retaining only those points that belong to a specific LULC class (e.g., 1 = Trees).
        """
        print(len(df), 'points to be filtered')
        ee = self.ee
        modeLULC = self.modeLULC

        # Convert input DataFrame to a list of Earth Engine Features
        features = []
        for _, row in df.iterrows():
            lon, lat = row['longitude'], row['latitude']
            point = ee.Geometry.Point([lon, lat])
            features.append(ee.Feature(point).set({'longitude': lon, 'latitude': lat}))

        # Convert features list to a FeatureCollection
        fc = ee.FeatureCollection(features)

        # Map over the feature collection to attach LULC class and filter by a specific label (e.g., 1 for forest)
        filtered_fc = fc.map(lambda feature: self.filter_point_by_lulc(feature)) \
                        .filter(ee.Filter.eq('lulc_label', 1))  # Only keep LULC = 1 (trees)

        # Convert the filtered EE FeatureCollection back to a pandas DataFrame
        filtered_df = self.fc_to_dataframe(filtered_fc)
        return filtered_df

    def filter_point_by_lulc(self, feature):
        """
        For a single feature (point), extract the LULC label from the mode LULC image.
        """
        ee = self.ee
        modeLULC = self.modeLULC

        # Create a point geometry from the feature's longitude and latitude
        point = ee.Geometry.Point([
            ee.Number(feature.get('longitude')),
            ee.Number(feature.get('latitude'))
        ])

        # Extract the LULC class from the mode image using a reducer
        lulc_value = modeLULC.reduceRegion(
            reducer=ee.Reducer.mode(),  # Use mode reducer even though it's already a mode image, for safety
            geometry=point,
            scale=10,  # Dynamic World has 10m resolution
            maxPixels=1e9
        ).get('label')

        # Attach the LULC label to the feature
        return feature.set('lulc_label', lulc_value)

    def fc_to_dataframe(self, fc):
        """
        Converts an Earth Engine FeatureCollection to a pandas DataFrame.
        """
        features = fc.getInfo()['features']
        data = []

        for feature in features:
            properties = feature['properties']
            data.append({
                'longitude': properties.get('longitude'),
                'latitude': properties.get('latitude'),
                'lulc_label': properties.get('lulc_label')
            })

        return pd.DataFrame(data)
