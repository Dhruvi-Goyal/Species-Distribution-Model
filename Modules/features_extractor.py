from shapely.geometry import Point, shape, Polygon
import pandas as pd 
import concurrent.futures

# Mapping of bioclim variable codes to human-readable names
bioclim_names = {
    'bio01': 'annual_mean_temperature',
    'bio02': 'mean_diurnal_range',
    'bio03': 'isothermality',
    'bio04': 'temperature_seasonality',
    'bio05': 'max_temperature_warmest_month',
    'bio06': 'min_temperature_coldest_month',
    'bio07': 'temperature_annual_range',
    'bio08': 'mean_temperature_wettest_quarter',
    'bio09': 'mean_temperature_driest_quarter',
    'bio10': 'mean_temperature_warmest_quarter',
    'bio11': 'mean_temperature_coldest_quarter',
    'bio12': 'annual_precipitation',
    'bio13': 'precipitation_wettest_month',
    'bio14': 'precipitation_driest_month',
    'bio15': 'precipitation_seasonality',
    'bio16': 'precipitation_wettest_quarter',
    'bio17': 'precipitation_driest_quarter',
    'bio18': 'precipitation_warmest_quarter',
    'bio19': 'precipitation_coldest_quarter'
}

class Feature_Extractor():
    def __init__(self, ee):
        self.ee = ee  # Earth Engine API
        self.assets = self.load_assets()  # Load satellite & GIS datasets
        self.min_max_values = self.get_region_min_max_features()  # Precompute min-max feature values for normalization

    def load_assets(self):
        """
        Loads all the remote sensing and environmental datasets from Earth Engine.
        """
        ee = self.ee
        ecoregions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
        india = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
            .filter(ee.Filter.eq('country_co', 'IN'))

        bioclim = ee.Image("WORLDCLIM/V1/BIO")  # Bioclimatic variables
        malabar_ecoregion = ecoregions.filterBounds(india) \
            .filter(ee.Filter.eq('ECO_NAME', 'Malabar Coast moist forests')).first()

        species_occurrences = ee.FeatureCollection("projects/sigma-bay-425614-a6/assets/Mangifera_Malabar_f")
        lulc = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(malabar_ecoregion.geometry()).select('label')
        modeLULC = lulc.mode().clip(malabar_ecoregion.geometry())

        # Additional datasets
        additional_images = {
            'annual_precipitation': ee.Image("projects/ee-plantationsitescores/assets/AnnualPrecipitation"),
            'aridity_index': ee.Image("projects/ee-plantationsitescores/assets/India-AridityIndex"),
            'topsoil_ph': ee.Image("projects/ee-plantationsitescores/assets/Raster-T_PH_H2O"),
            'subsoil_ph': ee.Image("projects/ee-plantationsitescores/assets/Raster-S_PH_H2O"),
            'topsoil_texture': ee.Image("projects/ee-plantationsitescores/assets/Raster-T_TEXTURE"),
            'subsoil_texture': ee.Image("projects/ee-plantationsitescores/assets/Raster-S_USDA_TEX_CLASS"),
            'elevation': ee.Image("USGS/SRTMGL1_003").select('elevation')
        }

        return {
            'bioclim': bioclim,
            'malabar_ecoregion': malabar_ecoregion,
            'species_occurrences': species_occurrences,
            'modeLULC': modeLULC,
            **additional_images
        }

    def get_feature_values_at_point(self, lat, lon):
        """
        Extracts environmental feature values from all datasets at a given latitude and longitude.
        """
        assets = self.assets
        ee = self.ee
        point = ee.Geometry.Point(lon, lat)
        all_values = {}

        try:
            # Extract bioclimatic variables
            bioclim_values = assets['bioclim'].reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=30,
                maxPixels=1e13,
                crs='EPSG:4326'
            ).getInfo()

            for bio_code, bio_name in bioclim_names.items():
                all_values[bio_name] = bioclim_values.get(bio_code, float('nan'))

            # Extract values from other raster datasets
            image_assets = {
                'aridity_index': assets['aridity_index'],
                'topsoil_ph': assets['topsoil_ph'],
                'subsoil_ph': assets['subsoil_ph'],
                'topsoil_texture': assets['topsoil_texture'],
                'subsoil_texture': assets['subsoil_texture'],
                'elevation': assets['elevation']
            }

            for name, asset in image_assets.items():
                try:
                    band = 'elevation' if name == 'elevation' else 'b1'
                    value = asset.reduceRegion(
                        reducer=ee.Reducer.first(),
                        geometry=point,
                        scale=10,
                        maxPixels=1e13,
                        crs='EPSG:4326'
                    ).get(band).getInfo()
                    all_values[name] = value if value is not None else float('nan')
                except Exception as e:
                    print(f"Error getting {name} value: {str(e)}")
                    all_values[name] = float('nan')

        except Exception as e:
            print(f"Error in get_feature_values_at_point: {str(e)}")
            return None

        return all_values

    def get_region_min_max_features(self):
        """
        Computes min/max feature values across the Malabar region for normalization.
        """
        assets = self.assets
        region = assets['malabar_ecoregion'].geometry()
        ee = self.ee
        bioclim_region = assets['bioclim']
        min_max_dict = {}

        for bio_code, bio_name in bioclim_names.items():
            try:
                band = bioclim_region.select([bio_code])
                min_val = band.reduceRegion(
                    reducer=ee.Reducer.min(), 
                    geometry=region, 
                    scale=500, 
                    maxPixels=1e13,
                    crs='EPSG:4326'
                ).getInfo().get(bio_code, float('nan'))
                max_val = band.reduceRegion(
                    reducer=ee.Reducer.max(), 
                    geometry=region, 
                    scale=500, 
                    maxPixels=1e13,
                    crs='EPSG:4326'
                ).getInfo().get(bio_code, float('nan'))
                min_max_dict[bio_name] = {'min': min_val, 'max': max_val}
            except Exception as e:
                print(f"Error getting min/max for {bio_name}: {str(e)}")
                min_max_dict[bio_name] = {'min': float('nan'), 'max': float('nan')}

        for name, asset in {
            'aridity_index': assets['aridity_index'],
            'topsoil_ph': assets['topsoil_ph'],
            'subsoil_ph': assets['subsoil_ph'],
            'topsoil_texture': assets['topsoil_texture'],
            'subsoil_texture': assets['subsoil_texture'],
            'elevation': assets['elevation']
        }.items():
            try:
                band = 'elevation' if name == 'elevation' else 'b1'
                min_val = asset.reduceRegion(
                    reducer=ee.Reducer.min(), 
                    geometry=region, 
                    scale=500, 
                    maxPixels=1e13,
                    crs='EPSG:4326'
                ).getInfo().get(band, float('nan'))
                max_val = asset.reduceRegion(
                    reducer=ee.Reducer.max(), 
                    geometry=region, 
                    scale=500, 
                    maxPixels=1e13,
                    crs='EPSG:4326'
                ).getInfo().get(band, float('nan'))
                min_max_dict[name] = {'min': min_val, 'max': max_val}
            except Exception as e:
                print(f"Error getting min/max for {name}: {str(e)}")
                min_max_dict[name] = {'min': float('nan'), 'max': float('nan')}

        return min_max_dict

    def normalize_bioclim_values(self, values_dict):
        """
        Normalizes feature values using min-max normalization.
        """
        min_max_dict = self.min_max_values

        for key in min_max_dict:
            if min_max_dict[key]['min'] is not None and min_max_dict[key]['max'] is not None and values_dict.get(key) is not None:
                min_max_dict[key]['min'] = min(min_max_dict[key]['min'], values_dict[key])
                min_max_dict[key]['max'] = max(min_max_dict[key]['max'], values_dict[key])

        self.min_max_values = min_max_dict

        normalized = {}
        cnt = 0
        for key, value in values_dict.items():
            if value is None:
                cnt += 1
                normalized[key] = None
                continue

            if not isinstance(value, (float, int)):
                print(f"Invalid type for {key}: {value}")
                continue

            min_val = min_max_dict[key]['min']
            max_val = min_max_dict[key]['max']
            if max_val - min_val == 0:
                normalized[key] = 0
            else:
                normalized[key] = (value - min_val) / (max_val - min_val)

        if cnt > 0:
            print(f"Warning: None value for {cnt}. Skipping normalization for these keys.")
        return normalized

    def process_point(self, row):
        """
        Handles one row (one geo-point) and returns its normalized feature values.
        """
        latitude = row['latitude']
        longitude = row['longitude']

        if pd.isna(latitude) or pd.isna(longitude):
            return None

        values = self.get_feature_values_at_point(latitude, longitude)
        normalized_values = self.normalize_bioclim_values(values)
        return {'longitude': longitude, 'latitude': latitude, **normalized_values}

    def add_features(self, occurrences, batch_size=4000):
        """
        Adds normalized environmental features to a batch of occurrence points.
        Uses parallel processing for efficiency.
        """
        total_size = occurrences.shape[0]
        num_batches = (total_size + batch_size - 1) // batch_size
        all_presence_points = []

        for i in range(num_batches):
            start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, total_size)
            batch_df = occurrences.iloc[start_idx:end_idx]

            # Process points in parallel using threads
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_point, row) for _, row in batch_df.iterrows()]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_presence_points.append(result)

        return pd.DataFrame(all_presence_points)
