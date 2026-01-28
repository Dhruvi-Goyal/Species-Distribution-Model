import ee
import geemap
import geemap.ml as ml

# -----------------------------------
# Initialize Earth Engine
# -----------------------------------
ee.Authenticate()
ee.Initialize(project="ee-ictd-dhruvi")

print("✅ Earth Engine initialized")

# -----------------------------------
# Load trained RF model from GEE
# -----------------------------------
rf_asset_id = "projects/ee-ictd-dhruvi/assets/rf_Syzygium_20251226_131527"
print("Loading RF from:", rf_asset_id)

rf_fc = ee.FeatureCollection(rf_asset_id)
rf = ml.fc_to_classifier(rf_fc)

print("✅ RF classifier loaded")

# -----------------------------------
# Feature list (FIXED TYPO)
# -----------------------------------
feature_cols = [
    'annual_mean_temperature',
    'mean_diurnal_range',
    'isothermality',
    'temperature_seasonality',
    'max_temperature_warmest_month',
    'min_temperature_coldest_month',  # FIXED
    'temperature_annual_range',
    'mean_temperature_wettest_quarter',
    'mean_temperature_driest_quarter',
    'mean_temperature_warmest_quarter',
    'mean_temperature_coldest_quarter',
    'annual_precipitation',
    'precipitation_wettest_month',
    'precipitation_driest_month',
    'precipitation_seasonality',
    'precipitation_wettest_quarter',
    'precipitation_driest_quarter',
    'precipitation_warmest_quarter',
    'precipitation_coldest_quarter',
    'aridity_index',
    'topsoil_ph',
    'subsoil_ph',
    'topsoil_texture',
    'subsoil_texture',
    'elevation'
]

print("Feature count:", len(feature_cols))

# -----------------------------------
# Load predictor image
# -----------------------------------
predictor_asset = "projects/ee-ictd-dhruvi/assets/Syzygium_predictor_stack_named"
print("Loading predictor image:", predictor_asset)

predictor_img = ee.Image(predictor_asset)

# DEBUG: print available bands
available_bands = predictor_img.bandNames().getInfo()
print("✅ Predictor bands available:")
print(available_bands)

# DEBUG: check missing bands
missing = [b for b in feature_cols if b not in available_bands]
if missing:
    print("❌ MISSING BANDS:", missing)
    raise ValueError("Some required bands are missing from predictor image")

print("✅ All required bands found")

# -----------------------------------
# Classify
# -----------------------------------
traits_img = predictor_img.select(feature_cols).classify(rf)
print("✅ Classification completed")

# -----------------------------------
# Visualize
# -----------------------------------
Map = geemap.Map(center=[20, 78], zoom=4)

Map.addLayer(
    traits_img,
    {'min': 0, 'max': 1, 'palette': ['white', 'green']},
    'Syzygium Prediction'
)

print("✅ Map layer added")

Map
