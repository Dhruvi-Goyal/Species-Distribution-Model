import geopandas as gpd
import matplotlib.pyplot as plt

# 1. Point this at the main .shp file you downloaded:
shp_path = "Inputs/India_Country_Boundary.shp"

# 2. Read it in
india = gpd.read_file(shp_path)

# 3. Inspect a bit
print(india.crs)
print(india.head())

# 4. Plot it
ax = india.plot(figsize=(8, 6), linewidth=0.5)
ax.set_title("India Boundary")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()
