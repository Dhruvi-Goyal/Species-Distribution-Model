# Import necessary libraries for API requests, file handling, and data processing
import requests
import csv
from time import sleep
import pandas as pd

class Presence_dataloader():
    """
    A class for downloading and processing species occurrence data from GBIF.
    Now extended to fetch ALL species under the target genus.
    Includes rate limiting and retry logic to handle API limits.
    """

    def __init__(self):
        return

    def _get_species_under_genus(self, genus_name):
        """
        Query GBIF API to get all species that belong to a given genus.
        """
        print(f"\n🔍 Fetching all species under genus '{genus_name}' from GBIF...")
        species_list = []
        offset = 0
        limit = 300

        while True:
            params = {
                "q": genus_name,
                "rank": "SPECIES",
                "limit": limit,
                "offset": offset
            }
            url = "https://api.gbif.org/v1/species/search"

            # Retry logic for rate limiting
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = requests.get(url, params=params, timeout=30)
                    
                    if resp.status_code == 429:
                        wait_time = 60 * (attempt + 1)  # Exponential backoff: 60s, 120s, 180s
                        print(f"⏳ Rate limited. Waiting {wait_time} seconds before retry...")
                        sleep(wait_time)
                        continue
                    
                    resp.raise_for_status()
                    data = resp.json()
                    break  # Success, exit retry loop
                    
                except requests.exceptions.HTTPError as e:
                    if attempt == max_retries - 1:
                        print(f"⚠️ Error fetching species list after {max_retries} attempts: {e}")
                        return species_list
                except Exception as e:
                    print(f"⚠️ Error fetching species list: {e}")
                    return species_list
            
            # Add a small delay between successful requests
            sleep(0.5)

            results = data.get("results", [])
            if not results:
                break

            for sp in results:
                if sp.get("genus", "").lower() == genus_name.lower():
                    species_list.append((sp["scientificName"], sp["key"]))

            if data.get("endOfRecords", True):
                break

            offset += limit

        print(f"✅ Found {len(species_list)} species belonging to genus '{genus_name}'.")
        return species_list

    def load_raw_presence_data(self, maxp=2000):
        """
        Download species occurrence data from GBIF for ALL species in a given genus.
        """
        with open("Inputs/polygon.wkt", "r") as input_polygon:
            polygon_wkt = input_polygon.read().strip()

        with open("Inputs/genus_name.txt", "r") as genus:
            genus_name = genus.read().strip()

        occurrence_points = set()

        try:
            with open("data/presence.csv", "w") as presence_data:
                writer = csv.writer(presence_data)
                writer.writerow(["longitude", "latitude"])
        except FileNotFoundError:
            pass

        print(f"\n🌿 Starting data download for all species in genus '{genus_name}'")

        # Get all species under this genus
        species_list = self._get_species_under_genus(genus_name)

        if not species_list:
            print(f"⚠️ No species found for genus '{genus_name}'. Check GBIF spelling.")
            return occurrence_points

        # Iterate through all species in the genus
        for idx, (species_name, species_key) in enumerate(species_list, start=1):
            print(f"\n[{idx}/{len(species_list)}] 📍 Fetching occurrences for species: {species_name}")

            offset, limit = 0, 300
            total_species_points = 0

            while True:
                gbif_url = "https://api.gbif.org/v1/occurrence/search"
                params = {
                    "taxonKey": species_key,
                    "hasCoordinate": "true",
                    "country": "IN",
                    "limit": limit,
                    "offset": offset
                }

                # Retry logic with exponential backoff for rate limiting
                max_retries = 5
                success = False
                
                for attempt in range(max_retries):
                    try:
                        response = requests.get(gbif_url, params=params, timeout=30)
                        
                        if response.status_code == 429:
                            # Rate limited - wait before retrying
                            wait_time = 60 * (2 ** attempt)  # Exponential backoff: 60s, 120s, 240s, etc.
                            print(f"   ⏳ Rate limited. Waiting {wait_time} seconds (attempt {attempt + 1}/{max_retries})...")
                            sleep(wait_time)
                            continue
                        
                        response.raise_for_status()
                        data = response.json()
                        success = True
                        break  # Success, exit retry loop
                        
                    except requests.exceptions.HTTPError as e:
                        if response.status_code == 429 and attempt < max_retries - 1:
                            continue  # Retry on 429
                        print(f"⚠️ HTTP Error for {species_name}: {e}")
                        break
                    except Exception as e:
                        print(f"⚠️ Error fetching data for {species_name}: {e}")
                        break
                
                if not success:
                    print(f"   ❌ Skipping {species_name} after {max_retries} failed attempts")
                    break

                # Add small delay between successful requests to avoid triggering rate limits
                sleep(1)

                results = data.get("results", [])
                if not results:
                    break

                new_points = set()
                for result in results:
                    lon = result.get("decimalLongitude")
                    lat = result.get("decimalLatitude")
                    if lon is None or lat is None:
                        continue
                    try:
                        lon = float(lon)
                        lat = float(lat)
                        if -180 <= lon <= 180 and -90 <= lat <= 90:
                            point = (lon, lat)
                            if point not in occurrence_points:
                                new_points.add(point)
                                occurrence_points.add(point)
                    except:
                        continue

                if new_points:
                    with open("data/presence.csv", "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(new_points)
                    total_species_points += len(new_points)
                    print(f"   ↳ Added {len(new_points)} new points (total for species: {total_species_points})")

                if len(results) < limit:
                    break

                offset += limit
                if len(occurrence_points) >= maxp:
                    print(f"🛑 Reached maximum limit of {maxp} total points.")
                    break

            # Stop if global limit reached
            if len(occurrence_points) >= maxp:
                break

        print(f"\n✅ Download complete! Total unique points collected: {len(occurrence_points)}")
        return occurrence_points

    def load_unique_lon_lats(self):
        """
        Load and deduplicate occurrence data from saved CSV.
        """
        df = pd.read_csv("data/presence.csv")
        df_unique = df.drop_duplicates(subset=["longitude", "latitude"])
        return df_unique