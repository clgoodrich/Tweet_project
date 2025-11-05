"""
ArcGIS Pro 3.5 / arcpy Translation of test.ipynb
Replicates ALL notebook functionality for tweet spatial analysis with hierarchical cascade

Input data required:
- helene.geojson: Tweet points with GPE, time fields
- cities1000.csv: US cities data
- cb_2023_us_state_20m.shp: US States
- cb_2023_us_county_20m.shp: US Counties

Output: tw_project.gdb with temporal (4-hour binned) incremental and cumulative counts
"""

import arcpy
import os
from datetime import datetime, timedelta
import re
from fuzzywuzzy import fuzz, process

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = r"C:\users\colto\documents\github\tweet_project"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# Input paths
GEOJSON_PATH = os.path.join(DATA_ROOT, "geojson", "helene.geojson")
CITIES_CSV = os.path.join(DATA_ROOT, "tables", "cities1000.csv")
STATES_SHP = os.path.join(DATA_ROOT, "shape_files", "cb_2023_us_state_20m.shp")
COUNTIES_SHP = os.path.join(DATA_ROOT, "shape_files", "cb_2023_us_county_20m.shp")

# Output geodatabase
GDB_PATH = os.path.join(PROJECT_ROOT, "tw_project.gdb")
SCRATCH_GDB = os.path.join(PROJECT_ROOT, "scratch.gdb")

# Output directory for shapefiles
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "arcgis_outputs")
TEMPORAL_DIR = os.path.join(OUTPUT_DIR, "temporal_4hour_bins")
INCREMENTAL_DIR = os.path.join(TEMPORAL_DIR, "incremental")
CUMULATIVE_DIR = os.path.join(TEMPORAL_DIR, "cumulative")

# Spatial Reference
SR_WGS84 = arcpy.SpatialReference(4326)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Set up ArcGIS environment and create geodatabases"""
    print("="*80)
    print("SETTING UP ARCGIS ENVIRONMENT")
    print("="*80)

    # Create geodatabases
    if not arcpy.Exists(GDB_PATH):
        print(f"\nCreating geodatabase: {GDB_PATH}")
        arcpy.management.CreateFileGDB(os.path.dirname(GDB_PATH), os.path.basename(GDB_PATH))

    if not arcpy.Exists(SCRATCH_GDB):
        print(f"Creating scratch geodatabase: {SCRATCH_GDB}")
        arcpy.management.CreateFileGDB(os.path.dirname(SCRATCH_GDB), os.path.basename(SCRATCH_GDB))

    # Set environment
    arcpy.env.workspace = GDB_PATH
    arcpy.env.scratchWorkspace = SCRATCH_GDB
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = SR_WGS84

    # Create output directories
    os.makedirs(INCREMENTAL_DIR, exist_ok=True)
    os.makedirs(CUMULATIVE_DIR, exist_ok=True)

    print(f"\nWorkspace: {arcpy.env.workspace}")
    print(f"Scratch: {arcpy.env.scratchWorkspace}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nEnvironment setup complete!")

# ============================================================================
# DATA IMPORT
# ============================================================================

def import_data():
    """Import all input data to geodatabase"""
    print("\n" + "="*80)
    print("IMPORTING DATA TO GEODATABASE")
    print("="*80)

    # Import tweets from GeoJSON
    print("\n1. Importing helene.geojson as tweet points...")
    tweets_fc = os.path.join(GDB_PATH, "tweets_helene")
    if not arcpy.Exists(tweets_fc):
        arcpy.conversion.JSONToFeatures(GEOJSON_PATH, tweets_fc)
        print(f"   Created: {tweets_fc}")
        count = int(arcpy.management.GetCount(tweets_fc)[0])
        print(f"   Features: {count}")
    else:
        print(f"   Already exists: {tweets_fc}")

    # Import states
    print("\n2. Importing US States...")
    states_fc = os.path.join(GDB_PATH, "us_states")
    if not arcpy.Exists(states_fc):
        arcpy.conversion.FeatureClassToFeatureClass(STATES_SHP, GDB_PATH, "us_states")
        print(f"   Created: {states_fc}")
        count = int(arcpy.management.GetCount(states_fc)[0])
        print(f"   Features: {count}")
    else:
        print(f"   Already exists: {states_fc}")

    # Import counties
    print("\n3. Importing US Counties...")
    counties_fc = os.path.join(GDB_PATH, "us_counties")
    if not arcpy.Exists(counties_fc):
        arcpy.conversion.FeatureClassToFeatureClass(COUNTIES_SHP, GDB_PATH, "us_counties")
        print(f"   Created: {counties_fc}")
        count = int(arcpy.management.GetCount(counties_fc)[0])
        print(f"   Features: {count}")
    else:
        print(f"   Already exists: {counties_fc}")

    # Import cities from CSV and create points
    print("\n4. Importing US Cities from CSV...")
    cities_fc = os.path.join(GDB_PATH, "us_cities")
    if not arcpy.Exists(cities_fc):
        # First convert CSV to table
        cities_table = os.path.join(GDB_PATH, "cities_table")
        arcpy.conversion.TableToTable(CITIES_CSV, GDB_PATH, "cities_table")

        # Filter to US cities and create points
        # Based on notebook: country_code == 'US', feature_class == 'P', population not null
        cities_layer = arcpy.management.MakeTableView(cities_table, "cities_view",
            "country_code = 'US' AND feature_class = 'P' AND population IS NOT NULL AND latitude IS NOT NULL AND longitude IS NOT NULL")

        # Create feature class from XY
        arcpy.management.XYTableToPoint(cities_layer, cities_fc,
                                       "longitude", "latitude",
                                       coordinate_system=SR_WGS84)

        print(f"   Created: {cities_fc}")
        count = int(arcpy.management.GetCount(cities_fc)[0])
        print(f"   Features: {count}")

        # Clean up
        arcpy.management.Delete(cities_table)
    else:
        print(f"   Already exists: {cities_fc}")

    print("\nData import complete!")
    return tweets_fc, states_fc, counties_fc, cities_fc

# ============================================================================
# TEXT NORMALIZATION (preprocess_place_name from notebook)
# ============================================================================

def add_normalized_fields():
    """Add normalized text fields to all feature classes for matching"""
    print("\n" + "="*80)
    print("ADDING NORMALIZED TEXT FIELDS")
    print("="*80)

    tweets_fc = os.path.join(GDB_PATH, "tweets_helene")
    states_fc = os.path.join(GDB_PATH, "us_states")
    counties_fc = os.path.join(GDB_PATH, "us_counties")
    cities_fc = os.path.join(GDB_PATH, "us_cities")

    # Add normalized GPE field to tweets
    print("\n1. Adding normalized GPE field to tweets...")
    if "GPE_NORM" not in [f.name for f in arcpy.ListFields(tweets_fc)]:
        arcpy.management.AddField(tweets_fc, "GPE_NORM", "TEXT", field_length=500)

    # Normalization code block (replicates preprocess_place_name)
    normalize_code = """
import re

def normalize(text):
    if not text or text in ['', 'NAN']:
        return None
    text = str(text).upper().strip()
    # Replace abbreviations
    text = re.sub(r'\\bST\\.?\\b', 'SAINT', text)
    text = re.sub(r'\\bMT\\.?\\b', 'MOUNT', text)
    text = re.sub(r'\\bFT\\.?\\b', 'FORT', text)
    # Remove punctuation
    text = re.sub(r'[^\\w\\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\\s+', ' ', text)
    return text.strip()
"""

    arcpy.management.CalculateField(tweets_fc, "GPE_NORM", "normalize(!GPE!)",
                                   "PYTHON3", normalize_code)
    print("   Normalized GPE field")

    # Add normalized NAME fields to states
    print("\n2. Adding normalized NAME field to states...")
    if "NAME_NORM" not in [f.name for f in arcpy.ListFields(states_fc)]:
        arcpy.management.AddField(states_fc, "NAME_NORM", "TEXT", field_length=100)

    arcpy.management.CalculateField(states_fc, "NAME_NORM", "normalize(!NAME!)",
                                   "PYTHON3", normalize_code)

    # Also normalize STUSPS
    if "STUSPS_NORM" not in [f.name for f in arcpy.ListFields(states_fc)]:
        arcpy.management.AddField(states_fc, "STUSPS_NORM", "TEXT", field_length=10)

    arcpy.management.CalculateField(states_fc, "STUSPS_NORM", "normalize(!STUSPS!)",
                                   "PYTHON3", normalize_code)
    print("   Normalized state names")

    # Add normalized NAME field to counties
    print("\n3. Adding normalized NAME field to counties...")
    if "NAME_NORM" not in [f.name for f in arcpy.ListFields(counties_fc)]:
        arcpy.management.AddField(counties_fc, "NAME_NORM", "TEXT", field_length=100)

    arcpy.management.CalculateField(counties_fc, "NAME_NORM", "normalize(!NAME!)",
                                   "PYTHON3", normalize_code)
    print("   Normalized county names")

    # Add normalized name field to cities
    print("\n4. Adding normalized name field to cities...")
    if "name_norm" not in [f.name for f in arcpy.ListFields(cities_fc)]:
        arcpy.management.AddField(cities_fc, "name_norm", "TEXT", field_length=200)

    arcpy.management.CalculateField(cities_fc, "name_norm", "normalize(!name!)",
                                   "PYTHON3", normalize_code)
    print("   Normalized city names")

    print("\nNormalization complete!")

# ============================================================================
# LOOKUP DICTIONARIES (create_lookup_dictionaries from notebook)
# ============================================================================

def build_lookup_dictionaries():
    """Build lookup dictionaries for fuzzy matching (replicates notebook function)"""
    print("\n" + "="*80)
    print("BUILDING LOOKUP DICTIONARIES")
    print("="*80)

    states_fc = os.path.join(GDB_PATH, "us_states")
    counties_fc = os.path.join(GDB_PATH, "us_counties")
    cities_fc = os.path.join(GDB_PATH, "us_cities")

    # States lookup
    print("\n1. Building state lookup...")
    state_lookup = {}
    state_abbrev_to_name = {}

    with arcpy.da.SearchCursor(states_fc, ['NAME_NORM', 'STUSPS', 'STUSPS_NORM', 'STATEFP', 'SHAPE@']) as cursor:
        for row in cursor:
            name_norm = row[0]
            stusps = row[1]
            stusps_norm = row[2]
            statefp = row[3]
            shape = row[4]

            if name_norm:
                state_lookup[name_norm] = {
                    'STUSPS': stusps,
                    'STATEFP': statefp,
                    'NAME': name_norm,
                    'SHAPE': shape
                }

            if stusps_norm:
                state_abbrev_to_name[stusps_norm] = name_norm
                state_lookup[stusps_norm] = {
                    'STUSPS': stusps,
                    'STATEFP': statefp,
                    'NAME': name_norm,
                    'SHAPE': shape
                }

    print(f"   States in lookup: {len(state_lookup)}")

    # Counties lookup
    print("\n2. Building county lookup...")
    county_lookup = {}

    with arcpy.da.SearchCursor(counties_fc, ['NAME_NORM', 'GEOID', 'STATEFP', 'NAME', 'SHAPE@']) as cursor:
        for row in cursor:
            name_norm = row[0]
            geoid = row[1]
            statefp = row[2]
            name = row[3]
            shape = row[4]

            if name_norm:
                county_lookup[name_norm] = {
                    'GEOID': geoid,
                    'STATEFP': statefp,
                    'NAME': name,
                    'SHAPE': shape
                }

    print(f"   Counties in lookup: {len(county_lookup)}")

    # Cities lookup
    print("\n3. Building city lookup...")
    city_lookup = {}

    with arcpy.da.SearchCursor(cities_fc, ['name_norm', 'geonameid', 'name', 'population', 'SHAPE@']) as cursor:
        for row in cursor:
            name_norm = row[0]
            geonameid = row[1]
            name = row[2]
            population = row[3]
            shape = row[4]

            if name_norm:
                city_lookup[name_norm] = {
                    'geonameid': geonameid,
                    'name': name,
                    'population': population,
                    'SHAPE': shape
                }

    print(f"   Cities in lookup: {len(city_lookup)}")

    print("\nLookup dictionaries complete!")
    return state_lookup, county_lookup, city_lookup, state_abbrev_to_name

# ============================================================================
# GPE PARSING (parse_gpe_entities from notebook)
# ============================================================================

def parse_gpe_entities(gpe_string):
    """Split GPE field into individual place mentions (exact replica of notebook)"""
    if not gpe_string or gpe_string.strip() == '':
        return []

    gpe_string = str(gpe_string).strip()
    entities = []

    for part in [p.strip() for p in gpe_string.split(',')]:
        if not part:
            continue
        for sub in re.split(r'[;&|]', part):
            sub = sub.strip()
            if sub and len(sub) > 1:
                entities.append(sub)

    # Remove duplicates while preserving order
    seen = set()
    clean = []
    for e in entities:
        if e not in seen:
            clean.append(e)
            seen.add(e)

    return clean

# ============================================================================
# FUZZY MATCHING (fuzzy_match_entity from notebook)
# ============================================================================

def fuzzy_match_entity(entity, lookup_dict, threshold=85):
    """Fuzzy match entity to lookup dictionary (exact replica of notebook)"""
    if entity in lookup_dict:
        return lookup_dict[entity], 100

    names = list(lookup_dict.keys())
    if not names:
        return None, 0

    match = process.extractOne(entity, names, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return lookup_dict[match[0]], match[1]

    return None, 0

# ============================================================================
# TIME BINNING
# ============================================================================

def add_time_bins():
    """Add 4-hour time bin field to tweets (replicates dt.floor('4h'))"""
    print("\n" + "="*80)
    print("ADDING TIME BINS (4-HOUR)")
    print("="*80)

    tweets_fc = os.path.join(GDB_PATH, "tweets_helene")

    # Add bin_start field
    if "bin_start" not in [f.name for f in arcpy.ListFields(tweets_fc)]:
        arcpy.management.AddField(tweets_fc, "bin_start", "DATE")

    # Add bin_label field for display
    if "bin_label" not in [f.name for f in arcpy.ListFields(tweets_fc)]:
        arcpy.management.AddField(tweets_fc, "bin_label", "TEXT", field_length=50)

    # Calculate bin_start (floor to 4 hours)
    bin_code = """
from datetime import datetime, timedelta

def floor_to_4h(time_str):
    if not time_str:
        return None
    # Parse datetime
    dt = datetime.fromisoformat(str(time_str).replace('+00:00', ''))
    # Floor to 4 hours
    hour_floored = (dt.hour // 4) * 4
    dt_floored = dt.replace(hour=hour_floored, minute=0, second=0, microsecond=0)
    return dt_floored
"""

    arcpy.management.CalculateField(tweets_fc, "bin_start", "floor_to_4h(!time!)",
                                   "PYTHON3", bin_code)

    # Calculate bin_label for display
    arcpy.management.CalculateField(tweets_fc, "bin_label",
                                   "!bin_start!.strftime('%Y-%m-%d %H:%M:%S') if !bin_start! else ''",
                                   "PYTHON3")

    # Get unique bins
    bins = set()
    with arcpy.da.SearchCursor(tweets_fc, ['bin_start']) as cursor:
        for row in cursor:
            if row[0]:
                bins.add(row[0])

    bins = sorted(list(bins))
    print(f"\nTime bins created: {len(bins)}")
    if bins:
        print(f"Range: {bins[0]} to {bins[-1]}")

    print("\nTime binning complete!")
    return bins

# ============================================================================
# HIERARCHICAL CASCADE WITH TEMPORAL COUNTING
# (count_mentions_in_tweets_temporal_with_cascade from notebook)
# ============================================================================

def count_mentions_temporal_with_cascade(time_bins, state_lookup, county_lookup, city_lookup):
    """
    Count mentions by time bin WITH hierarchical cascade.
    Exact replica of notebook function.

    CASCADING RULES (from notebook):
    1. Tweet mentions are counted at the level mentioned (city/county/state)
    2. Tweet POINTS (lat/lon) are also spatially joined to add cascade counts:
       - Each tweet point finds its containing county → +1 to county
       - Each county cascades to its state → +1 to state
       - Each tweet point finds nearest city (within 50km) → +1 to city
    """
    print("\n" + "="*80)
    print("COUNTING MENTIONS BY TIME BIN WITH HIERARCHICAL CASCADE")
    print("="*80)

    tweets_fc = os.path.join(GDB_PATH, "tweets_helene")
    states_fc = os.path.join(GDB_PATH, "us_states")
    counties_fc = os.path.join(GDB_PATH, "us_counties")
    cities_fc = os.path.join(GDB_PATH, "us_cities")

    # Initialize dictionaries for each time bin
    temporal_state_mentions = {tb: {} for tb in time_bins}
    temporal_county_mentions = {tb: {} for tb in time_bins}
    temporal_city_mentions = {tb: {} for tb in time_bins}

    # Track tweet details
    temporal_state_details = {tb: {} for tb in time_bins}
    temporal_county_details = {tb: {} for tb in time_bins}
    temporal_city_details = {tb: {} for tb in time_bins}

    # Process tweets
    print(f"\nProcessing tweets...")

    idx = 0
    with arcpy.da.SearchCursor(tweets_fc, ['GPE_NORM', 'time', 'bin_start', 'GPE', 'SHAPE@']) as cursor:
        for row in cursor:
            if idx % 100 == 0:
                print(f"  Processing tweet {idx}...")

            gpe_norm = row[0]
            time_str = str(row[1]) if row[1] else ''
            bin_start = row[2]
            original_gpe = row[3] if row[3] else ''
            tweet_point = row[4]

            if not bin_start:
                idx += 1
                continue

            # Parse GPE entities
            entities = parse_gpe_entities(gpe_norm)

            # === PART 1: COUNT MENTIONS (text-based) ===
            for entity in entities:
                # Try state match (threshold 90)
                state_match, state_score = fuzzy_match_entity(entity, state_lookup, threshold=90)
                if state_match is not None:
                    state_code = state_match['STUSPS']
                    temporal_state_mentions[bin_start][state_code] = temporal_state_mentions[bin_start].get(state_code, 0) + 1

                    if state_code not in temporal_state_details[bin_start]:
                        temporal_state_details[bin_start][state_code] = []
                    temporal_state_details[bin_start][state_code].append({
                        'original_gpe': original_gpe,
                        'matched_entity': entity,
                        'time': time_str
                    })
                    continue

                # Try county match (threshold 85)
                county_match, county_score = fuzzy_match_entity(entity, county_lookup, threshold=85)
                if county_match is not None:
                    county_id = county_match['GEOID']
                    temporal_county_mentions[bin_start][county_id] = temporal_county_mentions[bin_start].get(county_id, 0) + 1

                    if county_id not in temporal_county_details[bin_start]:
                        temporal_county_details[bin_start][county_id] = []
                    temporal_county_details[bin_start][county_id].append({
                        'original_gpe': original_gpe,
                        'matched_entity': entity,
                        'time': time_str
                    })
                    continue

                # Try city match (threshold 85)
                city_match, city_score = fuzzy_match_entity(entity, city_lookup, threshold=85)
                if city_match is not None:
                    city_id = city_match['geonameid']
                    temporal_city_mentions[bin_start][city_id] = temporal_city_mentions[bin_start].get(city_id, 0) + 1

                    if city_id not in temporal_city_details[bin_start]:
                        temporal_city_details[bin_start][city_id] = []
                    temporal_city_details[bin_start][city_id].append({
                        'original_gpe': original_gpe,
                        'matched_entity': entity,
                        'time': time_str
                    })

            # === PART 2: CASCADE FROM TWEET POINT (spatial-based) ===
            # Find containing county
            containing_county = None
            with arcpy.da.SearchCursor(counties_fc, ['GEOID', 'STATEFP', 'NAME', 'SHAPE@']) as county_cursor:
                for county_row in county_cursor:
                    county_geoid = county_row[0]
                    county_statefp = county_row[1]
                    county_name = county_row[2]
                    county_shape = county_row[3]

                    if county_shape.contains(tweet_point):
                        containing_county = {
                            'GEOID': county_geoid,
                            'STATEFP': county_statefp,
                            'NAME': county_name
                        }
                        break

            if containing_county:
                county_geoid = containing_county['GEOID']
                county_statefp = containing_county['STATEFP']
                county_name = containing_county['NAME']

                # CASCADE: Increment county count
                temporal_county_mentions[bin_start][county_geoid] = temporal_county_mentions[bin_start].get(county_geoid, 0) + 1

                if county_geoid not in temporal_county_details[bin_start]:
                    temporal_county_details[bin_start][county_geoid] = []
                temporal_county_details[bin_start][county_geoid].append({
                    'original_gpe': f'[CASCADE from point in {county_name}]',
                    'matched_entity': f'{county_name} County',
                    'time': time_str
                })

                # CASCADE: Find containing state
                with arcpy.da.SearchCursor(states_fc, ['STUSPS', 'NAME', 'STATEFP']) as state_cursor:
                    for state_row in state_cursor:
                        if state_row[2] == county_statefp:
                            state_code = state_row[0]
                            state_name = state_row[1]

                            # CASCADE: Increment state count
                            temporal_state_mentions[bin_start][state_code] = temporal_state_mentions[bin_start].get(state_code, 0) + 1

                            if state_code not in temporal_state_details[bin_start]:
                                temporal_state_details[bin_start][state_code] = []
                            temporal_state_details[bin_start][state_code].append({
                                'original_gpe': f'[CASCADE from point in {state_name}]',
                                'matched_entity': state_name,
                                'time': time_str
                            })
                            break

            # CASCADE: Find nearest city (within 50km)
            # Buffer ~50km = 0.45 degrees
            buffer_geom = tweet_point.buffer(0.45)

            closest_city = None
            min_distance = float('inf')

            with arcpy.da.SearchCursor(cities_fc, ['geonameid', 'name', 'SHAPE@']) as city_cursor:
                for city_row in city_cursor:
                    city_id = city_row[0]
                    city_name = city_row[1]
                    city_shape = city_row[2]

                    if buffer_geom.contains(city_shape):
                        distance = tweet_point.distanceTo(city_shape)
                        if distance < min_distance:
                            min_distance = distance
                            closest_city = {
                                'geonameid': city_id,
                                'name': city_name
                            }

            if closest_city:
                city_id = closest_city['geonameid']
                city_name = closest_city['name']

                # CASCADE: Increment city count
                temporal_city_mentions[bin_start][city_id] = temporal_city_mentions[bin_start].get(city_id, 0) + 1

                if city_id not in temporal_city_details[bin_start]:
                    temporal_city_details[bin_start][city_id] = []
                temporal_city_details[bin_start][city_id].append({
                    'original_gpe': f'[CASCADE from nearby point]',
                    'matched_entity': city_name,
                    'time': time_str
                })

            idx += 1

    print(f"\nProcessed {idx} tweets")
    print("\nCounting complete!")

    return (temporal_state_mentions, temporal_county_mentions, temporal_city_mentions,
            temporal_state_details, temporal_county_details, temporal_city_details)

# ============================================================================
# EXPORT TEMPORAL DATA (export_temporal_to_arcgis from notebook)
# ============================================================================

def export_temporal_data(time_bins, temporal_state_mentions, temporal_county_mentions,
                         temporal_city_mentions, temporal_state_details, temporal_county_details,
                         temporal_city_details):
    """
    Export temporal (4-hour binned) data for states, counties, and cities.
    Creates BOTH incremental and cumulative count files.
    Exact replica of notebook function.
    """
    print("\n" + "="*80)
    print("EXPORTING TEMPORAL DATA - INCREMENTAL & CUMULATIVE")
    print("="*80)

    states_fc = os.path.join(GDB_PATH, "us_states")
    counties_fc = os.path.join(GDB_PATH, "us_counties")
    cities_fc = os.path.join(GDB_PATH, "us_cities")

    print(f"\nTime bins: {len(time_bins)}")
    print(f"Output directory: {TEMPORAL_DIR}")

    # Track individual bin files for later merging
    incremental_bin_files = {'states': [], 'counties': [], 'cities': []}
    cumulative_bin_files = {'states': [], 'counties': [], 'cities': []}

    # Cumulative tracking dictionaries
    cumulative_state_counts = {}
    cumulative_county_counts = {}
    cumulative_city_counts = {}

    # First pass: collect all entities ever mentioned
    all_mentioned_states = set()
    all_mentioned_counties = set()
    all_mentioned_cities = set()

    for bin_time in time_bins:
        all_mentioned_states.update(temporal_state_mentions[bin_time].keys())
        all_mentioned_counties.update(temporal_county_mentions[bin_time].keys())
        all_mentioned_cities.update(temporal_city_mentions[bin_time].keys())

    print(f"\nEntities ever mentioned:")
    print(f"  States: {len(all_mentioned_states)}")
    print(f"  Counties: {len(all_mentioned_counties)}")
    print(f"  Cities: {len(all_mentioned_cities)}")

    # Process each time bin
    for idx, bin_time in enumerate(time_bins):
        bin_str = bin_time.strftime('%Y%m%d_%H%M')
        bin_label = bin_time.strftime('%Y-%m-%d %H:%M:%S')

        print(f"\n  Processing time bin {idx+1}/{len(time_bins)}: {bin_label}")

        # === STATES ===
        # Update cumulative counts
        for state_code, count in temporal_state_mentions[bin_time].items():
            cumulative_state_counts[state_code] = cumulative_state_counts.get(state_code, 0) + count

        # INCREMENTAL: Only states with mentions in THIS bin
        if temporal_state_mentions[bin_time]:
            inc_fc = os.path.join(GDB_PATH, f"states_inc_{bin_str}")
            arcpy.conversion.FeatureClassToFeatureClass(states_fc, GDB_PATH, f"states_inc_{bin_str}")

            # Add fields
            arcpy.management.AddField(inc_fc, "tweet_cnt", "LONG")
            arcpy.management.AddField(inc_fc, "smpl_gpe", "TEXT", field_length=254)
            arcpy.management.AddField(inc_fc, "time_bin", "TEXT", field_length=50)

            # Update counts
            with arcpy.da.UpdateCursor(inc_fc, ['STUSPS', 'tweet_cnt', 'smpl_gpe', 'time_bin']) as cursor:
                for row in cursor:
                    state_code = row[0]
                    if state_code in temporal_state_mentions[bin_time]:
                        row[1] = temporal_state_mentions[bin_time][state_code]
                        # Get sample GPE text
                        details = temporal_state_details[bin_time][state_code][:3]
                        row[2] = ' | '.join([d['original_gpe'][:100] for d in details])
                        row[3] = bin_label
                        cursor.updateRow(row)
                    else:
                        cursor.deleteRow()

            # Export to shapefile
            shp_path = os.path.join(INCREMENTAL_DIR, f"states_inc_{bin_str}.shp")
            arcpy.conversion.FeatureClassToShapefile(inc_fc, INCREMENTAL_DIR)
            incremental_bin_files['states'].append(shp_path)

            count = int(arcpy.management.GetCount(inc_fc)[0])
            print(f"    States incremental: {count} features")

        # CUMULATIVE: ALL states that have ever been mentioned
        cum_fc = os.path.join(GDB_PATH, f"states_cum_{bin_str}")
        arcpy.conversion.FeatureClassToFeatureClass(states_fc, GDB_PATH, f"states_cum_{bin_str}")

        arcpy.management.AddField(cum_fc, "cumul_cnt", "LONG")
        arcpy.management.AddField(cum_fc, "time_bin", "TEXT", field_length=50)

        with arcpy.da.UpdateCursor(cum_fc, ['STUSPS', 'cumul_cnt', 'time_bin']) as cursor:
            for row in cursor:
                state_code = row[0]
                if state_code in cumulative_state_counts:
                    row[1] = cumulative_state_counts[state_code]
                    row[2] = bin_label
                    cursor.updateRow()
                else:
                    cursor.deleteRow()

        shp_path = os.path.join(CUMULATIVE_DIR, f"states_cum_{bin_str}.shp")
        arcpy.conversion.FeatureClassToShapefile(cum_fc, CUMULATIVE_DIR)
        cumulative_bin_files['states'].append(shp_path)

        count = int(arcpy.management.GetCount(cum_fc)[0])
        print(f"    States cumulative: {count} features")

        # === COUNTIES ===
        # Update cumulative counts
        for county_id, count in temporal_county_mentions[bin_time].items():
            cumulative_county_counts[county_id] = cumulative_county_counts.get(county_id, 0) + count

        # INCREMENTAL
        if temporal_county_mentions[bin_time]:
            inc_fc = os.path.join(GDB_PATH, f"counties_inc_{bin_str}")
            arcpy.conversion.FeatureClassToFeatureClass(counties_fc, GDB_PATH, f"counties_inc_{bin_str}")

            arcpy.management.AddField(inc_fc, "tweet_cnt", "LONG")
            arcpy.management.AddField(inc_fc, "smpl_gpe", "TEXT", field_length=254)
            arcpy.management.AddField(inc_fc, "time_bin", "TEXT", field_length=50)

            with arcpy.da.UpdateCursor(inc_fc, ['GEOID', 'tweet_cnt', 'smpl_gpe', 'time_bin']) as cursor:
                for row in cursor:
                    geoid = row[0]
                    if geoid in temporal_county_mentions[bin_time]:
                        row[1] = temporal_county_mentions[bin_time][geoid]
                        details = temporal_county_details[bin_time][geoid][:3]
                        row[2] = ' | '.join([d['original_gpe'][:100] for d in details])
                        row[3] = bin_label
                        cursor.updateRow(row)
                    else:
                        cursor.deleteRow()

            shp_path = os.path.join(INCREMENTAL_DIR, f"counties_inc_{bin_str}.shp")
            arcpy.conversion.FeatureClassToShapefile(inc_fc, INCREMENTAL_DIR)
            incremental_bin_files['counties'].append(shp_path)

            count = int(arcpy.management.GetCount(inc_fc)[0])
            print(f"    Counties incremental: {count} features")

        # CUMULATIVE
        cum_fc = os.path.join(GDB_PATH, f"counties_cum_{bin_str}")
        arcpy.conversion.FeatureClassToFeatureClass(counties_fc, GDB_PATH, f"counties_cum_{bin_str}")

        arcpy.management.AddField(cum_fc, "cumul_cnt", "LONG")
        arcpy.management.AddField(cum_fc, "time_bin", "TEXT", field_length=50)

        with arcpy.da.UpdateCursor(cum_fc, ['GEOID', 'cumul_cnt', 'time_bin']) as cursor:
            for row in cursor:
                geoid = row[0]
                if geoid in cumulative_county_counts:
                    row[1] = cumulative_county_counts[geoid]
                    row[2] = bin_label
                    cursor.updateRow(row)
                else:
                    cursor.deleteRow()

        shp_path = os.path.join(CUMULATIVE_DIR, f"counties_cum_{bin_str}.shp")
        arcpy.conversion.FeatureClassToShapefile(cum_fc, CUMULATIVE_DIR)
        cumulative_bin_files['counties'].append(shp_path)

        count = int(arcpy.management.GetCount(cum_fc)[0])
        print(f"    Counties cumulative: {count} features")

        # === CITIES ===
        # Update cumulative counts
        for city_id, count in temporal_city_mentions[bin_time].items():
            cumulative_city_counts[city_id] = cumulative_city_counts.get(city_id, 0) + count

        # INCREMENTAL
        if temporal_city_mentions[bin_time]:
            inc_fc = os.path.join(GDB_PATH, f"cities_inc_{bin_str}")
            arcpy.conversion.FeatureClassToFeatureClass(cities_fc, GDB_PATH, f"cities_inc_{bin_str}")

            arcpy.management.AddField(inc_fc, "tweet_cnt", "LONG")
            arcpy.management.AddField(inc_fc, "mtchd_ent", "TEXT", field_length=254)
            arcpy.management.AddField(inc_fc, "orig_gpe", "TEXT", field_length=254)
            arcpy.management.AddField(inc_fc, "time_bin", "TEXT", field_length=50)

            with arcpy.da.UpdateCursor(inc_fc, ['geonameid', 'tweet_cnt', 'mtchd_ent', 'orig_gpe', 'time_bin']) as cursor:
                for row in cursor:
                    city_id = row[0]
                    if city_id in temporal_city_mentions[bin_time]:
                        row[1] = temporal_city_mentions[bin_time][city_id]
                        details = temporal_city_details[bin_time][city_id]
                        row[2] = '; '.join([d['matched_entity'] for d in details])[:254]
                        row[3] = ' | '.join([d['original_gpe'] for d in details])[:254]
                        row[4] = bin_label
                        cursor.updateRow(row)
                    else:
                        cursor.deleteRow()

            shp_path = os.path.join(INCREMENTAL_DIR, f"cities_inc_{bin_str}.shp")
            arcpy.conversion.FeatureClassToShapefile(inc_fc, INCREMENTAL_DIR)
            incremental_bin_files['cities'].append(shp_path)

            count = int(arcpy.management.GetCount(inc_fc)[0])
            print(f"    Cities incremental: {count} features")

        # CUMULATIVE
        cum_fc = os.path.join(GDB_PATH, f"cities_cum_{bin_str}")
        arcpy.conversion.FeatureClassToFeatureClass(cities_fc, GDB_PATH, f"cities_cum_{bin_str}")

        arcpy.management.AddField(cum_fc, "cumul_cnt", "LONG")
        arcpy.management.AddField(cum_fc, "time_bin", "TEXT", field_length=50)

        with arcpy.da.UpdateCursor(cum_fc, ['geonameid', 'cumul_cnt', 'time_bin']) as cursor:
            for row in cursor:
                city_id = row[0]
                if city_id in cumulative_city_counts:
                    row[1] = cumulative_city_counts[city_id]
                    row[2] = bin_label
                    cursor.updateRow(row)
                else:
                    cursor.deleteRow()

        shp_path = os.path.join(CUMULATIVE_DIR, f"cities_cum_{bin_str}.shp")
        arcpy.conversion.FeatureClassToShapefile(cum_fc, CUMULATIVE_DIR)
        cumulative_bin_files['cities'].append(shp_path)

        count = int(arcpy.management.GetCount(cum_fc)[0])
        print(f"    Cities cumulative: {count} features")

    # === CREATE MASTER FILES BY MERGING ===
    print(f"\n  Creating master files by merging shapefiles...")

    # INCREMENTAL MASTERS
    if incremental_bin_files['states']:
        print(f"    Merging {len(incremental_bin_files['states'])} state incremental files...")
        master_fc = os.path.join(GDB_PATH, "states_INCREMENTAL_ALL")
        arcpy.management.Merge(incremental_bin_files['states'], master_fc)
        shp_path = os.path.join(INCREMENTAL_DIR, "states_INCREMENTAL_ALL.shp")
        arcpy.conversion.FeatureClassToShapefile(master_fc, INCREMENTAL_DIR)
        count = int(arcpy.management.GetCount(master_fc)[0])
        print(f"    ✓ States incremental master: {count} records")

    if incremental_bin_files['counties']:
        print(f"    Merging {len(incremental_bin_files['counties'])} county incremental files...")
        master_fc = os.path.join(GDB_PATH, "counties_INCREMENTAL_ALL")
        arcpy.management.Merge(incremental_bin_files['counties'], master_fc)
        shp_path = os.path.join(INCREMENTAL_DIR, "counties_INCREMENTAL_ALL.shp")
        arcpy.conversion.FeatureClassToShapefile(master_fc, INCREMENTAL_DIR)
        count = int(arcpy.management.GetCount(master_fc)[0])
        print(f"    ✓ Counties incremental master: {count} records")

    if incremental_bin_files['cities']:
        print(f"    Merging {len(incremental_bin_files['cities'])} city incremental files...")
        master_fc = os.path.join(GDB_PATH, "cities_INCREMENTAL_ALL")
        arcpy.management.Merge(incremental_bin_files['cities'], master_fc)
        shp_path = os.path.join(INCREMENTAL_DIR, "cities_INCREMENTAL_ALL.shp")
        arcpy.conversion.FeatureClassToShapefile(master_fc, INCREMENTAL_DIR)
        count = int(arcpy.management.GetCount(master_fc)[0])
        print(f"    ✓ Cities incremental master: {count} records")

    # CUMULATIVE MASTERS
    if cumulative_bin_files['states']:
        print(f"    Merging {len(cumulative_bin_files['states'])} state cumulative files...")
        master_fc = os.path.join(GDB_PATH, "states_CUMULATIVE_ALL")
        arcpy.management.Merge(cumulative_bin_files['states'], master_fc)
        shp_path = os.path.join(CUMULATIVE_DIR, "states_CUMULATIVE_ALL.shp")
        arcpy.conversion.FeatureClassToShapefile(master_fc, CUMULATIVE_DIR)
        count = int(arcpy.management.GetCount(master_fc)[0])
        print(f"    ✓ States cumulative master: {count} records")

    if cumulative_bin_files['counties']:
        print(f"    Merging {len(cumulative_bin_files['counties'])} county cumulative files...")
        master_fc = os.path.join(GDB_PATH, "counties_CUMULATIVE_ALL")
        arcpy.management.Merge(cumulative_bin_files['counties'], master_fc)
        shp_path = os.path.join(CUMULATIVE_DIR, "counties_CUMULATIVE_ALL.shp")
        arcpy.conversion.FeatureClassToShapefile(master_fc, CUMULATIVE_DIR)
        count = int(arcpy.management.GetCount(master_fc)[0])
        print(f"    ✓ Counties cumulative master: {count} records")

    if cumulative_bin_files['cities']:
        print(f"    Merging {len(cumulative_bin_files['cities'])} city cumulative files...")
        master_fc = os.path.join(GDB_PATH, "cities_CUMULATIVE_ALL")
        arcpy.management.Merge(cumulative_bin_files['cities'], master_fc)
        shp_path = os.path.join(CUMULATIVE_DIR, "cities_CUMULATIVE_ALL.shp")
        arcpy.conversion.FeatureClassToShapefile(master_fc, CUMULATIVE_DIR)
        count = int(arcpy.management.GetCount(master_fc)[0])
        print(f"    ✓ Cities cumulative master: {count} records")

    print("\n" + "="*80)
    print("TEMPORAL EXPORT COMPLETE!")
    print("="*80)
    print(f"\nFiles saved to: {os.path.abspath(TEMPORAL_DIR)}")
    print(f"\nOutput structure:")
    print(f"  incremental/ - Counts for just that 4-hour bin")
    print(f"  cumulative/  - Running total (persists even if bin has 0 new mentions)")
    print(f"\nIndividual bin files + merged master *_ALL.shp files")
    print(f"\nTo use in ArcGIS Pro:")
    print(f"  1. Add *_INCREMENTAL_ALL.shp or *_CUMULATIVE_ALL.shp")
    print(f"  2. Enable time using 'time_bin' field")
    print(f"  3. Set time step to 4 hours")
    print(f"  4. Animate!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("ARCGIS PRO 3.5 TWEET PROCESSOR")
    print("Replication of test.ipynb")
    print("="*80)

    try:
        # 1. Setup environment
        setup_environment()

        # 2. Import data
        tweets_fc, states_fc, counties_fc, cities_fc = import_data()

        # 3. Add normalized fields
        add_normalized_fields()

        # 4. Build lookup dictionaries
        state_lookup, county_lookup, city_lookup, state_abbrev_to_name = build_lookup_dictionaries()

        # 5. Add time bins
        time_bins = add_time_bins()

        # 6. Count mentions with cascade
        (temporal_state_mentions, temporal_county_mentions, temporal_city_mentions,
         temporal_state_details, temporal_county_details, temporal_city_details) = \
            count_mentions_temporal_with_cascade(time_bins, state_lookup, county_lookup, city_lookup)

        # 7. Export temporal data
        export_temporal_data(time_bins, temporal_state_mentions, temporal_county_mentions,
                           temporal_city_mentions, temporal_state_details, temporal_county_details,
                           temporal_city_details)

        print("\n" + "="*80)
        print("PROCESSING COMPLETE!")
        print("="*80)
        print(f"\nSummary:")
        print(f"Total time bins: {len(time_bins)}")
        if time_bins:
            print(f"Time range: {time_bins[0].strftime('%Y-%m-%d %H:%M:%S')} to {time_bins[-1].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nGeodatabase: {GDB_PATH}")
        print(f"Output directory: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
