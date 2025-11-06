"""
ARCGIS TWEET PROCESSOR - COMPLETE CONVERSION
Converts test.ipynb functionality entirely to ArcGIS Pro/arcpy environment

This script replicates ALL functionality from the Jupyter notebook:
1. Load tweets (GeoJSON), cities (CSV), states/counties (shapefiles)
2. Parse and fuzzy match place name entities (GPE field)
3. Count mentions by entity with spatial cascade
4. Create temporal bins (4-hour intervals)
5. Export incremental and cumulative shapefiles

Requirements: ArcGIS Pro 3.x with arcpy, fuzzywuzzy library
"""

import arcpy
import os
import re
from datetime import datetime
from collections import defaultdict
from fuzzywuzzy import fuzz, process

# Set overwrite output
arcpy.env.overwriteOutput = True


# ==============================================================================
# HELPER FUNCTIONS FOR PATH MANAGEMENT
# ==============================================================================

def get_project_root():
    """Get project root directory (current working directory)"""
    return os.getcwd()


def get_data_file_path(*path_segments):
    """Build path to data files from project root"""
    project_root = get_project_root()
    return os.path.join(project_root, *path_segments)


# ==============================================================================
# DATA LOADING FUNCTIONS - ARCPY VERSIONS
# ==============================================================================

def load_tweets_geojson(hurricane_name, workspace="in_memory"):
    """
    Load hurricane tweets as feature class
    Returns: Feature class path
    """
    print(f"Loading tweets from {hurricane_name}.geojson...")
    geojson_path = get_data_file_path('data', 'geojson', f'{hurricane_name}.geojson')

    # Convert GeoJSON to feature class
    tweets_fc = os.path.join(workspace, f"tweets_{hurricane_name}")
    arcpy.conversion.JSONToFeatures(geojson_path, tweets_fc)

    # Project to WGS84 if needed
    sr = arcpy.Describe(tweets_fc).spatialReference
    if sr.factoryCode != 4326:
        tweets_fc_wgs84 = os.path.join(workspace, f"tweets_{hurricane_name}_wgs84")
        arcpy.management.Project(tweets_fc, tweets_fc_wgs84, arcpy.SpatialReference(4326))
        tweets_fc = tweets_fc_wgs84

    count = int(arcpy.management.GetCount(tweets_fc).getOutput(0))
    print(f"  Loaded {count} tweet features")

    return tweets_fc


def load_cities_csv(workspace="in_memory"):
    """
    Load cities1000.csv and convert to point feature class
    Returns: Feature class path
    """
    print("Loading cities from CSV...")
    csv_path = get_data_file_path('data', 'tables', 'cities1000.csv')

    # Create feature layer from XY data
    cities_fc = os.path.join(workspace, "us_cities")

    # Use MakeXYEventLayer then copy to feature class
    arcpy.management.MakeXYEventLayer(
        csv_path,
        "longitude",
        "latitude",
        "cities_layer",
        arcpy.SpatialReference(4326)
    )

    # Filter to US cities
    arcpy.management.SelectLayerByAttribute(
        "cities_layer",
        "NEW_SELECTION",
        "country_code = 'US' AND feature_class = 'P' AND population IS NOT NULL"
    )

    # Copy to permanent feature class
    arcpy.management.CopyFeatures("cities_layer", cities_fc)

    count = int(arcpy.management.GetCount(cities_fc).getOutput(0))
    print(f"  Loaded {count} US city features")

    return cities_fc


def load_states_shapefile(workspace="in_memory"):
    """
    Load states shapefile and project to WGS84
    Returns: Feature class path
    """
    print("Loading states shapefile...")
    shp_path = get_data_file_path('data', 'shape_files', 'cb_2023_us_state_20m.shp')

    states_fc = os.path.join(workspace, "us_states")

    # Copy and project to WGS84
    arcpy.management.Project(shp_path, states_fc, arcpy.SpatialReference(4326))

    count = int(arcpy.management.GetCount(states_fc).getOutput(0))
    print(f"  Loaded {count} state features")

    return states_fc


def load_counties_shapefile(workspace="in_memory"):
    """
    Load counties shapefile and project to WGS84
    Returns: Feature class path
    """
    print("Loading counties shapefile...")
    shp_path = get_data_file_path('data', 'shape_files', 'cb_2023_us_county_20m.shp')

    counties_fc = os.path.join(workspace, "us_counties")

    # Copy and project to WGS84
    arcpy.management.Project(shp_path, counties_fc, arcpy.SpatialReference(4326))

    count = int(arcpy.management.GetCount(counties_fc).getOutput(0))
    print(f"  Loaded {count} county features")

    return counties_fc


# ==============================================================================
# PLACE NAME PREPROCESSING AND PARSING
# ==============================================================================

def preprocess_place_name(name):
    """Standardize place names for matching (same as notebook)"""
    if not name or name == 'NAN' or name == '':
        return None

    name = str(name).upper().strip()
    name = re.sub(r'\bST\.?\b', 'SAINT', name)
    name = re.sub(r'\bMT\.?\b', 'MOUNT', name)
    name = re.sub(r'\bFT\.?\b', 'FORT', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)

    return name.strip() if name.strip() else None


def parse_gpe_entities(gpe_string):
    """Split GPE field into individual place mentions (same as notebook)"""
    if not gpe_string or str(gpe_string).strip() == '':
        return []

    gpe_string = str(gpe_string).strip()
    entities = []

    for part in [p.strip() for p in gpe_string.split(',')]:
        if not part:
            continue
        for sub in re.split(r'[;&|]', part):
            sub = preprocess_place_name(sub)
            if sub and len(sub) > 1:
                entities.append(sub)

    # Remove duplicates while preserving order
    seen, clean = set(), []
    for e in entities:
        if e not in seen:
            clean.append(e)
            seen.add(e)

    return clean


# ==============================================================================
# CREATE LOOKUP DICTIONARIES FROM FEATURE CLASSES
# ==============================================================================

def create_lookup_dictionaries(states_fc, counties_fc, cities_fc):
    """
    Build name->attributes lookup dictionaries from feature classes
    Returns: state_lookup, county_lookup, city_lookup, state_abbrev_to_name
    """
    print("Building lookup dictionaries...")

    state_lookup = {}
    state_abbrev_to_name = {}
    county_lookup = {}
    city_lookup = {}

    # Build states lookup
    with arcpy.da.SearchCursor(states_fc, ['NAME', 'STUSPS', 'STATEFP', 'SHAPE@']) as cursor:
        for row in cursor:
            name = preprocess_place_name(row[0])
            if name:
                state_lookup[name] = {
                    'NAME': row[0],
                    'STUSPS': row[1],
                    'STATEFP': row[2],
                    'geometry': row[3]
                }

            abbr = str(row[1]).upper()
            state_abbrev_to_name[abbr] = name
            state_lookup[abbr] = {
                'NAME': row[0],
                'STUSPS': row[1],
                'STATEFP': row[2],
                'geometry': row[3]
            }

    # Build counties lookup
    with arcpy.da.SearchCursor(counties_fc, ['NAME', 'GEOID', 'STATEFP', 'SHAPE@']) as cursor:
        for row in cursor:
            name = preprocess_place_name(row[0])
            if name:
                county_lookup[name] = {
                    'NAME': row[0],
                    'GEOID': row[1],
                    'STATEFP': row[2],
                    'geometry': row[3]
                }

    # Build cities lookup
    with arcpy.da.SearchCursor(cities_fc, ['name', 'geonameid', 'population', 'SHAPE@']) as cursor:
        for row in cursor:
            name = preprocess_place_name(row[0])
            if name:
                city_lookup[name] = {
                    'name': row[0],
                    'geonameid': row[1],
                    'population': row[2],
                    'geometry': row[3]
                }

    print(f"  States: {len(state_lookup)}")
    print(f"  Counties: {len(county_lookup)}")
    print(f"  Cities: {len(city_lookup)}")

    return state_lookup, county_lookup, city_lookup, state_abbrev_to_name


# ==============================================================================
# FUZZY MATCHING
# ==============================================================================

def fuzzy_match_entity(entity, lookup_dict, threshold=85):
    """Fuzzy match entity to lookup dictionary (same as notebook)"""
    if entity in lookup_dict:
        return lookup_dict[entity], 100

    names = list(lookup_dict.keys())
    if not names:
        return None, 0

    match = process.extractOne(entity, names, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return lookup_dict[match[0]], match[1]

    return None, 0


# ==============================================================================
# COUNT MENTIONS WITH HIERARCHICAL CASCADE
# ==============================================================================

def count_mentions_with_cascade_temporal(tweets_fc, state_lookup, county_lookup, city_lookup,
                                         states_fc, counties_fc, cities_fc):
    """
    Count mentions by time bin WITH hierarchical cascade (matches notebook exactly)

    CASCADING RULES:
    1. Tweet mentions are counted at the level mentioned (city/county/state)
    2. Tweet POINTS (lat/lon) are also spatially joined to add cascade counts:
       - Each tweet point finds its containing county → +1 to county
       - Each county cascades to its state → +1 to state
       - Each tweet point finds nearest city (within 50km) → +1 to city

    Returns: time_bins, temporal_state_mentions, temporal_county_mentions, temporal_city_mentions,
             + detail dictionaries
    """
    print("\nCounting tweet mentions by time bin WITH HIERARCHICAL CASCADE...")

    # Get all tweets with their attributes
    tweets_data = []
    fields = ['SHAPE@', 'GPE', 'time']

    with arcpy.da.SearchCursor(tweets_fc, fields) as cursor:
        for row in cursor:
            tweets_data.append({
                'geometry': row[0],
                'GPE': row[1],
                'time': row[2]
            })

    print(f"  Processing {len(tweets_data)} tweets...")

    # Parse time and create 4-hour bins
    for tweet in tweets_data:
        time_str = str(tweet['time'])
        # Handle datetime parsing
        try:
            dt = datetime.fromisoformat(time_str.replace('+00:00', ''))
        except:
            dt = datetime.strptime(time_str.split('+')[0], '%Y-%m-%d %H:%M:%S')

        # Floor to 4-hour bin
        hour_bin = (dt.hour // 4) * 4
        tweet['bin'] = dt.replace(hour=hour_bin, minute=0, second=0, microsecond=0)
        tweet['time_dt'] = dt

    # Get unique time bins
    time_bins = sorted(set([t['bin'] for t in tweets_data]))
    print(f"  Time bins: {len(time_bins)}")

    # Initialize dictionaries for each time bin
    temporal_state_mentions = {tb: {} for tb in time_bins}
    temporal_county_mentions = {tb: {} for tb in time_bins}
    temporal_city_mentions = {tb: {} for tb in time_bins}

    temporal_state_details = {tb: {} for tb in time_bins}
    temporal_county_details = {tb: {} for tb in time_bins}
    temporal_city_details = {tb: {} for tb in time_bins}

    # Create spatial index for counties (for contains check)
    print("  Creating county spatial index...")
    county_geoms = {}
    with arcpy.da.SearchCursor(counties_fc, ['GEOID', 'STATEFP', 'NAME', 'SHAPE@']) as cursor:
        for row in cursor:
            county_geoms[row[0]] = {
                'geoid': row[0],
                'statefp': row[1],
                'name': row[2],
                'geometry': row[3]
            }

    # Create spatial index for states
    print("  Creating state spatial index...")
    state_geoms = {}
    with arcpy.da.SearchCursor(states_fc, ['STUSPS', 'NAME', 'STATEFP', 'SHAPE@']) as cursor:
        for row in cursor:
            state_geoms[row[2]] = {  # Key by STATEFP
                'stusps': row[0],
                'name': row[1],
                'statefp': row[2],
                'geometry': row[3]
            }

    # Create spatial index for cities
    print("  Creating city spatial index...")
    city_geoms = {}
    with arcpy.da.SearchCursor(cities_fc, ['geonameid', 'name', 'SHAPE@']) as cursor:
        for row in cursor:
            city_geoms[row[0]] = {
                'geonameid': row[0],
                'name': row[1],
                'geometry': row[2]
            }

    # Process each tweet
    for idx, tweet in enumerate(tweets_data):
        if idx % 100 == 0:
            print(f"    Processing tweet {idx}/{len(tweets_data)}")

        time_bin = tweet['bin']
        entities = parse_gpe_entities(tweet['GPE'])
        original_gpe = str(tweet['GPE']) if tweet['GPE'] else ''
        tweet_time = str(tweet['time'])
        tweet_point = tweet['geometry']

        # === PART 1: COUNT MENTIONS (text-based) ===
        for entity in entities:
            # Try state match
            state_match, state_score = fuzzy_match_entity(entity, state_lookup, threshold=90)
            if state_match is not None:
                state_code = state_match['STUSPS']
                temporal_state_mentions[time_bin][state_code] = temporal_state_mentions[time_bin].get(state_code, 0) + 1

                if state_code not in temporal_state_details[time_bin]:
                    temporal_state_details[time_bin][state_code] = []
                temporal_state_details[time_bin][state_code].append({
                    'original_gpe': original_gpe,
                    'matched_entity': entity,
                    'time': tweet_time
                })
                continue

            # Try county match
            county_match, county_score = fuzzy_match_entity(entity, county_lookup, threshold=85)
            if county_match is not None:
                county_id = county_match['GEOID']
                temporal_county_mentions[time_bin][county_id] = temporal_county_mentions[time_bin].get(county_id, 0) + 1

                if county_id not in temporal_county_details[time_bin]:
                    temporal_county_details[time_bin][county_id] = []
                temporal_county_details[time_bin][county_id].append({
                    'original_gpe': original_gpe,
                    'matched_entity': entity,
                    'time': tweet_time
                })
                continue

            # Try city match
            city_match, city_score = fuzzy_match_entity(entity, city_lookup, threshold=85)
            if city_match is not None:
                city_id = city_match['geonameid']
                temporal_city_mentions[time_bin][city_id] = temporal_city_mentions[time_bin].get(city_id, 0) + 1

                if city_id not in temporal_city_details[time_bin]:
                    temporal_city_details[time_bin][city_id] = []
                temporal_city_details[time_bin][city_id].append({
                    'original_gpe': original_gpe,
                    'matched_entity': entity,
                    'time': tweet_time
                })

        # === PART 2: CASCADE FROM TWEET POINT (spatial-based) ===
        # Find containing county
        containing_county = None
        for county_id, county_data in county_geoms.items():
            if county_data['geometry'].contains(tweet_point):
                containing_county = county_data
                break

        if containing_county:
            county_geoid = containing_county['geoid']
            county_statefp = containing_county['statefp']
            county_name = containing_county['name']

            # CASCADE: Increment county count
            temporal_county_mentions[time_bin][county_geoid] = temporal_county_mentions[time_bin].get(county_geoid, 0) + 1

            if county_geoid not in temporal_county_details[time_bin]:
                temporal_county_details[time_bin][county_geoid] = []
            temporal_county_details[time_bin][county_geoid].append({
                'original_gpe': f'[CASCADE from point in {county_name}]',
                'matched_entity': f'{county_name} County',
                'time': tweet_time
            })

            # CASCADE: Find containing state
            if county_statefp in state_geoms:
                state_data = state_geoms[county_statefp]
                state_code = state_data['stusps']
                state_name = state_data['name']

                # CASCADE: Increment state count
                temporal_state_mentions[time_bin][state_code] = temporal_state_mentions[time_bin].get(state_code, 0) + 1

                if state_code not in temporal_state_details[time_bin]:
                    temporal_state_details[time_bin][state_code] = []
                temporal_state_details[time_bin][state_code].append({
                    'original_gpe': f'[CASCADE from point in {state_name}]',
                    'matched_entity': state_name,
                    'time': tweet_time
                })

        # CASCADE: Find nearest city (within 50km)
        # Buffer point by ~50km (0.45 degrees)
        tweet_buffer = tweet_point.buffer(0.45)

        nearest_city = None
        min_distance = float('inf')

        for city_id, city_data in city_geoms.items():
            if tweet_buffer.contains(city_data['geometry']):
                distance = tweet_point.distanceTo(city_data['geometry'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_city = city_data

        if nearest_city:
            city_id = nearest_city['geonameid']
            city_name = nearest_city['name']

            # CASCADE: Increment city count
            temporal_city_mentions[time_bin][city_id] = temporal_city_mentions[time_bin].get(city_id, 0) + 1

            if city_id not in temporal_city_details[time_bin]:
                temporal_city_details[time_bin][city_id] = []
            temporal_city_details[time_bin][city_id].append({
                'original_gpe': '[CASCADE from nearby point]',
                'matched_entity': city_name,
                'time': tweet_time
            })

    print(f"\n  Found mentions across {len(time_bins)} time bins")

    return (time_bins, temporal_state_mentions, temporal_county_mentions, temporal_city_mentions,
            temporal_state_details, temporal_county_details, temporal_city_details)


# ==============================================================================
# EXPORT TEMPORAL DATA TO SHAPEFILES
# ==============================================================================

def export_temporal_to_shapefiles(hurricane_name, time_bins, temporal_state_mentions, temporal_county_mentions,
                                  temporal_city_mentions, temporal_state_details, temporal_county_details,
                                  temporal_city_details, states_fc, counties_fc, cities_fc,
                                  output_base_dir='arcgis_outputs'):
    """
    Export temporal (4-hour binned) data for states, counties, and cities.
    Creates BOTH incremental and cumulative count shapefiles (matches notebook exactly)
    """

    # Create temporal output directories for this hurricane
    output_dir = os.path.join(output_base_dir, hurricane_name)
    temporal_dir = os.path.join(output_dir, 'temporal_4hour_bins')
    incremental_dir = os.path.join(temporal_dir, 'incremental')
    cumulative_dir = os.path.join(temporal_dir, 'cumulative')

    os.makedirs(incremental_dir, exist_ok=True)
    os.makedirs(cumulative_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("EXPORTING TEMPORAL DATA - INCREMENTAL & CUMULATIVE")
    print("="*60)
    print(f"\nTime bins: {len(time_bins)}")
    print(f"Output directory: {temporal_dir}")

    # Track all entities that have ever been mentioned
    all_mentioned_states = set()
    all_mentioned_counties = set()
    all_mentioned_cities = set()

    # First pass: collect all entities
    for bin_time in time_bins:
        all_mentioned_states.update(temporal_state_mentions[bin_time].keys())
        all_mentioned_counties.update(temporal_county_mentions[bin_time].keys())
        all_mentioned_cities.update(temporal_city_mentions[bin_time].keys())

    print(f"\nEntities ever mentioned:")
    print(f"  States: {len(all_mentioned_states)}")
    print(f"  Counties: {len(all_mentioned_counties)}")
    print(f"  Cities: {len(all_mentioned_cities)}")

    # Cumulative tracking dictionaries
    cumulative_state_counts = {}
    cumulative_county_counts = {}
    cumulative_city_counts = {}

    # Lists to track individual bin files for merging
    incremental_bin_files = {'states': [], 'counties': [], 'cities': []}
    cumulative_bin_files = {'states': [], 'counties': [], 'cities': []}

    # Process each time bin
    for idx, bin_time in enumerate(time_bins):
        bin_str = bin_time.strftime('%Y%m%d_%H%M')
        bin_label = bin_time.strftime('%Y-%m-%d %H:%M:%S')

        print(f"\n  Processing time bin {idx+1}/{len(time_bins)}: {bin_label}")

        # Update cumulative counts
        for state_code, count in temporal_state_mentions[bin_time].items():
            cumulative_state_counts[state_code] = cumulative_state_counts.get(state_code, 0) + count

        for county_id, count in temporal_county_mentions[bin_time].items():
            cumulative_county_counts[county_id] = cumulative_county_counts.get(county_id, 0) + count

        for city_id, count in temporal_city_mentions[bin_time].items():
            cumulative_city_counts[city_id] = cumulative_city_counts.get(city_id, 0) + count

        # === EXPORT STATES ===
        # INCREMENTAL
        if temporal_state_mentions[bin_time]:
            states_inc_shp = os.path.join(incremental_dir, f'states_inc_{bin_str}.shp')
            export_states_incremental(states_fc, temporal_state_mentions[bin_time],
                                     temporal_state_details[bin_time], bin_label, states_inc_shp)
            incremental_bin_files['states'].append(states_inc_shp)

            inc_count = int(arcpy.management.GetCount(states_inc_shp).getOutput(0))
            print(f"    States incremental: {inc_count} features")

        # CUMULATIVE
        states_cum_shp = os.path.join(cumulative_dir, f'states_cum_{bin_str}.shp')
        export_states_cumulative(states_fc, cumulative_state_counts, bin_label, states_cum_shp)
        cumulative_bin_files['states'].append(states_cum_shp)

        cum_count = int(arcpy.management.GetCount(states_cum_shp).getOutput(0))
        print(f"    States cumulative: {cum_count} features")

        # === EXPORT COUNTIES ===
        # INCREMENTAL
        if temporal_county_mentions[bin_time]:
            counties_inc_shp = os.path.join(incremental_dir, f'counties_inc_{bin_str}.shp')
            export_counties_incremental(counties_fc, temporal_county_mentions[bin_time],
                                       temporal_county_details[bin_time], bin_label, counties_inc_shp)
            incremental_bin_files['counties'].append(counties_inc_shp)

            inc_count = int(arcpy.management.GetCount(counties_inc_shp).getOutput(0))
            print(f"    Counties incremental: {inc_count} features")

        # CUMULATIVE
        counties_cum_shp = os.path.join(cumulative_dir, f'counties_cum_{bin_str}.shp')
        export_counties_cumulative(counties_fc, cumulative_county_counts, bin_label, counties_cum_shp)
        cumulative_bin_files['counties'].append(counties_cum_shp)

        cum_count = int(arcpy.management.GetCount(counties_cum_shp).getOutput(0))
        print(f"    Counties cumulative: {cum_count} features")

        # === EXPORT CITIES ===
        # INCREMENTAL
        if temporal_city_mentions[bin_time]:
            cities_inc_shp = os.path.join(incremental_dir, f'cities_inc_{bin_str}.shp')
            export_cities_incremental(cities_fc, temporal_city_mentions[bin_time],
                                     temporal_city_details[bin_time], bin_label, cities_inc_shp)
            incremental_bin_files['cities'].append(cities_inc_shp)

            inc_count = int(arcpy.management.GetCount(cities_inc_shp).getOutput(0))
            print(f"    Cities incremental: {inc_count} features")

        # CUMULATIVE
        cities_cum_shp = os.path.join(cumulative_dir, f'cities_cum_{bin_str}.shp')
        export_cities_cumulative(cities_fc, cumulative_city_counts, bin_label, cities_cum_shp)
        cumulative_bin_files['cities'].append(cities_cum_shp)

        cum_count = int(arcpy.management.GetCount(cities_cum_shp).getOutput(0))
        print(f"    Cities cumulative: {cum_count} features")

    # === CREATE MASTER FILES BY MERGING ===
    print(f"\n  Creating master files by merging shapefiles...")

    # Merge incremental states
    if incremental_bin_files['states']:
        master_path = os.path.join(incremental_dir, 'states_INCREMENTAL_ALL.shp')
        arcpy.management.Merge(incremental_bin_files['states'], master_path)
        print(f"    ✓ States incremental master: {master_path}")

    # Merge cumulative states
    if cumulative_bin_files['states']:
        master_path = os.path.join(cumulative_dir, 'states_CUMULATIVE_ALL.shp')
        arcpy.management.Merge(cumulative_bin_files['states'], master_path)
        print(f"    ✓ States cumulative master: {master_path}")

    # Merge incremental counties
    if incremental_bin_files['counties']:
        master_path = os.path.join(incremental_dir, 'counties_INCREMENTAL_ALL.shp')
        arcpy.management.Merge(incremental_bin_files['counties'], master_path)
        print(f"    ✓ Counties incremental master: {master_path}")

    # Merge cumulative counties
    if cumulative_bin_files['counties']:
        master_path = os.path.join(cumulative_dir, 'counties_CUMULATIVE_ALL.shp')
        arcpy.management.Merge(cumulative_bin_files['counties'], master_path)
        print(f"    ✓ Counties cumulative master: {master_path}")

    # Merge incremental cities
    if incremental_bin_files['cities']:
        master_path = os.path.join(incremental_dir, 'cities_INCREMENTAL_ALL.shp')
        arcpy.management.Merge(incremental_bin_files['cities'], master_path)
        print(f"    ✓ Cities incremental master: {master_path}")

    # Merge cumulative cities
    if cumulative_bin_files['cities']:
        master_path = os.path.join(cumulative_dir, 'cities_CUMULATIVE_ALL.shp')
        arcpy.management.Merge(cumulative_bin_files['cities'], master_path)
        print(f"    ✓ Cities cumulative master: {master_path}")

    print(f"\n{'='*60}")
    print("TEMPORAL EXPORT COMPLETE!")
    print("="*60)
    print(f"\nFiles saved to: {os.path.abspath(temporal_dir)}")
    print(f"\nOutput structure:")
    print(f"  incremental/ - Counts for just that 4-hour bin")
    print(f"  cumulative/  - Running total (persists even if bin has 0 new mentions)")
    print(f"\nIndividual bin files + merged master *_ALL.shp files")
    print(f"\nTo use in ArcGIS Pro:")
    print(f"  1. Add *_INCREMENTAL_ALL.shp or *_CUMULATIVE_ALL.shp")
    print(f"  2. Enable time using 'time_bin' field")
    print(f"  3. Set time step to 4 hours")
    print(f"  4. Animate!")


# ==============================================================================
# HELPER FUNCTIONS FOR EXPORTING INDIVIDUAL ENTITY TYPES
# ==============================================================================

def export_states_incremental(states_fc, mention_counts, mention_details, bin_label, output_shp):
    """Export incremental state counts to shapefile"""
    # Create feature class with schema
    arcpy.management.CreateFeatureclass(
        os.path.dirname(output_shp),
        os.path.basename(output_shp),
        "POLYGON",
        spatial_reference=arcpy.SpatialReference(4326)
    )

    # Add fields
    arcpy.management.AddField(output_shp, "state_name", "TEXT", field_length=100)
    arcpy.management.AddField(output_shp, "state_code", "TEXT", field_length=2)
    arcpy.management.AddField(output_shp, "tweet_cnt", "LONG")
    arcpy.management.AddField(output_shp, "smpl_gpe", "TEXT", field_length=254)
    arcpy.management.AddField(output_shp, "time_bin", "TEXT", field_length=50)

    # Insert features
    with arcpy.da.InsertCursor(output_shp, ['SHAPE@', 'state_name', 'state_code', 'tweet_cnt', 'smpl_gpe', 'time_bin']) as ins_cursor:
        with arcpy.da.SearchCursor(states_fc, ['NAME', 'STUSPS', 'SHAPE@']) as cursor:
            for row in cursor:
                state_code = row[1]
                if state_code in mention_counts:
                    sample_gpe = ' | '.join([d['original_gpe'][:100] for d in mention_details[state_code][:3]])
                    ins_cursor.insertRow([
                        row[2],  # geometry
                        row[0],  # state name
                        state_code,
                        mention_counts[state_code],
                        sample_gpe[:254],
                        bin_label
                    ])


def export_states_cumulative(states_fc, cumulative_counts, bin_label, output_shp):
    """Export cumulative state counts to shapefile"""
    arcpy.management.CreateFeatureclass(
        os.path.dirname(output_shp),
        os.path.basename(output_shp),
        "POLYGON",
        spatial_reference=arcpy.SpatialReference(4326)
    )

    arcpy.management.AddField(output_shp, "state_name", "TEXT", field_length=100)
    arcpy.management.AddField(output_shp, "state_code", "TEXT", field_length=2)
    arcpy.management.AddField(output_shp, "cumul_cnt", "LONG")
    arcpy.management.AddField(output_shp, "time_bin", "TEXT", field_length=50)

    with arcpy.da.InsertCursor(output_shp, ['SHAPE@', 'state_name', 'state_code', 'cumul_cnt', 'time_bin']) as ins_cursor:
        with arcpy.da.SearchCursor(states_fc, ['NAME', 'STUSPS', 'SHAPE@']) as cursor:
            for row in cursor:
                state_code = row[1]
                if state_code in cumulative_counts:
                    ins_cursor.insertRow([
                        row[2],  # geometry
                        row[0],  # state name
                        state_code,
                        cumulative_counts[state_code],
                        bin_label
                    ])


def export_counties_incremental(counties_fc, mention_counts, mention_details, bin_label, output_shp):
    """Export incremental county counts to shapefile"""
    arcpy.management.CreateFeatureclass(
        os.path.dirname(output_shp),
        os.path.basename(output_shp),
        "POLYGON",
        spatial_reference=arcpy.SpatialReference(4326)
    )

    arcpy.management.AddField(output_shp, "cnty_name", "TEXT", field_length=100)
    arcpy.management.AddField(output_shp, "cnty_id", "TEXT", field_length=5)
    arcpy.management.AddField(output_shp, "state_fp", "TEXT", field_length=2)
    arcpy.management.AddField(output_shp, "tweet_cnt", "LONG")
    arcpy.management.AddField(output_shp, "smpl_gpe", "TEXT", field_length=254)
    arcpy.management.AddField(output_shp, "time_bin", "TEXT", field_length=50)

    with arcpy.da.InsertCursor(output_shp, ['SHAPE@', 'cnty_name', 'cnty_id', 'state_fp', 'tweet_cnt', 'smpl_gpe', 'time_bin']) as ins_cursor:
        with arcpy.da.SearchCursor(counties_fc, ['NAME', 'GEOID', 'STATEFP', 'SHAPE@']) as cursor:
            for row in cursor:
                county_id = row[1]
                if county_id in mention_counts:
                    sample_gpe = ' | '.join([d['original_gpe'][:100] for d in mention_details[county_id][:3]])
                    ins_cursor.insertRow([
                        row[3],  # geometry
                        row[0],  # county name
                        county_id,
                        row[2],  # state FP
                        mention_counts[county_id],
                        sample_gpe[:254],
                        bin_label
                    ])


def export_counties_cumulative(counties_fc, cumulative_counts, bin_label, output_shp):
    """Export cumulative county counts to shapefile"""
    arcpy.management.CreateFeatureclass(
        os.path.dirname(output_shp),
        os.path.basename(output_shp),
        "POLYGON",
        spatial_reference=arcpy.SpatialReference(4326)
    )

    arcpy.management.AddField(output_shp, "cnty_name", "TEXT", field_length=100)
    arcpy.management.AddField(output_shp, "cnty_id", "TEXT", field_length=5)
    arcpy.management.AddField(output_shp, "state_fp", "TEXT", field_length=2)
    arcpy.management.AddField(output_shp, "cumul_cnt", "LONG")
    arcpy.management.AddField(output_shp, "time_bin", "TEXT", field_length=50)

    with arcpy.da.InsertCursor(output_shp, ['SHAPE@', 'cnty_name', 'cnty_id', 'state_fp', 'cumul_cnt', 'time_bin']) as ins_cursor:
        with arcpy.da.SearchCursor(counties_fc, ['NAME', 'GEOID', 'STATEFP', 'SHAPE@']) as cursor:
            for row in cursor:
                county_id = row[1]
                if county_id in cumulative_counts:
                    ins_cursor.insertRow([
                        row[3],  # geometry
                        row[0],  # county name
                        county_id,
                        row[2],  # state FP
                        cumulative_counts[county_id],
                        bin_label
                    ])


def export_cities_incremental(cities_fc, mention_counts, mention_details, bin_label, output_shp):
    """Export incremental city counts to shapefile"""
    arcpy.management.CreateFeatureclass(
        os.path.dirname(output_shp),
        os.path.basename(output_shp),
        "POINT",
        spatial_reference=arcpy.SpatialReference(4326)
    )

    arcpy.management.AddField(output_shp, "city_name", "TEXT", field_length=200)
    arcpy.management.AddField(output_shp, "city_id", "LONG")
    arcpy.management.AddField(output_shp, "population", "LONG")
    arcpy.management.AddField(output_shp, "tweet_cnt", "LONG")
    arcpy.management.AddField(output_shp, "mtchd_ent", "TEXT", field_length=254)
    arcpy.management.AddField(output_shp, "orig_gpe", "TEXT", field_length=254)
    arcpy.management.AddField(output_shp, "time_bin", "TEXT", field_length=50)

    with arcpy.da.InsertCursor(output_shp, ['SHAPE@', 'city_name', 'city_id', 'population', 'tweet_cnt', 'mtchd_ent', 'orig_gpe', 'time_bin']) as ins_cursor:
        with arcpy.da.SearchCursor(cities_fc, ['name', 'geonameid', 'population', 'SHAPE@']) as cursor:
            for row in cursor:
                city_id = row[1]
                if city_id in mention_counts:
                    matched_entities = '; '.join([d['matched_entity'] for d in mention_details[city_id]])
                    orig_gpe = ' | '.join([d['original_gpe'] for d in mention_details[city_id]])

                    ins_cursor.insertRow([
                        row[3],  # geometry
                        row[0],  # city name
                        city_id,
                        row[2],  # population
                        mention_counts[city_id],
                        matched_entities[:254],
                        orig_gpe[:254],
                        bin_label
                    ])


def export_cities_cumulative(cities_fc, cumulative_counts, bin_label, output_shp):
    """Export cumulative city counts to shapefile"""
    arcpy.management.CreateFeatureclass(
        os.path.dirname(output_shp),
        os.path.basename(output_shp),
        "POINT",
        spatial_reference=arcpy.SpatialReference(4326)
    )

    arcpy.management.AddField(output_shp, "city_name", "TEXT", field_length=200)
    arcpy.management.AddField(output_shp, "city_id", "LONG")
    arcpy.management.AddField(output_shp, "population", "LONG")
    arcpy.management.AddField(output_shp, "cumul_cnt", "LONG")
    arcpy.management.AddField(output_shp, "time_bin", "TEXT", field_length=50)

    with arcpy.da.InsertCursor(output_shp, ['SHAPE@', 'city_name', 'city_id', 'population', 'cumul_cnt', 'time_bin']) as ins_cursor:
        with arcpy.da.SearchCursor(cities_fc, ['name', 'geonameid', 'population', 'SHAPE@']) as cursor:
            for row in cursor:
                city_id = row[1]
                if city_id in cumulative_counts:
                    ins_cursor.insertRow([
                        row[3],  # geometry
                        row[0],  # city name
                        city_id,
                        row[2],  # population
                        cumulative_counts[city_id],
                        bin_label
                    ])


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def process_hurricane(hurricane_name, workspace="in_memory"):
    """Process a single hurricane"""
    print("\n" + "="*60)
    print(f"PROCESSING HURRICANE: {hurricane_name.upper()}")
    print("="*60)
    print()

    # Load all data
    print("STEP 1: Loading Data")
    print("-" * 60)
    tweets_fc = load_tweets_geojson(hurricane_name, workspace)
    cities_fc = load_cities_csv(workspace)
    states_fc = load_states_shapefile(workspace)
    counties_fc = load_counties_shapefile(workspace)
    print()

    # Create lookup dictionaries
    print("STEP 2: Creating Lookup Dictionaries")
    print("-" * 60)
    state_lookup, county_lookup, city_lookup, state_abbrev_to_name = create_lookup_dictionaries(
        states_fc, counties_fc, cities_fc
    )
    print()

    # Count mentions with temporal binning and cascade
    print("STEP 3: Counting Mentions with Temporal Binning and Cascade")
    print("-" * 60)
    (time_bins, temporal_state_mentions, temporal_county_mentions, temporal_city_mentions,
     temporal_state_details, temporal_county_details, temporal_city_details) = \
        count_mentions_with_cascade_temporal(
            tweets_fc, state_lookup, county_lookup, city_lookup,
            states_fc, counties_fc, cities_fc
        )
    print()

    # Export temporal data
    print("STEP 4: Exporting Temporal Data")
    print("-" * 60)
    export_temporal_to_shapefiles(
        hurricane_name, time_bins, temporal_state_mentions, temporal_county_mentions, temporal_city_mentions,
        temporal_state_details, temporal_county_details, temporal_city_details,
        states_fc, counties_fc, cities_fc
    )
    print()

    print("="*60)
    print(f"PROCESSING COMPLETE FOR {hurricane_name.upper()}!")
    print("="*60)
    print(f"\nTime range: {time_bins[0].strftime('%Y-%m-%d %H:%M:%S')} to {time_bins[-1].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time bins: {len(time_bins)}")
    print(f"\nOutputs saved to: arcgis_outputs/{hurricane_name}/temporal_4hour_bins/")


def main():
    """Main execution function - processes all hurricanes"""

    print("="*60)
    print("ARCGIS TWEET PROCESSOR - COMPLETE CONVERSION")
    print("Processing multiple hurricanes")
    print("="*60)
    print()

    # Hurricane list
    hurricanes = ["helene", "francine"]

    # Set workspace
    workspace = "in_memory"

    for hurricane_name in hurricanes:
        process_hurricane(hurricane_name, workspace)

    print("\n" + "="*60)
    print("ALL HURRICANES PROCESSED SUCCESSFULLY!")
    print("="*60)
    print(f"\nProcessed hurricanes: {', '.join(hurricanes)}")
    print(f"\nAll outputs saved to: arcgis_outputs/")


if __name__ == "__main__":
    main()
