# =============================================================================
# DATA LOADING AND PREPROCESSING
# This cell handles the initial loading and preparation of the hurricane tweet data.
# Key steps include:
# 1. Importing necessary libraries for data manipulation, file paths, and time handling.
# 2. Constructing file paths to the GeoJSON data for Hurricanes Francine and Helene.
# 3. Loading the spatial data into GeoDataFrames.
# 4. Standardizing all timestamps to Coordinated Universal Time (UTC).
# 5. Aggregating the data into discrete 4-hour time bins for temporal analysis.
# 6. Creating various time-related columns (Unix timestamps, readable labels) for later use.
# =============================================================================

# Import core libraries
import geopandas as gpd  # Used for working with geospatial data.
import pandas as pd      # Used for data manipulation and analysis in DataFrames.
import os                # Provides a way of using operating system dependent functionality, like file paths.
from datetime import datetime, timezone # Used for handling date and time objects.
from fuzzywuzzy import fuzz, process
import re
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import rasterize
# Configuration
TARGET_CRS = 'EPSG:3857'  # Web Mercator
CELL_SIZE_M = 1000  # 5 km in meters
def preprocess_place_name(name):
    """Standardize place names for better matching"""
    if pd.isna(name) or name == 'NAN':
        return None

    name = str(name).upper().strip()

    # Common abbreviation standardizations
    name = re.sub(r'\bST\.?\b', 'SAINT', name)  # St. -> Saint
    name = re.sub(r'\bMT\.?\b', 'MOUNT', name)  # Mt. -> Mount
    name = re.sub(r'\bFT\.?\b', 'FORT', name)   # Ft. -> Fort
    name = re.sub(r'\bN\.?\b', 'NORTH', name)   # N. -> North
    name = re.sub(r'\bS\.?\b', 'SOUTH', name)   # S. -> South
    name = re.sub(r'\bE\.?\b', 'EAST', name)    # E. -> East
    name = re.sub(r'\bW\.?\b', 'WEST', name)    # W. -> West

    # Remove extra spaces and punctuation
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', ' ', name)     # Normalize spaces

    return name.strip()

def parse_gpe_entities(gpe_string):
    """Parse GPE string into multiple potential geographic entities"""
    if not gpe_string or pd.isna(gpe_string) or str(gpe_string).strip() == '':
        return []

    gpe_string = str(gpe_string).strip()

    # Split by common separators
    entities = []

    # Primary split by comma
    parts = [part.strip() for part in gpe_string.split(',')]

    for part in parts:
        if part:
            # Further split by other separators
            sub_parts = re.split(r'[;&|]', part)
            for sub_part in sub_parts:
                sub_part = sub_part.strip()
                if sub_part and len(sub_part) > 1:  # Ignore single characters
                    entities.append(preprocess_place_name(sub_part))

    # Remove None values and duplicates while preserving order
    clean_entities = []
    seen = set()
    for entity in entities:
        if entity and entity not in seen:
            clean_entities.append(entity)
            seen.add(entity)

    return clean_entities

def create_hierarchical_lookups(states_gdf, counties_gdf, cities_gdf):
    """Create hierarchical lookup dictionaries for fuzzy matching"""
    print("\nCreating hierarchical lookup dictionaries...")

    # 1. States - simple lookup with preprocessed names + abbreviations
    state_lookup = {}
    state_abbrev_to_name = {}  # Abbreviation to full name
    state_name_to_abbrev = {}  # Full name to abbreviation

    for idx, row in states_gdf.iterrows():
        state_name = preprocess_place_name(row['NAME'])
        if state_name:
            state_lookup[state_name] = row.geometry
            # Handle abbreviations if available
            if 'STUSPS' in row:
                abbrev = row['STUSPS'].upper()
                state_abbrev_to_name[abbrev] = state_name
                state_name_to_abbrev[state_name] = abbrev
                # Also add abbreviation as a lookup option
                state_lookup[abbrev] = row.geometry

    # 2. Counties - organized by state
    county_by_state = {}
    county_lookup = {}

    for idx, row in counties_gdf.iterrows():
        county_name = preprocess_place_name(row['NAME'])
        state_fips = row.get('STATEFP', '')

        if county_name:
            county_lookup[county_name] = row.geometry

            # Try to get state name from STATEFP or other fields
            state_name = None
            if 'STATE_NAME' in row:
                state_name = preprocess_place_name(row['STATE_NAME'])
            else:
                # Try to find state by FIPS code
                for s_idx, s_row in states_gdf.iterrows():
                    if s_row.get('STATEFP', '') == state_fips:
                        state_name = preprocess_place_name(s_row['NAME'])
                        break

            if state_name:
                if state_name not in county_by_state:
                    county_by_state[state_name] = {}
                county_by_state[state_name][county_name] = row.geometry

    # 3. Cities - organized by state
    city_by_state = {}
    city_lookup = {}

    for idx, row in cities_gdf.iterrows():
        city_name = preprocess_place_name(row['NAME'])
        state_abbrev = row.get('ST', '').upper()

        if city_name:
            city_lookup[city_name] = row.geometry

            # Convert state abbreviation to full name
            if state_abbrev in state_abbrev_to_name:
                state_full = state_abbrev_to_name[state_abbrev]
                if state_full not in city_by_state:
                    city_by_state[state_full] = {}
                city_by_state[state_full][city_name] = row.geometry
    #


    return {
        'state_lookup': state_lookup,
        'county_lookup': county_lookup,
        'city_lookup': city_lookup,
        'county_by_state': county_by_state,
        'city_by_state': city_by_state,
        'state_abbrev_to_name': state_abbrev_to_name,
        'state_name_to_abbrev': state_name_to_abbrev
    }

def fuzzy_match_entity(entity, candidates, threshold=75):
    """Fuzzy match an entity against candidates"""
    if not entity or not candidates:
        return None, 0

    # Try exact match first
    if entity in candidates:
        return entity, 100

    # Use fuzzy matching
    match = process.extractOne(entity, candidates.keys(), scorer=fuzz.ratio)

    if match and match[1] >= threshold:
        return match[0], match[1]

    return None, 0

def find_all_geographic_matches(entities, lookups):
    """Find ALL geographic matches (state, county, city) for the entities"""
    if not entities:
        return []

    state_lookup = lookups['state_lookup']
    county_lookup = lookups['county_lookup']
    city_lookup = lookups['city_lookup']
    county_by_state = lookups['county_by_state']
    city_by_state = lookups['city_by_state']

    # Store all successful matches
    all_matches = []

    # Context tracking for better matching
    found_states = set()

    # STEP 1: Find all state matches first
    for entity in entities:
        state_match, state_score = fuzzy_match_entity(entity, state_lookup, threshold=75)
        if state_match:
            all_matches.append(('STATE', state_match, state_lookup[state_match], state_score))
            found_states.add(state_match)

    # STEP 2: Find county matches (global first, then state-specific)
    for entity in entities:
        # Global county search
        county_match, county_score = fuzzy_match_entity(entity, county_lookup, threshold=75)
        if county_match:
            all_matches.append(('COUNTY', county_match, county_lookup[county_match], county_score))

        # State-specific county search (higher accuracy)
        for state_name in found_states:
            if state_name in county_by_state:
                state_counties = county_by_state[state_name]
                state_county_match, state_county_score = fuzzy_match_entity(entity, state_counties, threshold=70)
                if state_county_match and state_county_score > county_score:
                    # Replace with better state-specific match
                    # Remove the global match if it exists
                    all_matches = [m for m in all_matches if not (m[0] == 'COUNTY' and m[1] == county_match)]
                    all_matches.append(('COUNTY', state_county_match, state_counties[state_county_match], state_county_score))

    # STEP 3: Find city matches (global first, then state-specific)
    for entity in entities:
        # Global city search
        city_match, city_score = fuzzy_match_entity(entity, city_lookup, threshold=75)
        if city_match:
            all_matches.append(('CITY', city_match, city_lookup[city_match], city_score))

        # State-specific city search (higher accuracy)
        for state_name in found_states:
            if state_name in city_by_state:
                state_cities = city_by_state[state_name]
                state_city_match, state_city_score = fuzzy_match_entity(entity, state_cities, threshold=70)
                if state_city_match and state_city_score > city_score:
                    # Replace with better state-specific match
                    # Remove the global match if it exists
                    all_matches = [m for m in all_matches if not (m[0] == 'CITY' and m[1] == city_match)]
                    all_matches.append(('CITY', state_city_match, state_cities[state_city_match], state_city_score))

    # Remove duplicates (same scale + name)
    unique_matches = []
    seen_combinations = set()
    for match in all_matches:
        combo = (match[0], match[1])  # (scale, name)
        if combo not in seen_combinations:
            unique_matches.append(match)
            seen_combinations.add(combo)

    return unique_matches

def multi_level_assign_scale_levels(row, lookups):
    """
    Return ALL geographic scale levels that match this tweet
    Returns a list of matches: [(scale, name, geom, score), ...]
    """
    gpe = str(row.get('GPE', '')).strip()
    fac = str(row.get('FAC', '')).strip()

    matches = []

    # Parse GPE into multiple entities
    entities = parse_gpe_entities(gpe)

    if entities:
        # Find all geographic matches
        geo_matches = find_all_geographic_matches(entities, lookups)
        matches.extend(geo_matches)

    # Add facility as separate match if available
    if fac and fac not in ['nan', 'NAN', '']:
        matches.append(('FACILITY', fac, row.geometry, 100))

    # If no matches found, return unmatched
    if not matches:
        matches.append(('UNMATCHED', None, row.geometry, 0))

    return matches

def expand_tweets_by_matches(gdf, lookups, dataset_name):
    """
    Expand the GeoDataFrame so each tweet creates multiple rows (one per geographic match)
    """
    print(f"\nExpanding {dataset_name} tweets by geographic matches...")

    expanded_rows = []

    for idx, row in gdf.iterrows():
        if idx % 100 == 0:
            print(idx)
        matches = multi_level_assign_scale_levels(row, lookups)

        # Create one row per match
        for scale, name, geom, score in matches:
            new_row = row.copy()
            new_row['scale_level'] = scale
            new_row['matched_name'] = name
            new_row['matched_geom'] = geom
            new_row['match_score'] = score
            new_row['original_index'] = idx  # Track original tweet
            expanded_rows.append(new_row)

    # Create new GeoDataFrame and preserve the original CRS
    expanded_gdf = gpd.GeoDataFrame(expanded_rows, crs=gdf.crs)

    # Show some examples of multi-level matches
    print(f"  Sample multi-level matches:")
    # Group by original tweet and show ones with multiple matches
    multi_matches = expanded_gdf.groupby('original_index').size()
    multi_match_indices = multi_matches[multi_matches > 1].head(5).index

    for orig_idx in multi_match_indices:
        tweet_matches = expanded_gdf[expanded_gdf['original_index'] == orig_idx]
        original_gpe = tweet_matches.iloc[0]['GPE']
        match_summary = ', '.join([f"{row['scale_level']}:{row['matched_name']}" for _, row in tweet_matches.iterrows()])
        # print(f"    '{original_gpe}' → {match_summary}")

    return expanded_gdf

# =============================================================================
# EXECUTE MULTI-LEVEL FUZZY MATCHING
# =============================================================================
# --- 1. Load GeoJSON files ---
# Get the parent directory of the current working directory to build relative paths.
# This makes the script more portable as it doesn't rely on a hardcoded absolute path.

local_path = os.path.dirname(os.getcwd())
states_dir = r"\data\shape_files\cb_2023_us_state_20m.shp"
counties_dir = r"\data\shape_files\cb_2023_us_county_20m.shp"
cities_dir = r"\data\shape_files\US_Cities.shp"
states_path = f"{local_path}{states_dir}"
counties_path = f"{local_path}{counties_dir}"
cities_path = f"{local_path}{cities_dir}"


# Load spatial reference data
states_gdf = gpd.read_file(states_path)
counties_gdf = gpd.read_file(counties_path)
cities_gdf = gpd.read_file(cities_path)
# Define the relative paths to the GeoJSON files for each hurricane.
francine_dir = r"\data\geojson\francine.geojson"
helene_dir = r"\data\geojson\helene.geojson"

# Combine the base path and relative directory to create full, absolute paths to the files.
francine_path = f"{local_path}{francine_dir}"
helene_path = f"{local_path}{helene_dir}"

# --- 2. Load data into GeoDataFrames ---
# A GeoDataFrame is a pandas DataFrame with a special 'geometry' column that allows for spatial operations.
francine_gdf = gpd.read_file(francine_path)
helene_gdf = gpd.read_file(helene_path)

# --- 3. Standardize timestamps to UTC ---
# Convert the original 'time' column into a pandas datetime object.
# Setting `utc=True` ensures all timestamps are in a single, unambiguous timezone (UTC).
# This is crucial for accurate temporal comparisons and binning.
francine_gdf['timestamp'] = pd.to_datetime(francine_gdf['time'], utc=True)
helene_gdf['timestamp'] = pd.to_datetime(helene_gdf['time'], utc=True)

# --- 4. Group data into 4-hour time bins ---
# The `dt.floor('4h')` function rounds each timestamp *down* to the nearest 4-hour interval.
# For example, 09:35 becomes 08:00, 15:59 becomes 12:00. This aggregates tweets into discrete time windows.
francine_gdf['time_bin'] = francine_gdf['timestamp'].dt.floor('4h')
helene_gdf['time_bin'] = helene_gdf['timestamp'].dt.floor('4h')

# --- 5. Create Unix timestamps and lookup dictionaries ---
# Convert the binned datetime objects into Unix timestamps (as an integer).
# The `// 1000` division is likely to convert from nanoseconds or microseconds to seconds, a more standard Unix format.
francine_gdf['unix_timestamp'] = francine_gdf['time_bin'].astype('int64') // 1000
helene_gdf['unix_timestamp'] = helene_gdf['time_bin'].astype('int64') // 1000

# Create dictionaries to map the numeric Unix timestamp back to its original datetime object.
# This provides a quick way to retrieve the readable time bin later in the script without recalculating it.
helene_timestamp_dict = dict(zip(helene_gdf['unix_timestamp'], helene_gdf['time_bin']))
francine_timestamp_dict = dict(zip(francine_gdf['unix_timestamp'], francine_gdf['time_bin']))

# --- 6. Create readable labels for file naming ---
# The `dt.strftime` function formats the datetime object into a specific string format.
# Here, '%Y%m%d_%H%M' creates a clean, sortable label like '20240926_0800', which is ideal for filenames.
francine_gdf['bin_label'] = francine_gdf['time_bin'].dt.strftime('%Y%m%d_%H%M')
helene_gdf['bin_label'] = helene_gdf['time_bin'].dt.strftime('%Y%m%d_%H%M')

# Create hierarchical lookups
lookups = create_hierarchical_lookups(states_gdf, counties_gdf, cities_gdf)

# Apply to both datasets (this will expand the datasets)
francine_gdf = expand_tweets_by_matches(francine_gdf, lookups, "FRANCINE")
helene_gdf = expand_tweets_by_matches(helene_gdf, lookups, "HELENE")

 # Group tweets by 4-hour intervals and scale level
# Using unix_timestamp for unambiguous temporal grouping

# Alternative approach:
francine_interval_counts = francine_gdf.groupby(['unix_timestamp', 'scale_level', 'matched_name']).agg({
    'matched_geom': 'first'
}).reset_index()

# Add count column separately
count_series = francine_gdf.groupby(['unix_timestamp', 'scale_level', 'matched_name']).size()
francine_interval_counts['count'] = count_series.values

# Same for Helene
helene_interval_counts = helene_gdf.groupby(['unix_timestamp', 'scale_level', 'matched_name']).agg({
    'matched_geom': 'first'
}).reset_index()
count_series = helene_gdf.groupby(['unix_timestamp', 'scale_level', 'matched_name']).size()
helene_interval_counts['count'] = count_series.values

# Sort by timestamp to ensure chronological order
francine_interval_counts = francine_interval_counts.sort_values('unix_timestamp')
helene_interval_counts = helene_interval_counts.sort_values('unix_timestamp')

# Calculate cumulative counts
francine_interval_counts['cumulative_count'] = francine_interval_counts.groupby(['scale_level', 'matched_name'])['count'].cumsum()
helene_interval_counts['cumulative_count'] = helene_interval_counts.groupby(['scale_level', 'matched_name'])['count'].cumsum()

# Get unique time bins for iteration
francine_time_bins = sorted(francine_gdf['unix_timestamp'].unique())
helene_time_bins = sorted(helene_gdf['unix_timestamp'].unique())

francine_proj = francine_gdf.to_crs(TARGET_CRS)
helene_proj = helene_gdf.to_crs(TARGET_CRS)

states_proj = states_gdf.to_crs(TARGET_CRS)
counties_proj = counties_gdf.to_crs(TARGET_CRS)
cities_proj = cities_gdf.to_crs(TARGET_CRS)

francine_bounds = francine_proj.total_bounds
helene_bounds = helene_proj.total_bounds

# Get union of both bounding boxes
minx = min(francine_bounds[0], helene_bounds[0])
miny = min(francine_bounds[1], helene_bounds[1])
maxx = max(francine_bounds[2], helene_bounds[2])
maxy = max(francine_bounds[3], helene_bounds[3])


# Calculate grid dimensions
width = int(np.ceil((maxx - minx) / CELL_SIZE_M))
height = int(np.ceil((maxy - miny) / CELL_SIZE_M))


# Create master transform
master_transform = from_bounds(minx, miny, maxx, maxy, width, height)


# Calculate actual coverage area
area_km2 = (width * height * CELL_SIZE_M * CELL_SIZE_M) / 1_000_000


# Store grid parameters for later use
grid_params = {
    'crs': TARGET_CRS,
    'cell_size': CELL_SIZE_M,
    'width': width,
    'height': height,
    'bounds': (minx, miny, maxx, maxy),
    'transform': master_transform
}

state_lookup_proj = dict(zip(states_proj['NAME'].str.upper(), states_proj.geometry))
county_lookup_proj = dict(zip(counties_proj['NAME'].str.upper(), counties_proj.geometry))
cities_lookup_proj = dict(zip(cities_proj['NAME'].str.upper(), cities_proj.geometry))

rasterize.rasterize_process('francine', francine_proj, francine_interval_counts, francine_time_bins, francine_timestamp_dict, local_path, grid_params)