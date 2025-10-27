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

# --- 1. Load GeoJSON files ---
# Get the parent directory of the current working directory to build relative paths.
# This makes the script more portable as it doesn't rely on a hardcoded absolute path.
local_path = os.path.dirname(os.getcwd())

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

# Load reference shapefiles
from fuzzywuzzy import fuzz, process
import re

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

# PLACE THIS CODE AFTER LOADING SHAPEFILES BUT BEFORE CREATING SIMPLE LOOKUPS
# =============================================================================
# MULTI-LEVEL GEOGRAPHIC MATCHING SYSTEM (ALL LEVELS)
# =============================================================================

from fuzzywuzzy import fuzz, process
import re

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

print("\n" + "="*60)
print("MULTI-LEVEL GEOGRAPHIC MATCHING (ALL LEVELS)")
print("="*60)

# Create hierarchical lookups
lookups = create_hierarchical_lookups(states_gdf, counties_gdf, cities_gdf)

# Apply to both datasets (this will expand the datasets)
francine_gdf = expand_tweets_by_matches(francine_gdf, lookups, "FRANCINE")
helene_gdf = expand_tweets_by_matches(helene_gdf, lookups, "HELENE")

print("\n" + "="*60)
print("MULTI-LEVEL FUZZY MATCHING COMPLETE ✓")
print("="*60)
print("\nNote: Datasets are now expanded - each original tweet may have multiple rows")
print("representing different geographic scales (STATE, COUNTY, CITY, etc.)")


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

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# ==============================================================================
# STEP 1: DEFINE MASTER GRID CANVAS
# ==============================================================================

# Configuration
TARGET_CRS = 'EPSG:3857'  # Web Mercator
CELL_SIZE_M = 1000  # 5 km in meters

print("=" * 60)
print("STEP 1: CREATING MASTER GRID CANVAS")
print("=" * 60)

# Project both datasets to target CRS
print(f"\nProjecting datasets to {TARGET_CRS}...")
francine_proj = francine_gdf.to_crs(TARGET_CRS)
helene_proj = helene_gdf.to_crs(TARGET_CRS)

# Also project reference geometries
print("Projecting reference geometries...")
states_proj = states_gdf.to_crs(TARGET_CRS)
counties_proj = counties_gdf.to_crs(TARGET_CRS)
cities_proj = cities_gdf.to_crs(TARGET_CRS)
# Calculate combined extent from both hurricanes"
print("\nCalculating master extent...")
francine_bounds = francine_proj.total_bounds
helene_bounds = helene_proj.total_bounds

# Get union of both bounding boxes
minx = min(francine_bounds[0], helene_bounds[0])
miny = min(francine_bounds[1], helene_bounds[1])
maxx = max(francine_bounds[2], helene_bounds[2])
maxy = max(francine_bounds[3], helene_bounds[3])
#
# print(f"  Master bounds (EPSG:3857):")
# print(f"    minx: {minx:,.2f}")
# print(f"    miny: {miny:,.2f}")
# print(f"    maxx: {maxx:,.2f}")
# print(f"    maxy: {maxy:,.2f}")

# Calculate grid dimensions
width = int(np.ceil((maxx - minx) / CELL_SIZE_M))
height = int(np.ceil((maxy - miny) / CELL_SIZE_M))

print(f"\nGrid Configuration:")
print(f"  Cell size: {CELL_SIZE_M:,} meters ({CELL_SIZE_M/1000} km)")
print(f"  Grid dimensions: {width} x {height} cells")
print(f"  Total cells: {width * height:,}")

# Create master transform
master_transform = from_bounds(minx, miny, maxx, maxy, width, height)

print(f"\nMaster Transform:")
print(f"  {master_transform}")

# Calculate actual coverage area
area_km2 = (width * height * CELL_SIZE_M * CELL_SIZE_M) / 1_000_000
print(f"\nCoverage area: {area_km2:,.2f} km²")

# Store grid parameters for later use
grid_params = {
    'crs': TARGET_CRS,
    'cell_size': CELL_SIZE_M,
    'width': width,
    'height': height,
    'bounds': (minx, miny, maxx, maxy),
    'transform': master_transform
}

print(f"\n{'=' * 60}")
print("MASTER GRID CANVAS READY ✓")
print(f"{'=' * 60}")

# Update lookup dictionaries with projected geometries
print("\nUpdating geometry lookups with projected coordinates...")
state_lookup_proj = dict(zip(states_proj['NAME'].str.upper(), states_proj.geometry))
county_lookup_proj = dict(zip(counties_proj['NAME'].str.upper(), counties_proj.geometry))
cities_lookup_proj = dict(zip(cities_proj['NAME'].str.upper(), cities_proj.geometry))
# validation_results = validate_city_matching(francine_gdf, helene_gdf, lookups['city_lookup'], lookups['state_lookup'], lookups['county_lookup'])
print("Lookup dictionaries updated with projected geometries ✓")

#%%
# !pip install fuzzywuzzy python-Levenshtein geopandas pandas numpy matplotlib
#%%
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

# --- 1. Load GeoJSON files ---
# Get the parent directory of the current working directory to build relative paths.
# This makes the script more portable as it doesn't rely on a hardcoded absolute path.
local_path = os.path.dirname(os.getcwd())

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
#%%
# Load reference shapefiles
from fuzzywuzzy import fuzz, process
import re

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

# PLACE THIS CODE AFTER LOADING SHAPEFILES BUT BEFORE CREATING SIMPLE LOOKUPS
# =============================================================================
# MULTI-LEVEL GEOGRAPHIC MATCHING SYSTEM (ALL LEVELS)
# =============================================================================

from fuzzywuzzy import fuzz, process
import re

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

print("\n" + "="*60)
print("MULTI-LEVEL GEOGRAPHIC MATCHING (ALL LEVELS)")
print("="*60)

# Create hierarchical lookups
lookups = create_hierarchical_lookups(states_gdf, counties_gdf, cities_gdf)

# Apply to both datasets (this will expand the datasets)
francine_gdf = expand_tweets_by_matches(francine_gdf, lookups, "FRANCINE")
helene_gdf = expand_tweets_by_matches(helene_gdf, lookups, "HELENE")

print("\n" + "="*60)
print("MULTI-LEVEL FUZZY MATCHING COMPLETE ✓")
print("="*60)
print("\nNote: Datasets are now expanded - each original tweet may have multiple rows")
print("representing different geographic scales (STATE, COUNTY, CITY, etc.)")
#%%
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


#%%
import numpy as np
import rasterio
from rasterio.transform import from_bounds

# ==============================================================================
# STEP 1: DEFINE MASTER GRID CANVAS
# ==============================================================================

# Configuration
TARGET_CRS = 'EPSG:3857'  # Web Mercator
CELL_SIZE_M = 1000  # 5 km in meters

print("=" * 60)
print("STEP 1: CREATING MASTER GRID CANVAS")
print("=" * 60)

# Project both datasets to target CRS
print(f"\nProjecting datasets to {TARGET_CRS}...")
francine_proj = francine_gdf.to_crs(TARGET_CRS)
helene_proj = helene_gdf.to_crs(TARGET_CRS)

# Also project reference geometries
print("Projecting reference geometries...")
states_proj = states_gdf.to_crs(TARGET_CRS)
counties_proj = counties_gdf.to_crs(TARGET_CRS)
cities_proj = cities_gdf.to_crs(TARGET_CRS)
# Calculate combined extent from both hurricanes"
print("\nCalculating master extent...")
francine_bounds = francine_proj.total_bounds
helene_bounds = helene_proj.total_bounds

# Get union of both bounding boxes
minx = min(francine_bounds[0], helene_bounds[0])
miny = min(francine_bounds[1], helene_bounds[1])
maxx = max(francine_bounds[2], helene_bounds[2])
maxy = max(francine_bounds[3], helene_bounds[3])
#
# print(f"  Master bounds (EPSG:3857):")
# print(f"    minx: {minx:,.2f}")
# print(f"    miny: {miny:,.2f}")
# print(f"    maxx: {maxx:,.2f}")
# print(f"    maxy: {maxy:,.2f}")

# Calculate grid dimensions
width = int(np.ceil((maxx - minx) / CELL_SIZE_M))
height = int(np.ceil((maxy - miny) / CELL_SIZE_M))

print(f"\nGrid Configuration:")
print(f"  Cell size: {CELL_SIZE_M:,} meters ({CELL_SIZE_M/1000} km)")
print(f"  Grid dimensions: {width} x {height} cells")
print(f"  Total cells: {width * height:,}")

# Create master transform
master_transform = from_bounds(minx, miny, maxx, maxy, width, height)

print(f"\nMaster Transform:")
print(f"  {master_transform}")

# Calculate actual coverage area
area_km2 = (width * height * CELL_SIZE_M * CELL_SIZE_M) / 1_000_000
print(f"\nCoverage area: {area_km2:,.2f} km²")

# Store grid parameters for later use
grid_params = {
    'crs': TARGET_CRS,
    'cell_size': CELL_SIZE_M,
    'width': width,
    'height': height,
    'bounds': (minx, miny, maxx, maxy),
    'transform': master_transform
}

print(f"\n{'=' * 60}")
print("MASTER GRID CANVAS READY ✓")
print(f"{'=' * 60}")

# Update lookup dictionaries with projected geometries
print("\nUpdating geometry lookups with projected coordinates...")
state_lookup_proj = dict(zip(states_proj['NAME'].str.upper(), states_proj.geometry))
county_lookup_proj = dict(zip(counties_proj['NAME'].str.upper(), counties_proj.geometry))
cities_lookup_proj = dict(zip(cities_proj['NAME'].str.upper(), cities_proj.geometry))
# validation_results = validate_city_matching(francine_gdf, helene_gdf, lookups['city_lookup'], lookups['state_lookup'], lookups['county_lookup'])
print("Lookup dictionaries updated with projected geometries ✓")
#%%
import os
from scipy.ndimage import gaussian_filter
from rasterio.features import rasterize
from rasterio.features import geometry_mask
# ==============================================================================
# STEP 2: MAIN RASTERIZATION LOOP - TIME ITERATION
# ==============================================================================

# Create output directories
rasters_dir = r"\rasters_output"
output_dir = f"{local_path}{rasters_dir}"
# output_dir = os.path.join(local_path, 'rasters_output')
# output_dir = r"C:\Users\colto\Documents\GitHub\Tweet_project\rasters_output"
os.makedirs(output_dir, exist_ok=True)


def create_hierarchical_rasters(data, grid_params, time_bin):
    """Create hierarchically weighted rasters with automatic parent state inclusion"""
    print(f"    Creating hierarchical raster for time {time_bin}...")

    output_grid = np.zeros((grid_params['height'], grid_params['width']), dtype=np.float32)
    states_to_include = set()  # Track which states need base layers

    # 1. First pass: identify all states that need base layers
    state_data = data[data['scale_level'] == 'STATE']
    if len(state_data) > 0:
        states_to_include.update(state_data['matched_name'].unique())

    # Check counties - add their parent states
    county_data = data[data['scale_level'] == 'COUNTY']
    for county_name in county_data['matched_name'].unique():
        if county_name in county_lookup_proj:
            # Find parent state by spatial containment
            county_geom = county_lookup_proj[county_name]
            for state_name, state_geom in state_lookup_proj.items():
                if state_geom.contains(county_geom.centroid):
                    states_to_include.add(state_name)
                    break

    # Check cities - add their parent states
    city_data = data[data['scale_level'] == 'CITY']
    for city_name in city_data['matched_name'].unique():
        if city_name in cities_lookup_proj:
            city_geom = cities_lookup_proj[city_name]
            for state_name, state_geom in state_lookup_proj.items():
                if state_geom.contains(city_geom.centroid):
                    states_to_include.add(state_name)
                    break

    # 2. Rasterize all states that need inclusion
    for state_name in states_to_include:
        if state_name in state_lookup_proj:
            state_geom = state_lookup_proj[state_name]
            mask = rasterize(
                [(state_geom, 1)],
                out_shape=(grid_params['height'], grid_params['width']),
                transform=grid_params['transform'],
                fill=0, dtype=np.float32, all_touched=True
            )

            # Get tweet count if state was mentioned, else use minimal base
            if state_name in state_data['matched_name'].values:
                tweet_count = state_data[state_data['matched_name'] == state_name]['count'].sum()
            else:
                tweet_count = 1  # Minimal base for implied states

            base_value = tweet_count * 2
            output_grid += mask * base_value

    # 3. Add counties (same as before)
    if len(county_data) > 0:
        county_counts = county_data.groupby('matched_name')['count'].sum()
        for county_name, tweet_count in county_counts.items():
            if county_name in county_lookup_proj:
                mask = rasterize(
                    [(county_lookup_proj[county_name], 1)],
                    out_shape=(grid_params['height'], grid_params['width']),
                    transform=grid_params['transform'],
                    fill=0, dtype=np.float32, all_touched=True
                )
                output_grid += mask * tweet_count * 5

    # 4. Add cities (same as before)
    if len(city_data) > 0:
        city_counts = city_data.groupby('matched_name')['count'].sum()
        for city_name, tweet_count in city_counts.items():
            if city_name in cities_lookup_proj:
                mask = rasterize(
                    [(cities_lookup_proj[city_name], 1)],
                    out_shape=(grid_params['height'], grid_params['width']),
                    transform=grid_params['transform'],
                    fill=0, dtype=np.float32, all_touched=True
                )
                output_grid += mask * tweet_count * 10

    # 5. Add facilities
    facility_data = data[data['scale_level'] == 'FACILITY']
    if len(facility_data) > 0:
        output_grid += create_facility_raster(data, grid_params)

    return output_grid

def process_hurricane(hurricane_name, gdf_proj, interval_counts, time_bins, timestamp_dict):
    """
    Process a single hurricane through all time bins
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {hurricane_name.upper()}")
    print(f"{'=' * 60}")
    print()
    print(gdf_proj)
    # Create hurricane-specific output directory
    hurricane_dir = os.path.join(output_dir, hurricane_name.lower())
    os.makedirs(hurricane_dir, exist_ok=True)

    # Initialize cumulative grid (persists across time bins)
    cumulative_grid = np.zeros((grid_params['height'], grid_params['width']), dtype=np.float32)

    # Loop through each time bin chronologically
    for idx, time_bin in enumerate(time_bins):
        # print(f"\n--- Time Bin {idx+1}/{len(time_bins)}: {time_bin} ---")

        # Filter data for current time bin
        current_data = interval_counts[interval_counts['unix_timestamp'] == time_bin]
        tweet_count = len(current_data)
        # print(f"  Tweets in this bin: {tweet_count}")

        # WITH THIS:
        incremental_grid = create_hierarchical_rasters(current_data, grid_params, time_bin)

        # === END PLACEHOLDERS ===

        # Update cumulative grid
        cumulative_grid += incremental_grid
        # Save rasters
        save_raster(incremental_grid, hurricane_dir, hurricane_name, time_bin, 'increment', timestamp_dict)
        save_raster(cumulative_grid, hurricane_dir, hurricane_name, time_bin, 'cumulative', timestamp_dict)

        print(f"  Incremental max value: {np.max(incremental_grid):.2f}")
        print(f"  Cumulative max value: {np.max(cumulative_grid):.2f}")

    print(f"\n{hurricane_name.upper()} processing complete!")
    print(f"Output saved to: {hurricane_dir}")
    return

# ==============================================================================
# PLACEHOLDER FUNCTIONS (TO BE IMPLEMENTED)
# ==============================================================================

def create_facility_raster(data, grid_params):
    """Create KDE raster for facility points with strong hotspot multiplier"""
    print("    [FACILITY] Creating facility raster...")

    # Initialize empty raster
    facility_grid = np.zeros((grid_params['height'], grid_params['width']), dtype=np.float32)

    # Filter for FACILITY-level tweets only
    facility_data = data[data['scale_level'] == 'FACILITY']

    if len(facility_data) == 0:
        print("      No facility-level tweets in this time bin")
        return facility_grid

    # Group by facility coordinates (using matched_name as proxy) and sum counts
    facility_counts = facility_data.groupby('matched_name')['count'].sum()

    print(f"      Processing {len(facility_counts)} unique facilities")

    # HOTSPOT PARAMETERS for facilities
    sigma_meters = 2 * grid_params['cell_size']  # 10 km for 5km cells
    sigma_pixels = sigma_meters / grid_params['cell_size']  # Convert to pixel units
    facility_multiplier = 10  # Make facilities 10x more prominent (strongest hotspots)

    # Process each facility
    facilities_processed = 0
    for facility_name, tweet_count in facility_counts.items():
        # Get facility data to extract geometry
        facility_rows = facility_data[facility_data['matched_name'] == facility_name]

        if len(facility_rows) > 0:
            # Get the point geometry (should be from the tweet's geocoded location)
            facility_point = facility_rows.iloc[0]['matched_geom']

            # Project point to grid CRS if needed
            if hasattr(facility_point, 'x') and hasattr(facility_point, 'y'):
                # Create GeoSeries to handle projection
                point_geoseries = gpd.GeoSeries([facility_point], crs='EPSG:4326')
                point_proj = point_geoseries.to_crs(grid_params['crs']).iloc[0]

                # Convert point coordinates to pixel indices
                px = (point_proj.x - grid_params['bounds'][0]) / grid_params['cell_size']
                py = (grid_params['bounds'][3] - point_proj.y) / grid_params['cell_size']

                # Check if point is within grid bounds
                if 0 <= px < grid_params['width'] and 0 <= py < grid_params['height']:
                    # Create point raster with tweet count at location
                    point_grid = np.zeros((grid_params['height'], grid_params['width']), dtype=np.float32)
                    point_grid[int(py), int(px)] = tweet_count

                    # Apply Gaussian filter to create kernel density
                    kernel_grid = gaussian_filter(point_grid, sigma=sigma_pixels, mode='constant', cval=0)

                    # FIXED: Only add once with proper multiplier
                    facility_grid += kernel_grid * facility_multiplier

                    facilities_processed += 1
                    effective_value = tweet_count * facility_multiplier
                else:
                    print(f"      WARNING: Facility '{facility_name}' outside grid bounds")
            else:
                print(f"      WARNING: Invalid geometry for facility '{facility_name}'")

    print(f"      Processed {facilities_processed} facilities with sigma={sigma_pixels:.2f} pixels")

    total_value = np.sum(facility_grid)
    max_value = np.max(facility_grid)
    # print(f"      Total facility grid value: {total_value:.2f}, Max pixel: {max_value:.2f}")

    return facility_grid

def save_raster(grid, output_dir, hurricane_name, time_bin, raster_type, timestamp_dict):
    """Save raster as GeoTIFF in type-specific subdirectory"""
    # Create subdirectory for raster type
    type_dir = os.path.join(output_dir, raster_type)
    os.makedirs(type_dir, exist_ok=True)
    print('max grid', np.max(grid))
    # Convert unix timestamp (microseconds) back to datetime
    time_str = timestamp_dict[time_bin].strftime('%Y%m%d_%H%M%S')
    # time_str = pd.Timestamp(time_bin, unit='us').strftime('%Y%m%d_%H%M%S')
    print([time_str])
    filename = f"{hurricane_name}_tweets_{time_str}.tif"
    filepath = os.path.join(type_dir, filename)
    print(grid_params)
    with rasterio.open(
        filepath, 'w',
        driver='GTiff',
        height=grid_params['height'],
        width=grid_params['width'],
        count=1,
        dtype=grid.dtype,
        crs=grid_params['crs'],
        transform=grid_params['transform'],
        compress='lzw'
    ) as dst:
        dst.write(grid, 1)

    print(f"    Saved: {raster_type}/{filename}")

# ==============================================================================
# EXECUTE PROCESSING FOR BOTH HURRICANES
# ==============================================================================

print("\n" + "=" * 60)
print("STARTING RASTERIZATION PROCESS")
print("=" * 60)

# Process Francine
process_hurricane('francine', francine_proj, francine_interval_counts, francine_time_bins, francine_timestamp_dict)

# Process Helene
process_hurricane('helene', helene_proj, helene_interval_counts, helene_time_bins, helene_timestamp_dict)

print("\n" + "=" * 60)
print("ALL PROCESSING COMPLETE! ✓")
print("=" * 60)
#%%
import arcpy
import os
from datetime import datetime


# Note this is to be inserted into the python command window
# Paths
gdb_path = r"C:\Users\colto\Documents\GitHub\Tweet_project\Tweet_project.gdb"


# raster_folder = r"C:\Users\colto\Documents\GitHub\Tweet_project\rasters_output\helene\cumulative"
# mosaic_name = "helene_cumulative_mosaic_v2"
#
# raster_folder = r"C:\Users\colto\Documents\GitHub\Tweet_project\rasters_output\helene\increment"
# mosaic_name = "helene_increment_mosaic_v2"
#
# raster_folder = r"C:\Users\colto\Documents\GitHub\Tweet_project\rasters_output\francine\cumulative"
# mosaic_name = "francine_cumulative_mosaic_v2"

raster_folder = r"C:\Users\colto\Documents\GitHub\Tweet_project\rasters_output\francine\increment"
mosaic_name = "francine_increment_mosaic_v2"



# Create geodatabase if it doesn't exist
if not arcpy.Exists(gdb_path):
    arcpy.CreateFileGDB_management(os.path.dirname(gdb_path), os.path.basename(gdb_path))

# Create mosaic dataset
mosaic_path = os.path.join(gdb_path, mosaic_name)
if arcpy.Exists(mosaic_path):
    arcpy.Delete_management(mosaic_path)

arcpy.CreateMosaicDataset_management(gdb_path, mosaic_name, "PROJCS['WGS_1984_Web_Mercator_Auxiliary_Sphere',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Mercator_Auxiliary_Sphere'],PARAMETER['False_Easting',0.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',0.0],PARAMETER['Standard_Parallel_1',0.0],PARAMETER['Auxiliary_Sphere_Type',0.0],UNIT['Meter',1.0]]")

print(f"Created mosaic dataset: {mosaic_path}")

# Add rasters to mosaic
arcpy.AddRastersToMosaicDataset_management(
    mosaic_path,
    "Raster Dataset",
    raster_folder,
    filter="*.tif"
)

print("Added rasters to mosaic dataset")

# Add time field
arcpy.AddField_management(mosaic_path, "date", "DATE")

# Calculate time from filename
with arcpy.da.UpdateCursor(mosaic_path, ["Name", "date"]) as cursor:
    for row in cursor:
        filename = row[0]
        # Remove .tif extension and split
        parts = filename.replace(".tif", "").split("_")

        # Join last two parts to get full timestamp: 20240926 + 080000
        time_str = parts[-2] + parts[-1]  # Combines date and time

        # Parse: 20240926080000 -> datetime
        dt = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        print(f"{filename} -> {time_str} -> {dt}")
        row[1] = dt
        cursor.updateRow(row)

print("Time field populated")

# Configure mosaic properties
arcpy.SetMosaicDatasetProperties_management(
    mosaic_path,
    start_time_field="date"
)

print("Mosaic dataset configured with time dimension")

print(f"\nMosaic dataset complete: {mosaic_path}")
print("To apply symbology in ArcGIS Pro:")
print(f"1. Add mosaic to map: {mosaic_path}")
print(f"2. Right-click layer > Symbology > Import")
print(f"3. Select: {symbology_file}")
print("4. Enable time slider to animate cumulative growth")