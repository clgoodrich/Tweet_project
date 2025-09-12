import geopandas as gpd
import pandas as pd
import os
pd.set_option('display.max_columns', None)


def get_project_root():
    """Gets the absolute path to the project's root folder."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_file_path(*path_segments):
    """Builds a full path to any data file from the project root."""
    project_root = get_project_root()
    return os.path.join(project_root, *path_segments)


def get_geojson(label):
    """Get path to helene.geojson"""
    geojson = get_data_file_path('data', 'geojson', f'{label}.geojson')
    return gpd.read_file(geojson)


def get_cities():
    """
    Load US cities data, starting with GeoJSON and supplementing with CSV data.
    Combines both sources to maximize data coverage.
    """
    cities_gdf = None

    # Try to load GeoJSON first
    try:
        geojson_path = get_data_file_path('data', 'geojson', f'us_cities.geojson')
        cities_gdf = gpd.read_file(geojson_path)

        if cities_gdf.crs is None:
            cities_gdf = cities_gdf.set_crs("EPSG:4326")

        print(f"✓ Loaded {len(cities_gdf)} cities from GeoJSON")

    except (FileNotFoundError, Exception) as e:
        print(f"⚠ GeoJSON loading failed ({e})")

    # Load CSV data
    try:
        df_path = get_data_file_path('data', 'tables', 'cities1000.csv')
        df = pd.read_csv(df_path)
        csv_cities_df = df[
            (df['country_code'] == 'US') &
            (df['feature_class'] == 'P') &
            (df['population'].notna()) &
            (df['latitude'].notna()) &
            (df['longitude'].notna())
            ].reset_index(drop=True)

        csv_cities_gdf = gpd.GeoDataFrame(
            csv_cities_df,
            geometry=gpd.points_from_xy(csv_cities_df.longitude, csv_cities_df.latitude),
            crs="EPSG:4326"
        )

        print(f"✓ Loaded {len(csv_cities_gdf)} cities from CSV")

        # If we have both, supplement GeoJSON with missing CSV cities
        if cities_gdf is not None:
            # Find missing cities (by geonameid if available, otherwise by name)
            if 'geonameid' in cities_gdf.columns and 'geonameid' in csv_cities_gdf.columns:
                existing_ids = set(cities_gdf['geonameid'])
                missing_cities = csv_cities_gdf[~csv_cities_gdf['geonameid'].isin(existing_ids)]
            else:
                # Fall back to name-based comparison
                existing_names = set(cities_gdf['name']) if 'name' in cities_gdf.columns else set()
                missing_cities = csv_cities_gdf[~csv_cities_gdf['name'].isin(existing_names)]

            if len(missing_cities) > 0:
                # Combine datasets
                cities_gdf = pd.concat([cities_gdf, missing_cities], ignore_index=True)
                print(f"✓ Added {len(missing_cities)} supplemental cities from CSV")
        else:

            # If GeoJSON failed, use CSV only
            cities_gdf = csv_cities_gdf

    except Exception as e:
        if cities_gdf is None:
            raise Exception(f"Failed to load both GeoJSON and CSV: {e}")
        print(f"⚠ CSV supplementation failed: {e}")

    print(f"✓ Total cities loaded: {len(cities_gdf)}")
    return cities_gdf


def get_states():
    gdf_path = get_data_file_path('data', 'shape_files', "cb_2023_us_state_20m.shp")
    return gpd.read_file(gdf_path)


def get_counties():
    gdf_path = get_data_file_path('data', 'shape_files', "cb_2023_us_county_20m.shp")
    return gpd.read_file(gdf_path)


def clean_and_select_columns(tweets_with_cities):
    """Select and rename essential columns"""
    cleaned = tweets_with_cities[[
        'FAC', 'LOC', 'GPE', 'time', 'Latitude', 'Longitude',
        'STUSPS__tweet', 'NAME__tweet', 'NAME__county', 'GEOID__county',
        'name', 'geonameid', 'population'
    ]].copy()

    cleaned = cleaned.rename(columns={
        'STUSPS__tweet': 'state_code',
        'NAME__tweet': 'state_name',
        'NAME__county': 'county_name',
        'GEOID__county': 'county_fips',
        'name': 'city_name',
        'geonameid': 'city_id'
    })

    return cleaned

def create_temporal_aggregations(tweets_df, time_bins, us_states_gdf):
    """Create aggregated counts for each time bin"""
    temporal_data = {}

    for bin_time in time_bins:
        bin_tweets = tweets_df[tweets_df['bin'] == bin_time]

        state_counts = bin_tweets.groupby('state_code').size().reset_index(name='tweet_count')
        county_counts = bin_tweets.groupby('county_fips').size().reset_index(name='tweet_count')
        city_counts = bin_tweets.groupby('city_id').size().reset_index(name='tweet_count')

        temporal_data[bin_time] = {
            'states': state_counts,
            'counties': county_counts,
            'cities': city_counts
        }
    return temporal_data

def create_wide_format_shapefile(temporal_data, gdf, output_path, level_name='states'):
    """
    Option 1: Create shapefile with time columns (wide format)
    Each time period becomes a separate column

    Args:
        temporal_data: Your temporal aggregation data
        gdf: GeoDataFrame (states or counties)
        output_path: Where to save the shapefile
        level_name: 'states' or 'counties'
    """
    # Start with the original GeoDataFrame
    result_gdf = gdf.copy()

    # Get the appropriate join columns
    join_col, data_col = _get_join_cols(level_name)

    # Add a column for each time period
    for bin_time, counts_data in temporal_data.items():
        # Create column name (shapefile field names have 10 char limit)
        col_name = f"t_{bin_time.strftime('%m%d_%H%M')}"

        # Get counts for this time period
        time_counts = counts_data[level_name]

        # Merge with result_gdf to get tweet counts
        merged = result_gdf.merge(
            time_counts,
            left_on=join_col,
            right_on=data_col,
            how='left'
        )

        # Add the tweet count column (fill NaN with 0)
        result_gdf[col_name] = merged['tweet_count'].fillna(0)

    # Clean up column names for shapefile compatibility
    result_gdf = clean_shapefile_columns(result_gdf)

    # Save as shapefile
    result_gdf.to_file(output_path)
    print(f"Wide format shapefile saved: {output_path}")

    # Create metadata file explaining the time columns
    metadata_path = output_path.replace('.shp', '_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("Time Column Mappings:\n")
        f.write("=" * 30 + "\n")
        for bin_time in temporal_data.keys():
            col_name = f"t_{bin_time.strftime('%m%d_%H%M')}"
            f.write(f"{col_name} = {bin_time.strftime('%Y-%m-%d %H:%M')}\n")

    return result_gdf


def create_long_format_shapefile(temporal_data, gdf, output_path, level_name='states'):
    """
    Option 2: Create shapefile with repeated geometries (long format)
    Each geometry appears once per time period

    Args:
        temporal_data: Your temporal aggregation data
        gdf: GeoDataFrame (states or counties)
        output_path: Where to save the shapefile
        level_name: 'states' or 'counties'
    """
    all_records = []

    # Get the appropriate join columns
    join_col, data_col = _get_join_cols(level_name)
    # For each time period, create records
    for bin_time, counts_data in temporal_data.items():
        time_counts = counts_data[level_name]

        # Merge with GeoDataFrame
        merged = gdf.merge(
            time_counts,
            left_on=join_col,
            right_on=data_col,
            how='left'
        )

        # Fill NaN tweet counts with 0
        merged['tweet_count'] = merged['tweet_count'].fillna(0)

        # Add timestamp columns
        merged['timestamp'] = bin_time
        merged['time_str'] = bin_time.strftime('%Y-%m-%d %H:%M')
        merged['unix_time'] = int(bin_time.timestamp())

        # Keep essential columns + geometry
        essential_cols = [join_col, 'NAME', 'geometry', 'timestamp', 'time_str', 'unix_time', 'tweet_count']
        if level_name == 'counties':
            essential_cols.append('STATEFP')  # State FIPS for counties

        # Filter to existing columns
        available_cols = [col for col in essential_cols if col in merged.columns]
        merged_clean = merged[available_cols].copy()

        all_records.append(merged_clean)

    # Combine all time periods
    result_gdf = pd.concat(all_records, ignore_index=True)

    # Clean column names for shapefile
    result_gdf = clean_shapefile_columns(result_gdf)

    # Save as shapefile
    result_gdf.to_file(output_path)
    print(f"Long format shapefile saved: {output_path}")
    print(f"Total records: {len(result_gdf)} (geometries × time periods)")

    return result_gdf


def create_separate_time_shapefiles(temporal_data, gdf, output_directory, level_name='states'):
    """
    Option 3: Create separate shapefile for each time period

    Args:
        temporal_data: Your temporal aggregation data
        gdf: GeoDataFrame (states or counties)
        output_directory: Directory to save shapefiles
        level_name: 'states' or 'counties'
    """
    os.makedirs(output_directory, exist_ok=True)

    # Get the appropriate join columns
    join_col, data_col = _get_join_cols(level_name)

    shapefile_paths = []

    for bin_time, counts_data in temporal_data.items():
        # Create filename
        time_str = bin_time.strftime('%Y%m%d_%H%M')
        filename = f"{level_name}_{time_str}.shp"
        output_path = os.path.join(output_directory, filename)

        # Get counts for this time period
        time_counts = counts_data[level_name]

        # Merge with GeoDataFrame
        merged = gdf.merge(
            time_counts,
            left_on=join_col,
            right_on=data_col,
            how='left'
        )

        # Fill NaN with 0
        merged['tweet_count'] = merged['tweet_count'].fillna(0)

        # Add timestamp info
        merged['timestamp'] = bin_time.strftime('%Y-%m-%d %H:%M')
        merged['unix_time'] = int(bin_time.timestamp())

        # Clean column names
        merged_clean = clean_shapefile_columns(merged)

        # Save shapefile
        merged_clean.to_file(output_path)
        shapefile_paths.append(output_path)


    # Create index file listing all shapefiles
    index_path = os.path.join(output_directory, 'shapefile_index.txt')
    with open(index_path, 'w') as f:
        f.write("Temporal Shapefiles Index\n")
        f.write("=" * 30 + "\n")
        for i, (bin_time, path) in enumerate(zip(temporal_data.keys(), shapefile_paths)):
            f.write(f"{i + 1:2d}. {bin_time.strftime('%Y-%m-%d %H:%M')} = {os.path.basename(path)}\n")

    return shapefile_paths


def clean_shapefile_columns(gdf):
    """
    Clean column names to be shapefile-compatible
    Shapefiles have 10-character field name limits
    """
    result = gdf.copy()

    # Rename long column names
    rename_dict = {}
    for col in result.columns:
        if col == 'geometry':
            continue
        if len(col) > 10:
            # Create shortened version
            if 'tweet_count' in col:
                rename_dict[col] = 'tweets'
            elif 'timestamp' in col:
                rename_dict[col] = 'time_stamp'
            elif col.startswith('t_'):
                rename_dict[col] = col[:10]  # Keep first 10 chars
            else:
                rename_dict[col] = col[:10]

    if rename_dict:
        result = result.rename(columns=rename_dict)

    return result


def create_qgis_project_file(shapefile_path, time_column, output_path):
    """
    Create a QGIS project file (.qgs) with temporal settings pre-configured
    This makes it easier to load the temporal data in QGIS
    """
    qgs_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.22.0" projectname="">
  <homePath path=""/>
  <title></title>
  <autotransaction active="0"/>
  <evaluateDefaultValues active="0"/>
  <trust active="0"/>
  <projectCrs>
    <spatialrefsys>
      <wkt>GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]</wkt>
      <proj4>+proj=longlat +datum=WGS84 +no_defs</proj4>
      <srsid>3452</srsid>
      <srid>4326</srid>
      <authid>EPSG:4326</authid>
      <description>WGS 84</description>
    </spatialrefsys>
  </projectCrs>
  <layer-tree-group>
    <customproperties/>
    <layer-tree-layer expanded="1" checked="Qt::Checked" id="temporal_layer" name="Temporal Data" source="{shapefile_path}" providerKey="ogr">
      <customproperties/>
    </layer-tree-layer>
  </layer-tree-group>
  <mapcanvas annotationsVisible="1" name="theMapCanvas">
    <units>degrees</units>
    <extent>
      <xmin>-180</xmin>
      <ymin>-90</ymin>
      <xmax>180</xmax>
      <ymax>90</ymax>
    </extent>
  </mapcanvas>
  <maplayers>
    <maplayer hasScaleBasedVisibilityFlag="0" refreshOnNotifyEnabled="0" maxScale="0" type="vector" styleCategories="AllStyleCategories" refreshOnNotifyMessage="" minScale="100000000" autoRefreshEnabled="0" geometry="Polygon" autoRefreshTime="0">
      <id>temporal_layer</id>
      <datasource>{shapefile_path}</datasource>
      <keywordList>
        <value></value>
      </keywordList>
      <layername>Temporal Data</layername>
      <srs>
        <spatialrefsys>
          <wkt>GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]</wkt>
          <proj4>+proj=longlat +datum=WGS84 +no_defs</proj4>
          <srsid>3452</srsid>
          <srid>4326</srid>
          <authid>EPSG:4326</authid>
          <description>WGS 84</description>
        </spatialrefsys>
      </srs>
      <temporalProperties>
        <enabled>1</enabled>
        <mode>0</mode>
        <startField>{time_column}</startField>
      </temporalProperties>
    </maplayer>
  </maplayers>
</qgis>'''

    with open(output_path, 'w') as f:
        f.write(qgs_content)


def _get_join_cols(level_name: str):
    """
    Return (join_col_on_geometry, data_col_on_temporal_counts)
    """
    if level_name == 'states':
        return 'STUSPS', 'state_code'
    elif level_name == 'counties':
        return 'GEOID', 'county_fips'
    elif level_name == 'cities':
        return 'geonameid', 'city_id'
    else:
        raise ValueError(f"Unknown level_name: {level_name}")

def convert_temporal_data_to_shapefiles(final_tweets, us_states_gdf, us_counties_gdf,us_cities_gdf, label):
    """
    Main function to convert all your temporal data to shapefiles
    """
    # Prepare temporal data (same as your existing code)
    final_tweets['time'] = pd.to_datetime(final_tweets['time'])
    final_tweets['bin'] = final_tweets['time'].dt.floor('4h')
    time_bins = sorted(final_tweets['bin'].unique())
    temporal_data = create_temporal_aggregations(final_tweets, time_bins, us_states_gdf)

    # Create output directory
    output_dir = 'temporal_shapefiles'
    os.makedirs(output_dir, exist_ok=True)

    states_long = create_long_format_shapefile(
        temporal_data, us_states_gdf,
        os.path.join(output_dir, f'{label}_states_long_format.shp'),
        'states'
    )

    counties_long = create_long_format_shapefile(
        temporal_data, us_counties_gdf,
        os.path.join(output_dir, f'{label}_counties_long_format.shp'),
        'counties'
    )
    cities_long   = create_long_format_shapefile(temporal_data, us_cities_gdf,
                       os.path.join(output_dir, f'{label}_cities_long_v2.shp'),   'cities')




# Keep your existing helper functions but add this updated main function
def main():
    label = 'francine'
    # Load and prepare data (same as before)
    tweets_gdf = get_geojson(label).to_crs("EPSG:4326")
    us_cities_gdf = get_cities().to_crs("EPSG:4326")
    us_states_gdf = get_states().to_crs("EPSG:4326")
    us_counties_gdf = get_counties().to_crs("EPSG:4326")

    # Spatial joins (same as before)
    tweets_with_states = gpd.sjoin(tweets_gdf, us_states_gdf, predicate='within', lsuffix='_tweet', rsuffix='_state')
    tweets_with_counties = gpd.sjoin(tweets_with_states, us_counties_gdf, predicate='within', lsuffix='_tweet',
                                     rsuffix='_county')
    tweets_with_cities = gpd.sjoin_nearest(tweets_with_counties, us_cities_gdf, max_distance=0.1,
                                           distance_col='distance_to_city').drop_duplicates()

    final_tweets = clean_and_select_columns(tweets_with_cities)
    convert_temporal_data_to_shapefiles(final_tweets, us_states_gdf, us_counties_gdf,us_cities_gdf, label)



