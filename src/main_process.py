import geopandas as gpd
import pandas as pd
import numpy as np
import os
import folium
import json
from folium import plugins
from folium.plugins import TimeSliderChoropleth
from shapely.geometry import Point
import branca.colormap as cm
from shapely.validation import make_valid
pd.set_option('display.max_columns', None)


def get_project_root():
    """Gets the absolute path to the project's root folder."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_file_path(*path_segments):
    """Builds a full path to any data file from the project root."""
    project_root = get_project_root()
    return os.path.join(project_root, *path_segments)


def get_geojson():
    """Get path to helene.geojson"""
    geojson = get_data_file_path('data', 'geojson', 'helene.geojson')
    return gpd.read_file(geojson)


def get_cities():
    df_path = get_data_file_path('data', 'tables', 'cities1000.csv')
    df = pd.read_csv(df_path)
    us_cities_df = df[
        (df['country_code'] == 'US') &
        (df['feature_class'] == 'P') &
        (df['population'].notna()) &
        (df['latitude'].notna()) &
        (df['longitude'].notna())
        ].reset_index(drop=True)

    us_cities_gdf = gpd.GeoDataFrame(
        us_cities_df,
        geometry=gpd.points_from_xy(us_cities_df.longitude, us_cities_df.latitude),
        crs="EPSG:4326"
    )
    return us_cities_gdf


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



def prepare_heatmap_with_time(temporal_data, cities_gdf):
    """Prepare data for HeatMapWithTime"""
    heat_data = []
    time_index = []

    for bin_time, counts_data in temporal_data.items():
        bin_cities = cities_gdf.merge(
            counts_data['cities'],
            left_on='geonameid',
            right_on='city_id',
            how='inner'
        )

        bin_heat_data = []
        for _, row in bin_cities.iterrows():
            # Ensure valid coordinates and counts
            if pd.notna(row.latitude) and pd.notna(row.longitude) and row.tweet_count > 0:
                bin_heat_data.append([
                    float(row.latitude),
                    float(row.longitude),
                    float(row.tweet_count)
                ])

        heat_data.append(bin_heat_data)
        time_index.append(bin_time.strftime('%Y-%m-%d %H:%M'))

    return heat_data, time_index

def temporal_data_process(final_tweets, gdf):
    time_columns = []
    final_tweets['time'] = pd.to_datetime(final_tweets['time'])
    final_tweets['bin'] = final_tweets['time'].dt.floor('4h')

    time_bins = sorted(final_tweets['bin'].unique())
    temporal_data = create_temporal_aggregations(final_tweets, time_bins, gdf)

    for bin_time, counts_data in temporal_data.items():
        # Create column name for this time bin
        time_col = f"tweets_{bin_time.strftime('%Y%m%d_%H%M')}"
        time_columns.append(time_col)
    return temporal_data, time_columns


def color_determiner(value):
    """Fixed color determiner with proper hex colors"""
    if value == 0:
        return '#ffffff'  # white - no data
    elif value == 1:
        return '#ffff99'  # light yellow
    elif 5 >= value > 1:
        return '#ff9933'  # orange
    elif 10 >= value > 5:
        return '#ff3333'  # red
    elif value > 10:
        return '#990000'  # dark red
    return '#ffffff'  # default white


def style_dict_process_fixed(time_bins, gdf, temporal_data, label):
    """
    Create styledict with proper timestamp formatting for TimeSliderChoropleth.

    This is the key fix - we need timestamps as strings matching the folium expectation.
    """
    styledict = {}

    # Configure based on geographic level
    if label == 'states':
        code, type_col = 'state_code', 'STUSPS'
    else:
        code, type_col = 'county_fips', 'GEOID'

    # Convert time_bins to timestamp strings (Unix timestamps)
    timestamp_strings = {}
    for bin_time in time_bins:
        # Convert to Unix timestamp string
        timestamp_str = str(int(bin_time.timestamp()))
        timestamp_strings[bin_time] = timestamp_str

    # Create styledict for each geographic feature
    for feature_idx, row in gdf.iterrows():
        feature_styles = {}

        # For each time period, determine the style
        for bin_time in time_bins:
            timestamp_str = timestamp_strings[bin_time]

            # Get temporal data for this time bin
            temporal_data_here = temporal_data[bin_time][label]
            feature_id = row[type_col]  # e.g., 'GA', 'FL', or county FIPS

            # Check if this feature has data for this time period
            feature_data = temporal_data_here[temporal_data_here[code] == feature_id]

            if not feature_data.empty:
                tweet_count = int(float(feature_data['tweet_count'].iloc[0]))
                color = color_determiner(tweet_count)
                opacity = max(0.3, min(0.9, tweet_count / 20))  # Scale opacity reasonably
                fill_opacity = opacity
            else:
                color = '#ffffff'  # White for no data
                opacity = 0.1
                fill_opacity = 0.0

            # Store style for this timestamp
            feature_styles[timestamp_str] = {
                'color': '#000000',  # Border color (black)
                'weight': 0.5,  # Border weight
                'fillColor': color,  # Fill color based on tweet count
                'fillOpacity': fill_opacity,
                'opacity': opacity
            }

        # Use feature index as key (important!)
        styledict[str(feature_idx)] = feature_styles

    return styledict


def create_separate_maps(final_tweets, us_states_gdf, us_counties_gdf, us_cities_gdf):
    """
    Create separate maps to avoid conflicts between TimeSliderChoropleth and HeatMapWithTime
    """
    # Prepare temporal data
    final_tweets['time'] = pd.to_datetime(final_tweets['time'])
    final_tweets['bin'] = final_tweets['time'].dt.floor('4h')
    time_bins = sorted(final_tweets['bin'].unique())

    temporal_data = create_temporal_aggregations(final_tweets, time_bins, us_states_gdf)

    # Create styledicts with proper timestamps
    styledata_states = style_dict_process_fixed(time_bins, us_states_gdf, temporal_data, 'states')
    styledata_counties = style_dict_process_fixed(time_bins, us_counties_gdf, temporal_data, 'counties')

    # Prepare city heatmap data
    city_heat_data, time_labels = prepare_heatmap_with_time(temporal_data, us_cities_gdf)

    return styledata_states, styledata_counties, city_heat_data, time_labels, time_bins


def create_choropleth_map(us_states_gdf, us_counties_gdf, styledata_states, styledata_counties):
    """
    Create map with only TimeSliderChoropleth layers
    """
    m = folium.Map(location=[32.0, -83.0], zoom_start=6)

    # Add states layer
    try:
        states_layer = TimeSliderChoropleth(
            data=us_states_gdf.to_json(),
            styledict=styledata_states,
            name='States Over Time'
        )
        states_layer.add_to(m)
        print("States TimeSliderChoropleth added successfully")
    except Exception as e:
        print(f"Error adding states layer: {e}")

    # Add counties layer
    try:
        counties_layer = TimeSliderChoropleth(
            data=us_counties_gdf.to_json(),
            styledict=styledata_counties,
            name='Counties Over Time'
        )
        counties_layer.add_to(m)
        print("Counties TimeSliderChoropleth added successfully")
    except Exception as e:
        print(f"Error adding counties layer: {e}")

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def create_heatmap_map(city_heat_data, time_labels):
    """
    Create separate map with only HeatMapWithTime
    """
    m = folium.Map(location=[32.0, -83.0], zoom_start=6)

    if city_heat_data and len(city_heat_data) > 0 and len(time_labels) > 0:
        try:
            heatmap_layer = plugins.HeatMapWithTime(
                data=city_heat_data,
                index=time_labels,  # Use index parameter for proper time labels
                auto_play=False,  # Start with manual control
                max_opacity=0.8,
                radius=15,
                name='City Heatmap Over Time'
            )
            heatmap_layer.add_to(m)
            print("HeatMapWithTime added successfully")
        except Exception as e:
            print(f"Error creating heatmap: {e}")

    folium.LayerControl().add_to(m)
    return m


def create_combined_map_carefully(us_states_gdf, styledata_states, city_heat_data, time_labels):
    """
    Attempt to combine both types, but with careful handling
    """
    m = folium.Map(location=[32.0, -83.0], zoom_start=6)

    # Add TimeSliderChoropleth first
    try:
        states_layer = TimeSliderChoropleth(
            data=us_states_gdf.to_json(),
            styledict=styledata_states,
            name='States Over Time'
        )
        states_layer.add_to(m)
        print("States layer added to combined map")
    except Exception as e:
        print(f"Error adding states to combined map: {e}")

    # Add HeatMapWithTime with different configuration
    if city_heat_data and len(city_heat_data) > 0:
        try:
            # Use different time labels format to avoid conflicts
            simplified_labels = [f"T{i}" for i in range(len(time_labels))]

            heatmap_layer = plugins.HeatMapWithTime(
                data=city_heat_data,
                index=simplified_labels,
                auto_play=False,
                max_opacity=0.6,
                radius=10,
                name='Cities'
            )
            heatmap_layer.add_to(m)
            print("Heatmap added to combined map")
        except Exception as e:
            print(f"Error adding heatmap to combined map: {e}")

    folium.LayerControl(collapsed=False).add_to(m)
    return m


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
    if level_name == 'states':
        join_col = 'STUSPS'
        data_col = 'state_code'
    else:
        join_col = 'GEOID'
        data_col = 'county_fips'

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
    if level_name == 'states':
        join_col = 'STUSPS'
        data_col = 'state_code'
    else:
        join_col = 'GEOID'
        data_col = 'county_fips'

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
    if level_name == 'states':
        join_col = 'STUSPS'
        data_col = 'state_code'
    else:
        join_col = 'GEOID'
        data_col = 'county_fips'

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

        print(f"Created: {filename}")

    # Create index file listing all shapefiles
    index_path = os.path.join(output_directory, 'shapefile_index.txt')
    with open(index_path, 'w') as f:
        f.write("Temporal Shapefiles Index\n")
        f.write("=" * 30 + "\n")
        for i, (bin_time, path) in enumerate(zip(temporal_data.keys(), shapefile_paths)):
            f.write(f"{i + 1:2d}. {bin_time.strftime('%Y-%m-%d %H:%M')} = {os.path.basename(path)}\n")

    print(f"\nCreated {len(shapefile_paths)} shapefiles in: {output_directory}")
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
        print(f"Renamed columns for shapefile compatibility: {rename_dict}")

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

    print(f"QGIS project file created: {output_path}")


def convert_temporal_data_to_shapefiles(final_tweets, us_states_gdf, us_counties_gdf):
    """
    Main function to convert all your temporal data to shapefiles
    """
    print("Converting temporal data to shapefiles...")

    # Prepare temporal data (same as your existing code)
    final_tweets['time'] = pd.to_datetime(final_tweets['time'])
    final_tweets['bin'] = final_tweets['time'].dt.floor('4h')
    time_bins = sorted(final_tweets['bin'].unique())
    temporal_data = create_temporal_aggregations(final_tweets, time_bins, us_states_gdf)

    # Create output directory
    output_dir = 'temporal_shapefiles'
    os.makedirs(output_dir, exist_ok=True)

    # Option 1: Wide format shapefiles
    print("\n1. Creating wide format shapefiles...")
    states_wide = create_wide_format_shapefile(
        temporal_data, us_states_gdf,
        os.path.join(output_dir, 'states_wide_format.shp'),
        'states'
    )

    counties_wide = create_wide_format_shapefile(
        temporal_data, us_counties_gdf,
        os.path.join(output_dir, 'counties_wide_format.shp'),
        'counties'
    )

    # Option 2: Long format shapefiles
    print("\n2. Creating long format shapefiles...")
    states_long = create_long_format_shapefile(
        temporal_data, us_states_gdf,
        os.path.join(output_dir, 'states_long_format.shp'),
        'states'
    )

    counties_long = create_long_format_shapefile(
        temporal_data, us_counties_gdf,
        os.path.join(output_dir, 'counties_long_format.shp'),
        'counties'
    )

    # Option 3: Separate shapefiles per time period
    print("\n3. Creating separate shapefiles per time period...")
    create_separate_time_shapefiles(
        temporal_data, us_states_gdf,
        os.path.join(output_dir, 'states_by_time'),
        'states'
    )

    create_separate_time_shapefiles(
        temporal_data, us_counties_gdf,
        os.path.join(output_dir, 'counties_by_time'),
        'counties'
    )

    # Create QGIS project files for easy loading
    create_qgis_project_file(
        os.path.join(output_dir, 'states_long_format.shp'),
        'time_str',
        os.path.join(output_dir, 'states_temporal.qgs')
    )

    create_qgis_project_file(
        os.path.join(output_dir, 'counties_long_format.shp'),
        'time_str',
        os.path.join(output_dir, 'counties_temporal.qgs')
    )

    print(f"\nAll shapefiles created in: {output_dir}")
    print("\nRecommended usage:")
    print("- Wide format: Good for ArcGIS Pro time slider")
    print("- Long format: Good for QGIS temporal controller")
    print("- Separate files: Good for manual time analysis")
    # print("\n" + "=" * 50)
    # print("QGIS SETUP INSTRUCTIONS:")
    # print("=" * 50)
    # print("1. Open QGIS")
    # print("2. Load the *_long_format.shp file")
    # print("3. Enable Temporal Controller Panel:")
    # print("   - Go to View menu → Panels → Temporal Controller")
    # print("   - Or press Ctrl+1 (Windows/Linux) or Cmd+1 (Mac)")
    # print("   - The temporal panel will appear (usually docked at bottom)")
    # print("4. Configure layer for time:")
    # print("   - Right-click layer → Properties → Temporal tab")
    # print("   - Check 'Dynamic Temporal Control'")
    # print("   - Set Configuration: 'Single Field with Date/Time'")
    # print("   - Set Field: 'timestamp' (this is now a proper datetime field)")
    # print("   - Click OK")
    # print("5. Use the temporal controls:")
    # print("   - In Temporal Controller panel, click the green play button")
    # print("   - Or manually drag the time slider")
    # print("   - Use step forward/backward buttons for manual control")
    # print("6. Optional - Set time range:")
    # print("   - In Temporal Controller, set Fixed Range")
    # print("   - Choose appropriate time step (e.g., 4 hours)")
    # print("=" * 50)

# Keep your existing helper functions but add this updated main function
def main():
    # Load and prepare data (same as before)
    tweets_gdf = get_geojson().to_crs("EPSG:4326")
    us_cities_gdf = get_cities().to_crs("EPSG:4326")
    us_states_gdf = get_states().to_crs("EPSG:4326")
    us_counties_gdf = get_counties().to_crs("EPSG:4326")

    # Spatial joins (same as before)
    tweets_with_states = gpd.sjoin(tweets_gdf, us_states_gdf, predicate='within', lsuffix='_tweet', rsuffix='_state')
    tweets_with_counties = gpd.sjoin(tweets_with_states, us_counties_gdf, predicate='within', lsuffix='_tweet',
                                     rsuffix='_county')
    tweets_with_cities = gpd.sjoin_nearest(tweets_with_counties, us_cities_gdf, max_distance=0.1,
                                           distance_col='distance_to_city')

    # Clean data (same as before)
    final_tweets = clean_and_select_columns(tweets_with_cities)
    convert_temporal_data_to_shapefiles(final_tweets, us_states_gdf, us_counties_gdf)
    # Create temporal data with fixed timestamps
    styledata_states, styledata_counties, city_heat_data, time_labels, time_bins = create_separate_maps(
        final_tweets, us_states_gdf, us_counties_gdf, us_cities_gdf
    )

    print(f"Created temporal data for {len(time_bins)} time periods")
    print(f"First timestamp in styledata: {list(next(iter(styledata_states.values())).keys())[0]}")

    # Create toggleable map using FeatureGroups
    toggleable_map = create_toggleable_timeslider_map(us_states_gdf, us_counties_gdf, styledata_states,
                                                      styledata_counties)
    toggleable_map.save('toggleable_timeslider.html')
    print("Toggleable map saved as 'toggleable_timeslider.html'")

    # Create separate maps to avoid conflicts

    # 1. Choropleth-only map (states and counties)
    choropleth_map = create_choropleth_map(us_states_gdf, us_counties_gdf, styledata_states, styledata_counties)
    choropleth_map.save('choropleth_timeslider.html')
    print("Choropleth map saved as 'choropleth_timeslider.html'")

    # 2. Heatmap-only map (cities)
    heatmap_map = create_heatmap_map(city_heat_data, time_labels)
    heatmap_map.save('heatmap_timeslider.html')
    print("Heatmap map saved as 'heatmap_timeslider.html'")

    print("All maps generated successfully!")


def create_toggleable_timeslider_map(us_states_gdf, us_counties_gdf, styledata_states, styledata_counties):
    """
    Create map with toggleable TimeSliderChoropleth layers using FeatureGroups
    """
    m = folium.Map(location=[32.0, -83.0], zoom_start=6)

    # Create FeatureGroups for each layer
    states_group = folium.FeatureGroup(name='States Over Time', show=True)
    counties_group = folium.FeatureGroup(name='Counties Over Time', show=False)  # Start hidden

    # Add TimeSliderChoropleth to FeatureGroups
    try:
        states_layer = TimeSliderChoropleth(
            data=us_states_gdf.to_json(),
            styledict=styledata_states,
            name='States Temporal'  # Internal name
        )
        states_layer.add_to(states_group)
        print("States TimeSliderChoropleth added to FeatureGroup")
    except Exception as e:
        print(f"Error adding states layer: {e}")

    try:
        counties_layer = TimeSliderChoropleth(
            data=us_counties_gdf.to_json(),
            styledict=styledata_counties,
            name='Counties Temporal'  # Internal name
        )
        counties_layer.add_to(counties_group)
        print("Counties TimeSliderChoropleth added to FeatureGroup")
    except Exception as e:
        print(f"Error adding counties layer: {e}")

    # Add FeatureGroups to map
    states_group.add_to(m)
    counties_group.add_to(m)

    # Add LayerControl for toggling
    folium.LayerControl(
        collapsed=False,
        position='topright'
    ).add_to(m)

    return m


# Debug function to inspect timestamp format
def debug_timestamps(styledata, sample_feature=None):
    """Debug helper to check timestamp formatting"""
    if not styledata:
        print("No styledata provided")
        return

    # Get first feature's data
    first_key = list(styledata.keys())[0] if sample_feature is None else str(sample_feature)
    first_feature_data = styledata[first_key]

    print(f"Feature {first_key} timestamps:")
    for timestamp, style in list(first_feature_data.items())[:3]:  # Show first 3
        print(f"  {timestamp}: {style}")

    # Check timestamp format
    sample_timestamp = list(first_feature_data.keys())[0]
    print(f"Sample timestamp: '{sample_timestamp}' (type: {type(sample_timestamp)})")

    # Try to convert back to datetime
    try:
        import datetime
        dt = datetime.datetime.fromtimestamp(int(sample_timestamp))
        print(f"Converts to: {dt}")
    except:
        print("Cannot convert timestamp back to datetime")