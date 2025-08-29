import geopandas as gpd
import pandas as pd
import numpy as np
import os
import folium
import json
from folium import plugins
from shapely.geometry import Point
from shapely.geometry import mapping
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


def tweets_counts(final_tweets, us_states_gdf, us_counties_gdf, us_cities_gdf):
    """Calculate tweet counts by geographic level"""
    state_counts = final_tweets.groupby('state_code').size().reset_index(name='tweet_count')
    county_counts = final_tweets.groupby('county_fips').size().reset_index(name='tweet_count')
    city_counts = final_tweets.groupby('city_id').size().reset_index(name='tweet_count')

    states_with_counts = us_states_gdf.merge(state_counts, left_on='STUSPS', right_on='state_code', how='left')
    states_with_counts['tweet_count'] = states_with_counts['tweet_count'].fillna(0)

    counties_with_counts = us_counties_gdf.merge(county_counts, left_on='GEOID', right_on='county_fips', how='left')
    counties_with_counts['tweet_count'] = counties_with_counts['tweet_count'].fillna(0)

    cities_with_counts = us_cities_gdf.merge(city_counts, left_on='geonameid', right_on='city_id', how='left')
    cities_with_counts['tweet_count'] = cities_with_counts['tweet_count'].fillna(0)

    return states_with_counts, counties_with_counts, cities_with_counts


def folium_process(states_with_counts, counties_with_counts, cities_with_counts):
    """Create static multi-layer map"""
    m = folium.Map(location=[32.0, -83.0], zoom_start=6)

    folium.Choropleth(
        geo_data=states_with_counts,
        data=states_with_counts,
        columns=['STUSPS', 'tweet_count'],
        key_on='feature.properties.STUSPS',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Tweets by State',
        name='State Level'
    ).add_to(m)

    folium.Choropleth(
        geo_data=counties_with_counts,
        data=counties_with_counts,
        columns=['GEOID', 'tweet_count'],
        key_on='feature.properties.GEOID',
        fill_color='Blues',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Tweets by County',
        name='County Level'
    ).add_to(m)

    cities_with_tweets = cities_with_counts[cities_with_counts['tweet_count'] > 0]

    for idx, row in cities_with_tweets.iterrows():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=max(3, np.sqrt(row['tweet_count']) * 2),
            popup=f"<b>{row['name']}</b><br>{int(row['tweet_count'])} tweets<br>Pop: {int(row['population']):,}",
            tooltip=f"{row['name']}: {int(row['tweet_count'])} tweets",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save('multi_layer_heatmap.html')


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


def prepare_timeslider_data(temporal_data, boundary_gdf, join_left, join_right, level_name):
    """
    Prepare data for TimeSliderChoropleth with proper geometry validation
    """
    timeslider_data = []

    for bin_time, counts_data in temporal_data.items():
        # Merge boundary data with tweet counts
        bin_gdf = boundary_gdf.merge(
            counts_data[level_name],
            left_on=join_left,
            right_on=join_right,
            how='left'
        )

        # Fill NaN values and ensure geometry exists
        bin_gdf['tweet_count'] = bin_gdf['tweet_count'].fillna(0)

        # Critical: Remove any rows with invalid geometry
        bin_gdf = bin_gdf[bin_gdf.geometry.notna()]
        bin_gdf = bin_gdf[bin_gdf.geometry.is_valid]

        # Ensure CRS is set
        if bin_gdf.crs is None:
            bin_gdf = bin_gdf.set_crs("EPSG:4326")
        else:
            bin_gdf = bin_gdf.to_crs("EPSG:4326")

        # Convert to GeoJSON dictionary
        geojson_dict = json.loads(bin_gdf.to_json())

        # Add timestamp to each feature's properties
        timestamp_str = bin_time.strftime('%Y-%m-%dT%H:%M:%S')
        for feature in geojson_dict['features']:
            if feature['properties'] is None:
                feature['properties'] = {}
            feature['properties']['time'] = timestamp_str
            # Ensure tweet_count exists
            if 'tweet_count' not in feature['properties']:
                feature['properties']['tweet_count'] = 0

        # Validate that all features have geometry
        valid_features = [f for f in geojson_dict['features']
                          if f.get('geometry') is not None]

        if valid_features:
            geojson_dict['features'] = valid_features
            timeslider_data.append(geojson_dict)
        # for i,k in geojson_dict.items():
        #     for r, m in k.items():
        #         print(r)
        #     # print(k)
    return timeslider_data


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


# def folium_process_dynamic(final_tweets, us_states_gdf, us_counties_gdf, us_cities_gdf):
#     """Process data for dynamic temporal visualization"""
#     final_tweets['time'] = pd.to_datetime(final_tweets['time'])
#     final_tweets['bin'] = final_tweets['time'].dt.floor('4h')
#
#     time_bins = sorted(final_tweets['bin'].unique())
#     temporal_data = create_temporal_aggregations(final_tweets, time_bins, us_states_gdf)
#
#     state_timeslider_data = prepare_timeslider_data(
#         temporal_data,
#         us_states_gdf,
#         'STUSPS',
#         'state_code',
#         'states'
#     )
#
#     # NEW
#     county_timestamped_data = prepare_timestamped_geojson(
#         temporal_data,
#         us_counties_gdf,
#         'GEOID',
#         'county_fips',
#         'counties'
#     )
#
#     # state_timeslider_data = prepare_timeslider_data(
#     #     temporal_data,
#     #     us_states_gdf,
#     #     'STUSPS',
#     #     'state_code',
#     #     'states'
#     # )
#     #
#     # county_timeslider_data = prepare_timeslider_data(
#     #     temporal_data,
#     #     us_counties_gdf,
#     #     'GEOID',
#     #     'county_fips',
#     #     'counties'
#     # )
#
#     city_heat_data, time_labels = prepare_heatmap_with_time(temporal_data, us_cities_gdf)
#
#     return state_timeslider_data, county_timestamped_data, city_heat_data, time_labels
def folium_process_dynamic(final_tweets, us_states_gdf, us_counties_gdf, us_cities_gdf):
    """Process data for dynamic temporal visualization"""
    final_tweets['time'] = pd.to_datetime(final_tweets['time'])
    final_tweets['bin'] = final_tweets['time'].dt.floor('4h')

    time_bins = sorted(final_tweets['bin'].unique())
    temporal_data = create_temporal_aggregations(final_tweets, time_bins, us_states_gdf)

    # Use TimestampedGeoJson instead of TimeSliderChoropleth
    state_timestamped_data = prepare_timestamped_geojson(
        temporal_data,
        us_states_gdf,
        'STUSPS',
        'state_code',
        'states'
    )

    county_timestamped_data = prepare_timestamped_geojson(
        temporal_data,
        us_counties_gdf,
        'GEOID',
        'county_fips',
        'counties'
    )

    city_heat_data, time_labels = prepare_heatmap_with_time(temporal_data, us_cities_gdf)

    return state_timestamped_data, county_timestamped_data, city_heat_data, time_labels


# def prepare_timestamped_geojson(temporal_data, boundary_gdf, join_left, join_right, level_name):
#     """Prepare data for TimestampedGeoJson"""
#     features = []
#
#     for bin_time, counts_data in temporal_data.items():
#         # Merge boundary data with counts
#         bin_gdf = boundary_gdf.merge(
#             counts_data[level_name],
#             left_on=join_left,
#             right_on=join_right,
#             how='left'
#         )
#         bin_gdf['tweet_count'] = bin_gdf['tweet_count'].fillna(0)
#         bin_gdf = bin_gdf[bin_gdf.geometry.notna()]
#
#         # Create features for this timestamp
#         for _, row in bin_gdf.iterrows():
#             feature = {
#                 'type': 'Feature',
#                 'geometry': json.loads(gpd.GeoSeries([row.geometry]).to_json())['features'][0]['geometry'],
#                 'properties': {
#                     'time': bin_time.isoformat(),
#                     'tweet_count': float(row['tweet_count']),
#                     'state_code': row[join_left],
#                     'style': {
#                         'fillColor': 'red' if row['tweet_count'] > 0 else 'lightgray',
#                         'fillOpacity': min(0.8, row['tweet_count'] / 50),  # Scale opacity
#                         'weight': 1
#                     }
#                 }
#             }
#             features.append(feature)
#
#     return {'type': 'FeatureCollection', 'features': features}
def prepare_timestamped_geojson(temporal_data, boundary_gdf, join_left, join_right, level_name):
    """Prepare data for TimestampedGeoJson with visible styling"""
    features = []

    for bin_time, counts_data in temporal_data.items():
        bin_gdf = boundary_gdf.merge(
            counts_data[level_name],
            left_on=join_left,
            right_on=join_right,
            how='left'
        )
        bin_gdf['tweet_count'] = bin_gdf['tweet_count'].fillna(0)
        bin_gdf = bin_gdf[bin_gdf.geometry.notna()]

        for _, row in bin_gdf.iterrows():
            # Calculate opacity: minimum 0.1 so all polygons are visible
            opacity = max(0.1, min(0.8, row['tweet_count'] / 20))

            feature = {
                'type': 'Feature',
                'geometry': mapping(row.geometry),
                'properties': {
                    'time': bin_time.isoformat(),
                    'tweet_count': float(row['tweet_count']),
                    'state_code': row[join_left],
                    'style': {
                        'fillColor': 'red' if row['tweet_count'] > 0 else 'lightblue',
                        'fillOpacity': opacity,  # Always visible
                        'color': 'black',  # Border color
                        'weight': 0.5
                    }
                }
            }
            features.append(feature)

    return {'type': 'FeatureCollection', 'features': features}

# def generate_folium_for_dynamic(state_timeslider_data, county_timeslider_data, city_heat_data, time_labels):
#     """Generate map with TimestampedGeoJson"""
#     m = folium.Map(location=[32.0, -83.0], zoom_start=6)
#
#     # Convert your existing data to TimestampedGeoJson format
#     timestamped_features = []
#     for time_bin in state_timeslider_data:
#         for feature in time_bin['features']:
#             feature['properties']['style'] = {
#                 'fillColor': 'red' if feature['properties']['tweet_count'] > 0 else 'lightgray',
#                 'fillOpacity': min(0.8, feature['properties']['tweet_count'] / 50),
#                 'weight': 1
#             }
#             timestamped_features.append(feature)
#
#     # Add TimestampedGeoJson layer
#     plugins.TimestampedGeoJson({
#         'type': 'FeatureCollection',
#         'features': timestamped_features
#     }, period='P4H', add_last_point=True).add_to(m)
#
#     # Add working HeatMapWithTime
#     plugins.HeatMapWithTime(
#         city_heat_data,
#         index=time_labels,
#         auto_play=True,
#         max_opacity=0.8,
#         radius=15
#     ).add_to(m)
#
#     return m

def generate_folium_for_dynamic(state_timestamped_data, county_timestamped_data, city_heat_data, time_labels):
    """Debug TimestampedGeoJson data structure"""
    m = folium.Map(location=[32.0, -83.0], zoom_start=6)
    print(state_timestamped_data)
    # Debug the actual data structure
    # print("=== TimestampedGeoJson Debug ===")
    # print(f"Data type: {type(state_timestamped_data)}")
    # print(
    #     f"Data keys: {list(state_timestamped_data.keys()) if isinstance(state_timestamped_data, dict) else 'Not a dict'}")
    #
    # if isinstance(state_timestamped_data, dict) and 'features' in state_timestamped_data:
    #     print(f"Number of features: {len(state_timestamped_data['features'])}")
    #
    #     if state_timestamped_data['features']:
    #         first_feature = state_timestamped_data['features'][0]
    #         print(f"First feature keys: {list(first_feature.keys())}")
    #         print(f"First feature properties: {first_feature.get('properties', {}).keys()}")
    #         print(f"First feature time: {first_feature.get('properties', {}).get('time')}")

    # Try adding TimestampedGeoJson with error handling
    try:
        plugins.TimestampedGeoJson(
            state_timestamped_data,
            period='P4H',
            add_last_point=True
        ).add_to(m)
        print("TimestampedGeoJson added successfully")
    except Exception as e:
        print(f"TimestampedGeoJson error: {e}")
        print("Falling back to cities only")

    # Add cities (this should always work)
    plugins.HeatMapWithTime(
        city_heat_data,
        index=time_labels,
        auto_play=True,
        max_opacity=0.8,
        radius=15
    ).add_to(m)

    return m

def main():
    # Load and prepare data
    tweets_gdf = get_geojson().to_crs("EPSG:4326")
    us_cities_gdf = get_cities().to_crs("EPSG:4326")
    us_states_gdf = get_states().to_crs("EPSG:4326")
    us_counties_gdf = get_counties().to_crs("EPSG:4326")

    # Spatial joins
    tweets_with_states = gpd.sjoin(tweets_gdf, us_states_gdf, predicate='within', lsuffix='_tweet', rsuffix='_state')
    tweets_with_counties = gpd.sjoin(tweets_with_states, us_counties_gdf, predicate='within', lsuffix='_tweet',
                                     rsuffix='_county')
    tweets_with_cities = gpd.sjoin_nearest(tweets_with_counties, us_cities_gdf, max_distance=0.1,
                                           distance_col='distance_to_city')

    # Clean data
    final_tweets = clean_and_select_columns(tweets_with_cities)

    # Create static map
    states_with_counts, counties_with_counts, cities_with_counts = tweets_counts(
        final_tweets, us_states_gdf, us_counties_gdf, us_cities_gdf
    )
    folium_process(states_with_counts, counties_with_counts, cities_with_counts)

    # Create temporal map
    state_timeslider_data, county_timeslider_data, city_heat_data, time_labels = folium_process_dynamic(
        final_tweets, us_states_gdf, us_counties_gdf, us_cities_gdf
    )
    temporal_map = generate_folium_for_dynamic(
        state_timeslider_data, county_timeslider_data, city_heat_data, time_labels
    )
    temporal_map.save('temporal_heatmap.html')

    print("Maps generated successfully!")


