import geopandas as gpd
import pandas as pd
import numpy as np
import os
import folium
import json
from folium import plugins
from shapely.geometry import Point

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


def folium_process_dynamic(final_tweets, us_states_gdf, us_counties_gdf, us_cities_gdf):
    """Process data for dynamic temporal visualization"""
    final_tweets['time'] = pd.to_datetime(final_tweets['time'])
    final_tweets['bin'] = final_tweets['time'].dt.floor('4h')

    time_bins = sorted(final_tweets['bin'].unique())
    print("\n\n\n\n\n\n\n______________________________")

    print(final_tweets)
    print("\n\n\n\n\n\n\n______________________________")
    print(time_bins)
    print("\n\n\n\n\n\n\n______________________________")

    print(us_states_gdf)
    temporal_data = create_temporal_aggregations(final_tweets, time_bins, us_states_gdf)



    state_timeslider_data = prepare_timeslider_data(
        temporal_data,
        us_states_gdf,
        'STUSPS',
        'state_code',
        'states'
    )

    county_timeslider_data = prepare_timeslider_data(
        temporal_data,
        us_counties_gdf,
        'GEOID',
        'county_fips',
        'counties'
    )

    city_heat_data, time_labels = prepare_heatmap_with_time(temporal_data, us_cities_gdf)

    return state_timeslider_data, county_timeslider_data, city_heat_data, time_labels


def generate_folium_for_dynamic(state_timeslider_data, county_timeslider_data, city_heat_data, time_labels):
    """Generate dynamic temporal map with TimeSliderChoropleth"""
    """Generate dynamic temporal map with debugging"""
    print(pd.DataFrame(state_timeslider_data))
    print(foo)

    m = folium.Map(location=[32.0, -83.0], zoom_start=6)
    # Validate data before creating layers
    data_frame_used = pd.DataFrame(state_timeslider_data).iloc[0]['features']
    # for i in data_frame_used:
    #     print(i)
    # print(all_keys)

    if state_timeslider_data and len(state_timeslider_data) > 0:
        # try:
        state_temporal_layer = plugins.TimeSliderChoropleth(
            data=state_timeslider_data,
            styledict={
                'fillColor': 'red',
                'fillOpacity': 0.7,
                'color': 'black',
                'weight': 1
            },
            name='States Over Time'
        )
        state_temporal_layer.add_to(m)
        # except ValueError as e:
        #     # print(f"Error creating state layer: {e}")
        #     pass
    # #
    # if county_timeslider_data and len(county_timeslider_data) > 0:
    #     try:
    #         county_temporal_layer = plugins.TimeSliderChoropleth(
    #             data=county_timeslider_data,
    #             styledict={
    #                 'fillColor': 'blue',
    #                 'fillOpacity': 0.5,
    #                 'color': 'black',
    #                 'weight': 0.5
    #             },
    #             name='Counties Over Time'
    #         )
    #         county_temporal_layer.add_to(m)
    #     except ValueError as e:
    #         print(f"Error creating county layer: {e}")
    if city_heat_data and len(city_heat_data) > 0 and len(time_labels) > 0:
        try:
            plugins.HeatMapWithTime(
                data=city_heat_data,
                auto_play=True,
                max_opacity=0.8,
                radius=15,
                name='City Heatmap Over Time'
            ).add_to(m)
        except Exception as e:
            print(f"Error creating heatmap: {e}")

    folium.LayerControl().add_to(m)
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


