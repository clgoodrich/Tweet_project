"""Utilities for preparing tweet data and generating Folium visualisations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium import plugins

pd.set_option("display.max_columns", None)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the project's root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_file_path(*path_segments: str) -> Path:
    """Build a path to a data file located within the project directory."""
    return get_project_root().joinpath(*path_segments)


def get_geojson() -> gpd.GeoDataFrame:
    """Load the tweet GeoJSON file."""
    geojson_path = get_data_file_path("data", "geojson", "helene.geojson")
    return gpd.read_file(geojson_path)


def get_cities() -> gpd.GeoDataFrame:
    """Load the GeoNames cities dataset."""
    dataset_path = get_data_file_path("data", "tables", "cities1000.csv")
    df = pd.read_csv(dataset_path)

    us_cities_df = df[
        (df["country_code"] == "US")
        & (df["feature_class"] == "P")
        & (df["population"].notna())
        & (df["latitude"].notna())
        & (df["longitude"].notna())
    ].reset_index(drop=True)

    return gpd.GeoDataFrame(
        us_cities_df,
        geometry=gpd.points_from_xy(us_cities_df.longitude, us_cities_df.latitude),
        crs="EPSG:4326",
    )


def get_states() -> gpd.GeoDataFrame:
    """Load the US states boundaries."""
    return gpd.read_file(get_data_file_path("data", "shape_files", "cb_2023_us_state_20m.shp"))


def get_counties() -> gpd.GeoDataFrame:
    """Load the US counties boundaries."""
    return gpd.read_file(get_data_file_path("data", "shape_files", "cb_2023_us_county_20m.shp"))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def clean_and_select_columns(tweets_with_cities: gpd.GeoDataFrame) -> pd.DataFrame:
    """Select the columns needed for downstream processing."""

    cleaned = tweets_with_cities[
        [
            "FAC",
            "LOC",
            "GPE",
            "time",
            "Latitude",
            "Longitude",
            "STUSPS__tweet",
            "NAME__tweet",
            "NAME__county",
            "GEOID__county",
            "name",
            "geonameid",
            "population",
        ]
    ].copy()

    cleaned = cleaned.rename(
        columns={
            "STUSPS__tweet": "state_code",
            "NAME__tweet": "state_name",
            "NAME__county": "county_name",
            "GEOID__county": "county_fips",
            "name": "city_name",
            "geonameid": "city_id",
        }
    )

    return cleaned


def tweets_counts(
    final_tweets: pd.DataFrame,
    us_states_gdf: gpd.GeoDataFrame,
    us_counties_gdf: gpd.GeoDataFrame,
    us_cities_gdf: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Merge tweet counts with the relevant geographic boundaries."""

    state_counts = final_tweets.groupby("state_code").size().reset_index(name="tweet_count")
    county_counts = final_tweets.groupby("county_fips").size().reset_index(name="tweet_count")
    city_counts = final_tweets.groupby("city_id").size().reset_index(name="tweet_count")

    states_with_counts = us_states_gdf.merge(
        state_counts, left_on="STUSPS", right_on="state_code", how="left"
    )
    states_with_counts["tweet_count"] = states_with_counts["tweet_count"].fillna(0)

    counties_with_counts = us_counties_gdf.merge(
        county_counts, left_on="GEOID", right_on="county_fips", how="left"
    )
    counties_with_counts["tweet_count"] = counties_with_counts["tweet_count"].fillna(0)

    cities_with_counts = us_cities_gdf.merge(
        city_counts, left_on="geonameid", right_on="city_id", how="left"
    )
    cities_with_counts["tweet_count"] = cities_with_counts["tweet_count"].fillna(0)

    return states_with_counts, counties_with_counts, cities_with_counts


# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------

def create_static_map(
    states_with_counts: gpd.GeoDataFrame,
    counties_with_counts: gpd.GeoDataFrame,
    cities_with_counts: gpd.GeoDataFrame,
    output_path: Path | str = "multi_layer_heatmap.html",
) -> None:
    """Create a static Folium map showing tweets by state, county, and city."""

    folium_map = folium.Map(location=[32.0, -83.0], zoom_start=6)

    folium.Choropleth(
        geo_data=states_with_counts,
        data=states_with_counts,
        columns=["STUSPS", "tweet_count"],
        key_on="feature.properties.STUSPS",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Tweets by State",
        name="State Level",
    ).add_to(folium_map)

    folium.Choropleth(
        geo_data=counties_with_counts,
        data=counties_with_counts,
        columns=["GEOID", "tweet_count"],
        key_on="feature.properties.GEOID",
        fill_color="Blues",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Tweets by County",
        name="County Level",
    ).add_to(folium_map)

    for _, row in cities_with_counts[cities_with_counts["tweet_count"] > 0].iterrows():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=max(3, np.sqrt(row["tweet_count"]) * 2),
            popup=(
                f"<b>{row['name']}</b><br>{int(row['tweet_count'])} tweets"  # type: ignore[index]
                f"<br>Pop: {int(row['population']):,}"
            ),
            tooltip=f"{row['name']}: {int(row['tweet_count'])} tweets",  # type: ignore[index]
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.6,
        ).add_to(folium_map)

    folium.LayerControl(collapsed=False).add_to(folium_map)
    folium_map.save(output_path)


# ---------------------------------------------------------------------------
# Temporal processing
# ---------------------------------------------------------------------------

def create_temporal_aggregations(
    tweets_df: pd.DataFrame, time_bins: Sequence[pd.Timestamp]
) -> Dict[pd.Timestamp, Dict[str, pd.DataFrame]]:
    """Aggregate tweet counts for each time bin."""

    temporal_data: Dict[pd.Timestamp, Dict[str, pd.DataFrame]] = {}

    for bin_time in time_bins:
        bin_tweets = tweets_df[tweets_df["bin"] == bin_time]

        state_counts = bin_tweets.groupby("state_code").size().reset_index(name="tweet_count")
        county_counts = bin_tweets.groupby("county_fips").size().reset_index(name="tweet_count")
        city_counts = bin_tweets.groupby("city_id").size().reset_index(name="tweet_count")

        temporal_data[bin_time] = {
            "states": state_counts,
            "counties": county_counts,
            "cities": city_counts,
        }

    return temporal_data


def prepare_heatmap_with_time(
    temporal_data: Dict[pd.Timestamp, Dict[str, pd.DataFrame]],
    cities_gdf: gpd.GeoDataFrame,
) -> Tuple[List[List[List[float]]], List[str]]:
    """Prepare data structures for :class:`HeatMapWithTime`."""

    heat_data: List[List[List[float]]] = []
    time_index: List[str] = []

    for bin_time, counts_data in temporal_data.items():
        bin_cities = cities_gdf.merge(
            counts_data["cities"],
            left_on="geonameid",
            right_on="city_id",
            how="inner",
        )

        bin_heat_data: List[List[float]] = []
        for _, row in bin_cities.iterrows():
            if pd.notna(row.latitude) and pd.notna(row.longitude) and row.tweet_count > 0:
                bin_heat_data.append(
                    [float(row.latitude), float(row.longitude), float(row.tweet_count)]
                )

        heat_data.append(bin_heat_data)
        time_index.append(bin_time.strftime("%Y-%m-%d %H:%M"))

    return heat_data, time_index


def folium_process_dynamic(
    final_tweets: pd.DataFrame, us_cities_gdf: gpd.GeoDataFrame
) -> Tuple[List[List[List[float]]], List[str]]:
    """Generate heatmap data grouped into four-hour bins."""

    tweets = final_tweets.copy()
    tweets["time"] = pd.to_datetime(tweets["time"], errors="coerce")
    tweets = tweets.dropna(subset=["time"])

    tweets["bin"] = tweets["time"].dt.floor("4h")
    time_bins = sorted(tweets["bin"].unique())

    temporal_data = create_temporal_aggregations(tweets, time_bins)
    return prepare_heatmap_with_time(temporal_data, us_cities_gdf)


def create_temporal_map(
    city_heat_data: List[List[List[float]]],
    time_labels: List[str],
    states_gdf: gpd.GeoDataFrame,
    output_path: Path | str = "temporal_heatmap.html",
) -> None:
    """Create a temporal heatmap visualisation for the tweet data."""

    temporal_map = folium.Map(location=[32.0, -83.0], zoom_start=6)

    if city_heat_data and time_labels:
        plugins.HeatMapWithTime(
            data=city_heat_data,
            index=time_labels,
            auto_play=True,
            max_opacity=0.8,
            radius=15,
            name="City Heatmap Over Time",
        ).add_to(temporal_map)

    folium.GeoJson(
        data=json.loads(states_gdf.to_json()),
        name="States",
        style_function=lambda _: {"color": "#555555", "weight": 1, "fillOpacity": 0},
    ).add_to(temporal_map)

    folium.LayerControl().add_to(temporal_map)
    temporal_map.save(output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load data, prepare aggregations, and generate the HTML maps."""

    tweets_gdf = get_geojson().to_crs("EPSG:4326")
    us_cities_gdf = get_cities().to_crs("EPSG:4326")
    us_states_gdf = get_states().to_crs("EPSG:4326")
    us_counties_gdf = get_counties().to_crs("EPSG:4326")

    tweets_with_states = gpd.sjoin(
        tweets_gdf, us_states_gdf, predicate="within", lsuffix="_tweet", rsuffix="_state"
    )
    tweets_with_counties = gpd.sjoin(
        tweets_with_states,
        us_counties_gdf,
        predicate="within",
        lsuffix="_tweet",
        rsuffix="_county",
    )
    tweets_with_cities = gpd.sjoin_nearest(
        tweets_with_counties, us_cities_gdf, max_distance=0.1, distance_col="distance_to_city"
    )

    final_tweets = clean_and_select_columns(tweets_with_cities)

    states_with_counts, counties_with_counts, cities_with_counts = tweets_counts(
        final_tweets, us_states_gdf, us_counties_gdf, us_cities_gdf
    )
    create_static_map(states_with_counts, counties_with_counts, cities_with_counts)

    city_heat_data, time_labels = folium_process_dynamic(final_tweets, us_cities_gdf)
    create_temporal_map(city_heat_data, time_labels, us_states_gdf)

    print("Maps generated successfully!")


if __name__ == "__main__":  # pragma: no cover - allow running as a script
    main()
