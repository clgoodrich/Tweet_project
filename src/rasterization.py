"""
Tweet Project — Rasterization (Step-by-Step)
===========================================

This module builds a master analysis grid and converts per-interval, multi-level
geographic matches into rasters with hierarchical weights. It writes both
incremental and cumulative GeoTIFFs per time bin for each hurricane.

OVERVIEW (End-to-End Steps)
---------------------------
STEP 1 — MASTER GRID
    • Reproject inputs to TARGET_CRS (config).
    • Compute union extent and grid dims from CELL_SIZE_M.
    • Build affine transform and projected geometry lookups.

STEP 2 — FACILITY HEAT (OPTIONAL)
    • For facility points, create a Gaussian-kernel "hotspot" layer, scaled
      by total facility counts and a FACILITY weight.

STEP 3 — HIERARCHICAL RASTER
    • Ensure parent states are present when counties/cities exist.
    • Rasterize STATE/COUNTY/CITY polygons with log1p(count) * weight.
    • Add FACILITY raster if present.

STEP 4 — WRITE GeoTIFF
    • Save a single-band GTiff using master transform/CRS.
    • Separate subfolders for 'increment' and 'cumulative'.

STEP 5 — PROCESS HURRICANE
    • For each time bin: build incremental grid, accumulate into cumulative,
      and save both.

Notes & Pitfalls
----------------
• Extent & alignment: using `from_bounds` with ceil dims may include a border
  of partial pixels; this is fine for visualization but be aware when aligning
  to other rasters.
• `all_touched=True` grows polygon fill across pixel edges; set False if you
  prefer stricter masks (edge pixels excluded unless centers are inside).
• State inclusion: parent state is inferred by centroid containment of county/city.
  If a county/city lies on borders, centroid test may pick a neighbor state.
• Facility CRS assumption: facility geometry is treated as EPSG:4326 then
  reprojected to TARGET_CRS. If your matched point is already projected, this
  is a no-op only if it was WGS84 originally (see inline note).

"""

from __future__ import annotations

# STEP 0 — IMPORTS
import os
from typing import Any, Dict, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter

import config


# ------------------------------------------------------------------------------
# STEP 1 — MASTER GRID
# ------------------------------------------------------------------------------

def create_master_grid(
    francine_gdf: gpd.GeoDataFrame,
    helene_gdf: gpd.GeoDataFrame,
    states_gdf: gpd.GeoDataFrame,
    counties_gdf: gpd.GeoDataFrame,
    cities_gdf: gpd.GeoDataFrame,
) -> Dict[str, Any]:
    """
    Create the master grid canvas and projected geometry lookups.

    Operations
    ----------
    1) Reproject input layers to config.TARGET_CRS.
    2) Compute a union bounding box from the two hurricane datasets.
    3) Derive width/height from CELL_SIZE_M and bounds.
    4) Build an affine transform for rasterization.
    5) Build projected name→geometry lookups for states/counties/cities.

    Returns
    -------
    dict
        {
          'crs' : str (TARGET_CRS),
          'cell_size' : int (meters),
          'width' : int,
          'height' : int,
          'bounds' : (minx, miny, maxx, maxy),
          'transform' : Affine,
          'state_lookup_proj'  : {NAME_UPPER -> Polygon/MultiPolygon},
          'county_lookup_proj' : {NAME_UPPER -> Polygon/MultiPolygon},
          'cities_lookup_proj' : {NAME_UPPER -> Point/MultiPoint},
          'francine_proj' : GeoDataFrame,
          'helene_proj'   : GeoDataFrame
        }

    Side Effects
    ------------
    Prints a configuration banner for quick sanity checks.
    """
    print("=" * 60)
    print("CREATING MASTER GRID CANVAS")
    print("=" * 60)

    # 1) Project datasets to target CRS
    print(f"\nProjecting datasets to {config.TARGET_CRS}...")
    francine_proj = francine_gdf.to_crs(config.TARGET_CRS)
    helene_proj = helene_gdf.to_crs(config.TARGET_CRS)

    states_proj = states_gdf.to_crs(config.TARGET_CRS)
    counties_proj = counties_gdf.to_crs(config.TARGET_CRS)
    cities_proj = cities_gdf.to_crs(config.TARGET_CRS)

    # 2) Compute union extent from francine+helene (reference layers are larger)
    print("\nCalculating master extent...")
    francine_bounds = francine_proj.total_bounds  # (minx, miny, maxx, maxy)
    helene_bounds = helene_proj.total_bounds

    minx = min(francine_bounds[0], helene_bounds[0])
    miny = min(francine_bounds[1], helene_bounds[1])
    maxx = max(francine_bounds[2], helene_bounds[2])
    maxy = max(francine_bounds[3], helene_bounds[3])

    # 3) Grid dimensions from cell size
    width = int(np.ceil((maxx - minx) / config.CELL_SIZE_M))
    height = int(np.ceil((maxy - miny) / config.CELL_SIZE_M))

    print(f"\nGrid Configuration:")
    print(f"  Cell size: {config.CELL_SIZE_M:,} meters")
    print(f"  Grid dimensions: {width} x {height} cells")
    print(f"  Total cells: {width * height:,}")

    # 4) Affine transform from bounds
    master_transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Report coverage area (km²)
    area_km2 = (width * height * config.CELL_SIZE_M * config.CELL_SIZE_M) / 1_000_000
    print(f"\nCoverage area: {area_km2:,.2f} km²")

    # 5) Projected lookups (names uppercased to match earlier preprocess)
    print("\nCreating projected geometry lookups...")
    state_lookup_proj = dict(zip(states_proj["NAME"].str.upper(), states_proj.geometry))
    county_lookup_proj = dict(zip(counties_proj["NAME"].str.upper(), counties_proj.geometry))
    cities_lookup_proj = dict(zip(cities_proj["NAME"].str.upper(), cities_proj.geometry))

    grid_params: Dict[str, Any] = {
        "crs": config.TARGET_CRS,
        "cell_size": config.CELL_SIZE_M,
        "width": width,
        "height": height,
        "bounds": (minx, miny, maxx, maxy),
        "transform": master_transform,
        "state_lookup_proj": state_lookup_proj,
        "county_lookup_proj": county_lookup_proj,
        "cities_lookup_proj": cities_lookup_proj,
        "francine_proj": francine_proj,
        "helene_proj": helene_proj,
    }

    print("\nMaster grid canvas ready ✓")
    return grid_params


# ------------------------------------------------------------------------------
# STEP 2 — FACILITY HEAT (OPTIONAL)
# ------------------------------------------------------------------------------

def create_facility_raster(
    data: gpd.GeoDataFrame,
    grid_params: Dict[str, Any],
) -> np.ndarray:
    """
    Create a Gaussian kernel density-like raster for facility points.

    Logic
    -----
    • Sum facility counts by name.
    • For each facility point:
        - Project to TARGET_CRS.
        - Stamp a single-pixel impulse at (row, col).
        - Convolve with Gaussian (sigma in pixels).
        - Scale by FACILITY weight and accumulate.

    Parameters
    ----------
    data : GeoDataFrame
        Expanded tweet rows for a single time bin (must include 'scale_level',
        'matched_name', 'matched_geom', 'count').
    grid_params : dict
        Output of create_master_grid().

    Returns
    -------
    np.ndarray
        Float32 grid matching (height, width).

    Notes
    -----
    • CRS assumption: facility geometry is treated as EPSG:4326 before projecting.
      If your matched_geom is already in TARGET_CRS, you may wish to replace the
      GeoSeries construction with `gpd.GeoSeries([facility_point], crs=grid_params['crs']).to_crs(...)`
      or directly use the point if its `.crs` is known.
    """
    facility_grid = np.zeros((grid_params["height"], grid_params["width"]), dtype=np.float32)
    facility_data = data[data["scale_level"] == "FACILITY"]

    if len(facility_data) == 0:
        return facility_grid

    # Aggregate facility counts over all rows for this bin
    facility_counts = facility_data.groupby("matched_name")["count"].sum()

    # Gaussian parameters
    sigma_meters = 2 * grid_params["cell_size"]
    sigma_pixels = sigma_meters / grid_params["cell_size"]
    facility_multiplier = config.WEIGHTS["FACILITY"]

    facilities_processed = 0
    for facility_name, tweet_count in facility_counts.items():
        facility_rows = facility_data[facility_data["matched_name"] == facility_name]
        if len(facility_rows) == 0:
            continue

        facility_point = facility_rows.iloc[0]["matched_geom"]
        if hasattr(facility_point, "x") and hasattr(facility_point, "y"):
            # Assumes WGS84 input; reproject to TARGET_CRS
            point_geoseries = gpd.GeoSeries([facility_point], crs="EPSG:4326")
            point_proj = point_geoseries.to_crs(grid_params["crs"]).iloc[0]

            # Convert projected x/y to pixel indices (col/row)
            px = (point_proj.x - grid_params["bounds"][0]) / grid_params["cell_size"]
            py = (grid_params["bounds"][3] - point_proj.y) / grid_params["cell_size"]

            if 0 <= px < grid_params["width"] and 0 <= py < grid_params["height"]:
                point_grid = np.zeros((grid_params["height"], grid_params["width"]), dtype=np.float32)
                point_grid[int(py), int(px)] = float(tweet_count)

                kernel_grid = gaussian_filter(point_grid, sigma=sigma_pixels, mode="constant", cval=0.0)
                facility_grid += kernel_grid * facility_multiplier
                facilities_processed += 1

    return facility_grid


# ------------------------------------------------------------------------------
# STEP 3 — HIERARCHICAL RASTER
# ------------------------------------------------------------------------------

def create_hierarchical_rasters(
    data: gpd.GeoDataFrame,
    grid_params: Dict[str, Any],
    time_bin: int,
) -> np.ndarray:
    """
    Create hierarchically weighted rasters with automatic parent state inclusion.

    Strategy
    --------
    1) Determine which states to include:
        • All directly matched states.
        • Any state that contains the centroid of matched counties/cities.
    2) Rasterize STATE, COUNTY, CITY masks:
        • Multiply by log1p(tweet_count) * WEIGHTS[level].
    3) Add FACILITY kernel layer (optional).
    4) Return final float32 grid.

    Parameters
    ----------
    data : GeoDataFrame
        Interval subset (single time bin) with columns:
        ['unix_timestamp','scale_level','matched_name','matched_geom','count', ...]
    grid_params : dict
        Output from create_master_grid().
    time_bin : int
        Epoch-like bin identifier (used only for logging/custom hooks).

    Returns
    -------
    np.ndarray
        Float32 raster grid (height, width).
    """
    output_grid = np.zeros((grid_params["height"], grid_params["width"]), dtype=np.float32)
    states_to_include: set[str] = set()

    state_lookup_proj = grid_params["state_lookup_proj"]
    county_lookup_proj = grid_params["county_lookup_proj"]
    cities_lookup_proj = grid_params["cities_lookup_proj"]

    # -- Identify base states to include
    state_data = data[data["scale_level"] == "STATE"]
    if len(state_data) > 0:
        states_to_include.update(state_data["matched_name"].unique())

    county_data = data[data["scale_level"] == "COUNTY"]
    for county_name in county_data["matched_name"].unique():
        if county_name in county_lookup_proj:
            county_geom = county_lookup_proj[county_name]
            for state_name, state_geom in state_lookup_proj.items():
                if state_geom.contains(county_geom.centroid):
                    states_to_include.add(state_name)
                    break

    city_data = data[data["scale_level"] == "CITY"]
    for city_name in city_data["matched_name"].unique():
        if city_name in cities_lookup_proj:
            city_geom = cities_lookup_proj[city_name]
            for state_name, state_geom in state_lookup_proj.items():
                if state_geom.contains(city_geom.centroid):
                    states_to_include.add(state_name)
                    break

    # -- Rasterize STATES
    for state_name in states_to_include:
        if state_name in state_lookup_proj:
            state_geom = state_lookup_proj[state_name]
            mask = rasterize(
                [(state_geom, 1)],
                out_shape=(grid_params["height"], grid_params["width"]),
                transform=grid_params["transform"],
                fill=0,
                dtype=np.float32,
                all_touched=True,
            )

            # Use actual state tweet count if present; else minimal base presence
            if state_name in state_data["matched_name"].values:
                tweet_count = state_data[state_data["matched_name"] == state_name]["count"].sum()
            else:
                tweet_count = 1

            base_value = np.log1p(float(tweet_count)) * config.WEIGHTS["STATE"]
            output_grid += mask * base_value

    # -- Rasterize COUNTIES
    if len(county_data) > 0:
        county_counts = county_data.groupby("matched_name")["count"].sum()
        for county_name, tweet_count in county_counts.items():
            if county_name in county_lookup_proj:
                mask = rasterize(
                    [(county_lookup_proj[county_name], 1)],
                    out_shape=(grid_params["height"], grid_params["width"]),
                    transform=grid_params["transform"],
                    fill=0,
                    dtype=np.float32,
                    all_touched=True,
                )
                output_grid += mask * np.log1p(float(tweet_count)) * config.WEIGHTS["COUNTY"]

    # -- Rasterize CITIES
    if len(city_data) > 0:
        city_counts = city_data.groupby("matched_name")["count"].sum()
        for city_name, tweet_count in city_counts.items():
            if city_name in cities_lookup_proj:
                mask = rasterize(
                    [(cities_lookup_proj[city_name], 1)],
                    out_shape=(grid_params["height"], grid_params["width"]),
                    transform=grid_params["transform"],
                    fill=0,
                    dtype=np.float32,
                    all_touched=True,
                )
                output_grid += mask * np.log1p(float(tweet_count)) * config.WEIGHTS["CITY"]

    # -- Add FACILITIES
    facility_data = data[data["scale_level"] == "FACILITY"]
    if len(facility_data) > 0:
        output_grid += create_facility_raster(data, grid_params)

    return output_grid


# ------------------------------------------------------------------------------
# STEP 4 — WRITE GeoTIFF
# ------------------------------------------------------------------------------

def save_raster(
    grid: np.ndarray,
    output_dir: str,
    hurricane_name: str,
    time_bin: int,
    raster_type: str,
    timestamp_dict: Dict[int, "pd.Timestamp"],  # forward ref; pandas type is optional here
    grid_params: Dict[str, Any],
) -> None:
    """
    Save a single-band raster to GTiff in a type-specific subfolder.

    File layout
    -----------
    <output_dir>/<raster_type>/<hurricane>_tweets_<YYYYmmdd_HHMMSS>.tif

    Parameters
    ----------
    grid : np.ndarray
        Float32 grid to be written.
    output_dir : str
        Hurricane output root (e.g., /.../rasters_output/francine).
    hurricane_name : str
        'francine' | 'helene' (lower/upper accepted).
    time_bin : int
        Epoch-like bin identifier used to look up a pandas Timestamp.
    raster_type : str
        'increment' or 'cumulative' (subfolder name).
    timestamp_dict : dict
        Maps time_bin -> pandas Timestamp.
    grid_params : dict
        From create_master_grid().

    Returns
    -------
    None
    """
    type_dir = os.path.join(output_dir, raster_type)
    os.makedirs(type_dir, exist_ok=True)

    time_str = timestamp_dict[time_bin].strftime("%Y%m%d_%H%M%S")
    filename = f"{hurricane_name}_tweets_{time_str}.tif"
    filepath = os.path.join(type_dir, filename)

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=grid_params["height"],
        width=grid_params["width"],
        count=1,
        dtype=grid.dtype,
        crs=grid_params["crs"],
        transform=grid_params["transform"],
        compress="lzw",
    ) as dst:
        dst.write(grid, 1)

    print(f"    Saved: {raster_type}/{filename}")


# ------------------------------------------------------------------------------
# STEP 5 — PROCESS HURRICANE
# ------------------------------------------------------------------------------

def process_hurricane(
    hurricane_name: str,
    gdf_proj: gpd.GeoDataFrame,  # kept for signature parity; not used internally
    interval_counts: "pd.DataFrame",
    time_bins: list[int],
    timestamp_dict: Dict[int, "pd.Timestamp"],
    grid_params: Dict[str, Any],
) -> str:
    """
    Process a single hurricane over all time bins and write rasters.

    Workflow
    --------
    For each bin:
        1) Filter interval_counts to current bin.
        2) Build incremental raster via create_hierarchical_rasters().
        3) Accumulate into cumulative grid.
        4) Save both 'increment' and 'cumulative' GeoTIFFs.

    Parameters
    ----------
    hurricane_name : str
        Name used for folder and filename labels.
    gdf_proj : GeoDataFrame
        Projected hurricane features (unused here; retained for API symmetry).
    interval_counts : DataFrame
        Output from geographic_matching.create_interval_counts().
    time_bins : List[int]
        Sorted unix-like bin identifiers.
    timestamp_dict : Dict[int, Timestamp]
        Maps bin id to pandas Timestamp (for filenames).
    grid_params : dict
        From create_master_grid().

    Returns
    -------
    str
        Path to the hurricane's output directory.
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {hurricane_name.upper()}")
    print(f"{'=' * 60}")

    hurricane_dir = os.path.join(config.OUTPUT_DIR, hurricane_name.lower())
    os.makedirs(hurricane_dir, exist_ok=True)

    cumulative_grid = np.zeros((grid_params["height"], grid_params["width"]), dtype=np.float32)

    for idx, time_bin in enumerate(time_bins):
        print(f"\nTime Bin {idx + 1}/{len(time_bins)}")

        current_data = interval_counts[interval_counts["unix_timestamp"] == time_bin]
        tweet_count = len(current_data)
        print(f"  Tweets in this bin: {tweet_count}")

        incremental_grid = create_hierarchical_rasters(current_data, grid_params, time_bin)
        cumulative_grid += incremental_grid

        save_raster(
            incremental_grid,
            hurricane_dir,
            hurricane_name,
            time_bin,
            "increment",
            timestamp_dict,
            grid_params,
        )
        save_raster(
            cumulative_grid,
            hurricane_dir,
            hurricane_name,
            time_bin,
            "cumulative",
            timestamp_dict,
            grid_params,
        )

        print(f"  Incremental max value: {np.max(incremental_grid):.2f}")
        print(f"  Cumulative  max value: {np.max(cumulative_grid):.2f}")

    print(f"\n{hurricane_name.upper()} processing complete!")
    return hurricane_dir
