"""
Tweet Project — Data Loading & Preprocessing (Step-by-Step)
==========================================================

This module loads hurricane GeoJSON files and reference shapefiles, then
standardizes/organizes time fields for downstream aggregation and rasterization.

OVERVIEW (End-to-End Steps)
---------------------------
STEP 1 — LOAD HURRICANE GEOJSONS
    • Read Francine and Helene tweet data into GeoDataFrames.

STEP 2 — STANDARDIZE TIMESTAMPS
    • Convert 'time' to UTC-aware pandas datetimes in column 'timestamp'.

STEP 3 — ASSIGN TIME BINS
    • Floor each timestamp to the configured bin width (e.g., 4 hours) as 'time_bin'.

STEP 4 — DERIVE UNIX-LIKE NUMERIC KEYS
    • Convert 'time_bin' to an integer epoch-like value in 'unix_timestamp'
      (see Notes on units).

STEP 5 — BUILD LABELS
    • Create compact, human-readable bin labels (YYYYMMDD_HHMM) in 'bin_label'.

STEP 6 — LOAD REFERENCE SHAPEFILES
    • Read states, counties, and cities shapefiles into GeoDataFrames.

STEP 7 — TIMESTAMP DICTIONARIES
    • Build fast lookup dicts: unix_timestamp → time_bin for each hurricane.

STEP 8 — UNIQUE, SORTED TIME BINS
    • Return sorted unique unix_timestamp values for ordered processing loops.

Notes & Pitfalls
----------------
• Input column requirement: this module expects a 'time' column in the hurricane
  GeoJSONs. If it’s named differently, adapt `load_hurricane_data()` accordingly.
• Timezone handling: timestamps are parsed with `utc=True` to standardize. If your
  raw 'time' strings include TZ offsets, pandas will normalize to UTC.
• Epoch units: `pandas.Timestamp.astype('int64')` yields **nanoseconds** since epoch.
  Dividing by 1000 (as done here) converts to **microseconds** (not seconds).
  - Keep as-is if your pipeline assumes higher precision bins.
  - If you need **seconds**, divide by 1_000_000_000 instead.
• CRS of reference layers: this module doesn’t reproject. Downstream components
  (e.g., rasterization) should handle target CRS.

"""

from __future__ import annotations

# STEP 0 — IMPORTS
import geopandas as gpd
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Tuple

try:
    from . import config
except ImportError:
    import config


# ------------------------------------------------------------------------------
# STEP 1–5 — HURRICANE DATA LOADING & TIME PREP
# ------------------------------------------------------------------------------

def load_hurricane_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load hurricane GeoJSON files and standardize timestamps, time bins, and labels.

    Returns
    -------
    (francine_gdf, helene_gdf) : Tuple[GeoDataFrame, GeoDataFrame]
        Each GeoDataFrame contains:
            - 'timestamp'      : UTC-aware pandas datetime64[ns, UTC]
            - 'time_bin'       : bin-aligned datetime (floored to TIME_BIN_HOURS)
            - 'unix_timestamp' : integer epoch-like value derived from 'time_bin'
                                 (see Notes on units in module docstring)
            - 'bin_label'      : string label (YYYYMMDD_%H%M) for filenames/keys

    Side Effects
    ------------
    Prints progress banners and counts for quick diagnostics.

    Raises
    ------
    Any exception from file I/O or parsing bubbles up to caller.
    """
    print("=" * 60)
    print("LOADING HURRICANE DATA")
    print("=" * 60)

    # STEP 1 — LOAD GEOJSONS
    print(f"\nLoading Francine data from: {config.FRANCINE_PATH}")
    francine_gdf = gpd.read_file(config.FRANCINE_PATH)

    print(f"Loading Helene data from: {config.HELENE_PATH}")
    helene_gdf = gpd.read_file(config.HELENE_PATH)

    # STEP 2 — STANDARDIZE TIMESTAMPS (UTC)
    print("\nStandardizing timestamps to UTC...")
    francine_gdf["timestamp"] = pd.to_datetime(francine_gdf["time"], utc=True)
    helene_gdf["timestamp"] = pd.to_datetime(helene_gdf["time"], utc=True)

    # STEP 3 — ASSIGN TIME BINS
    time_bin_str = f"{config.TIME_BIN_HOURS}h"
    print(f"Grouping data into {config.TIME_BIN_HOURS}-hour bins...")
    francine_gdf["time_bin"] = francine_gdf["timestamp"].dt.floor(time_bin_str)
    helene_gdf["time_bin"] = helene_gdf["timestamp"].dt.floor(time_bin_str)

    # STEP 4 — DERIVE UNIX-LIKE NUMERIC KEYS
    # NOTE: .astype('int64') on datetime64[ns, UTC] -> nanoseconds since epoch.
    # Dividing by 1000 yields microseconds (NOT seconds). Keep as-is for fidelity.
    francine_gdf["unix_timestamp"] = francine_gdf["time_bin"].astype("int64") // 1000
    helene_gdf["unix_timestamp"] = helene_gdf["time_bin"].astype("int64") // 1000

    # STEP 5 — HUMAN-READABLE BIN LABELS
    francine_gdf["bin_label"] = francine_gdf["time_bin"].dt.strftime("%Y%m%d_%H%M")
    helene_gdf["bin_label"] = helene_gdf["time_bin"].dt.strftime("%Y%m%d_%H%M")

    print(f"\nLoaded {len(francine_gdf)} Francine tweets")
    print(f"Loaded {len(helene_gdf)} Helene tweets")

    return francine_gdf, helene_gdf


# ------------------------------------------------------------------------------
# STEP 6 — REFERENCE SHAPEFILES
# ------------------------------------------------------------------------------

def load_reference_shapefiles() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load reference shapefiles for states, counties, and cities.

    Returns
    -------
    (states_gdf, counties_gdf, cities_gdf) : Tuple[GeoDataFrame, GeoDataFrame, GeoDataFrame]
        Each GeoDataFrame contains polygon/point geometries and associated attributes.

    Side Effects
    ------------
    Prints counts of loaded features for quick sanity checks.
    """
    print("\nLoading reference shapefiles...")

    states_gdf = gpd.read_file(config.STATES_PATH)
    print(f"  Loaded {len(states_gdf)} states")

    counties_gdf = gpd.read_file(config.COUNTIES_PATH)
    print(f"  Loaded {len(counties_gdf)} counties")

    cities_gdf = gpd.read_file(config.CITIES_PATH)
    print(f"  Loaded {len(cities_gdf)} cities")

    return states_gdf, counties_gdf, cities_gdf


# ------------------------------------------------------------------------------
# STEP 7 — TIMESTAMP DICTIONARIES
# ------------------------------------------------------------------------------

def create_timestamp_dictionaries(
    francine_gdf: gpd.GeoDataFrame, helene_gdf: gpd.GeoDataFrame
) -> Tuple[Dict[int, pd.Timestamp], Dict[int, pd.Timestamp]]:
    """
    Create lookup dictionaries mapping unix_timestamp -> time_bin.

    Parameters
    ----------
    francine_gdf : GeoDataFrame
        Hurricane Francine tweets with 'unix_timestamp' and 'time_bin'.
    helene_gdf : GeoDataFrame
        Hurricane Helene tweets with 'unix_timestamp' and 'time_bin'.

    Returns
    -------
    (francine_dict, helene_dict) : Tuple[Dict[int, Timestamp], Dict[int, Timestamp]]
        Dicts keyed by integer unix-like keys pointing to pandas Timestamps (time_bin).

    Notes
    -----
    - If 'unix_timestamp' is microseconds (as produced here), keys will be large ints.
    - Downstream callers should treat these as opaque bin IDs, not human times.
    """
    francine_dict = dict(zip(francine_gdf["unix_timestamp"], francine_gdf["time_bin"]))
    helene_dict = dict(zip(helene_gdf["unix_timestamp"], helene_gdf["time_bin"]))
    return francine_dict, helene_dict


# ------------------------------------------------------------------------------
# STEP 8 — UNIQUE, SORTED TIME BINS
# ------------------------------------------------------------------------------

def get_time_bins(gdf: gpd.GeoDataFrame) -> List[int]:
    """
    Get sorted unique unix_timestamp bins from a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain a 'unix_timestamp' column (integer epoch-like values).

    Returns
    -------
    List[int]
        Sorted list of unique unix_timestamp values for ordered iteration.

    Notes
    -----
    - Treat return values as **bin identifiers**. If you require readable labels,
      pair with 'bin_label', or map through the dicts from create_timestamp_dictionaries().
    """
    return sorted(gdf["unix_timestamp"].unique())
