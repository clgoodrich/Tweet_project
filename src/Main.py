"""
Tweet Project — Main Orchestration (Step-by-Step)
================================================

This script *coordinates* the entire pipeline, delegating actual data work to
project modules (`config`, `data_loader`, `geographic_matching`, `rasterization`).

OVERVIEW (End-to-End Steps)
---------------------------
STEP 0 — IMPORTS & ENV
    • Import stdlib modules, suppress noisy warnings.
    • Import project modules (must be importable on PYTHONPATH).

STEP 1 — LOAD HURRICANE DATA
    • data_loader.load_hurricane_data()
        -> Returns two GeoDataFrames: francine_gdf, helene_gdf
        EXPECTED: Each has valid geometry and a time field (e.g., 'time' or 'time_str').

STEP 2 — BUILD TIME LOOKUPS
    • data_loader.create_timestamp_dictionaries(francine_gdf, helene_gdf)
        -> Returns two dicts mapping time bins/strings to fast-lookup structures.
        PURPOSE: Speed up later aggregation/raster loops.

STEP 3 — LOAD REFERENCE SHAPEFILES
    • data_loader.load_reference_shapefiles()
        -> Returns states_gdf, counties_gdf, cities_gdf
        EXPECTED: Valid polygons, consistent CRS or convertible CRS.

STEP 4 — HIERARCHICAL MATCH LOOKUPS
    • geographic_matching.create_hierarchical_lookups(states, counties, cities)
        -> Returns 'lookups' object with fast join/index structures (state→county→city).

STEP 5 — EXPAND TWEETS BY MATCHES (MULTI-LEVEL)
    • geographic_matching.expand_tweets_by_matches(francine_gdf, lookups, "FRANCINE")
    • geographic_matching.expand_tweets_by_matches(helene_gdf, lookups, "HELENE")
        -> Returns expanded GeoDataFrames with derived matches at multiple admin levels.

STEP 6 — EXPORT PRE-MATCHED GEOJSON SNAPSHOTS
    • export_matched_geojsons.export_matched_geojsons(francine_expanded, helene_expanded)
        -> Writes GeoJSON files containing matched geometries for reuse in future runs.

STEP 7 — INTERVAL COUNTS (PER-TIME-BIN)
    • geographic_matching.create_interval_counts(francine_gdf)
    • geographic_matching.create_interval_counts(helene_gdf)
        -> Returns tidy per-bin counts used to drive rasterization.

STEP 8 — COLLECT ORDERED TIME BINS
    • data_loader.get_time_bins(francine_gdf), get_time_bins(helene_gdf)
        -> Returns ordered list/sequence of bins that define raster loop iterations.

STEP 9 — BUILD MASTER GRID
    • rasterization.create_master_grid(francine_gdf, helene_gdf, states, counties, cities)
        -> Returns grid_params dict with transform, width, height, bounds, and
           per-event target projections (e.g., 'francine_proj', 'helene_proj').

STEP 10 — ENSURE OUTPUT ROOT
    • os.makedirs(config.OUTPUT_DIR, exist_ok=True)

STEP 11 — PROCESS HURRICANE: FRANCINE
    • rasterization.process_hurricane(
          event_name='francine',
          target_proj=grid_params['francine_proj'],
          interval_counts=francine_interval_counts,
          time_bins=francine_time_bins,
          timestamp_dict=francine_dict,
          grid_params=grid_params
      )
        -> Returns path to Francine output directory with incremental & cumulative rasters.

STEP 12 — PROCESS HURRICANE: HELENE
    • Same as STEP 11 for 'helene' using helene_* inputs.

STEP 13 — SUMMARY & NEXT STEPS
    • Print output dirs, raster types created, and ArcGIS Pro next steps.

RETURNS
-------
Tuple[str, str] = (francine_output_dir, helene_output_dir)

PRECONDITIONS
-------------
• Project modules resolve and expose the required functions.
• Input GeoDataFrames have valid geometry + time fields.
• CRS issues are handled inside data_loader / rasterization as designed.

POSTCONDITIONS
--------------
• Two folders with time-sliced ("increment") and cumulative rasters per hurricane.

TYPICAL USE
-----------
    python main.py

TROUBLESHOOTING
---------------
• If no rasters: check that expansion/matching produced rows (STEP 5)
• If time slider empty: ensure mosaic item has valid time field & is time-enabled
• If CRS mismatch errors: verify create_master_grid() target projections exist

"""

from __future__ import annotations

# STEP 0 — IMPORTS & ENV
import os
import warnings
from typing import Tuple

warnings.filterwarnings('ignore')

# Project modules (must be discoverable on PYTHONPATH)
import sys
import os
# Add src to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))

try:
    from . import config
    from . import data_loader
    from . import geographic_matching
    from . import rasterization
    from . import export_matched_geojsons
except ImportError:
    # Fallback for direct execution
    import config
    import data_loader
    import geographic_matching
    import rasterization
    import export_matched_geojsons


def main() -> Tuple[str, str]:
    """
    Execute the pipeline with explicit, numbered phases (mirrors top docstring).

    Returns
    -------
    (francine_output_dir, helene_output_dir) : Tuple[str, str]
        Root output directories for raster products per hurricane.

    Raises
    ------
    Exception
        Any exception is printed in the __main__ guard and re-raised for visibility.
    """

    # Banner
    print("\n" + "=" * 80)
    print("TWEET PROJECT - HURRICANE SOCIAL MEDIA ANALYSIS")
    print("=" * 80)

    # STEP 1 — LOAD HURRICANE DATA
    print("\n[STEP 1/13] Loading Hurricane Data")
    print("-" * 40)
    francine_gdf, helene_gdf = data_loader.load_hurricane_data()

    # STEP 2 — BUILD TIME LOOKUPS (fast timestamp dictionaries)
    print("\n[STEP 2/13] Building Time Lookups")
    print("-" * 40)
    francine_dict, helene_dict = data_loader.create_timestamp_dictionaries(
        francine_gdf, helene_gdf
    )

    # STEP 3 — LOAD REFERENCE SHAPEFILES
    print("\n[STEP 3/13] Loading Reference Shapefiles")
    print("-" * 40)
    states_gdf, counties_gdf, cities_gdf = data_loader.load_reference_shapefiles()

    # STEP 4 — HIERARCHICAL MATCH LOOKUPS
    print("\n[STEP 4/13] Creating Hierarchical Geographic Lookups")
    print("-" * 40)
    lookups = geographic_matching.create_hierarchical_lookups(
        states_gdf, counties_gdf, cities_gdf
    )

    # STEP 5 — EXPAND TWEETS BY MATCHES (MULTI-LEVEL)
    print("\n[STEP 5/13] Expanding Tweets by Multi-Level Matches")
    print("-" * 40)
    francine_gdf = geographic_matching.expand_tweets_by_matches(
        francine_gdf, lookups, "FRANCINE"
    )
    helene_gdf = geographic_matching.expand_tweets_by_matches(
        helene_gdf, lookups, "HELENE"
    )

    # STEP 6 — EXPORT PRE-MATCHED GEOJSONS
    print("\n[STEP 6/13] Exporting Pre-Matched GeoJSONs")
    print("-" * 40)
    matched_exports = export_matched_geojsons.export_matched_geojsons(
        francine_expanded=francine_gdf,
        helene_expanded=helene_gdf,
    )
    for dataset, path in matched_exports.items():
        print(f"  {dataset.title()} matched GeoJSON: {path}")

    # STEP 7 — INTERVAL COUNTS (PER-TIME-BIN)
    print("\n[STEP 7/13] Creating Interval Counts")
    print("-" * 40)
    francine_interval_counts = geographic_matching.create_interval_counts(francine_gdf)
    helene_interval_counts = geographic_matching.create_interval_counts(helene_gdf)

    # STEP 8 — COLLECT ORDERED TIME BINS
    print("\n[STEP 8/13] Collecting Ordered Time Bins")
    print("-" * 40)
    francine_time_bins = data_loader.get_time_bins(francine_gdf)
    helene_time_bins = data_loader.get_time_bins(helene_gdf)
    print(f"  Francine time bins: {len(francine_time_bins)}")
    print(f"  Helene   time bins: {len(helene_time_bins)}")

    # STEP 9 — BUILD MASTER GRID (transform, size, bounds, per-event projections)
    print("\n[STEP 9/13] Creating Master Grid")
    print("-" * 40)
    grid_params = rasterization.create_master_grid(
        francine_gdf, helene_gdf, states_gdf, counties_gdf, cities_gdf
    )
    # Expected keys (typical):
    #   grid_params['transform'], ['width'], ['height'], ['bounds'],
    #   grid_params['francine_proj'], grid_params['helene_proj'], ...
    #   plus any additional metadata required by process_hurricane()

    # STEP 10 — ENSURE OUTPUT ROOT DIR
    print("\n[STEP 10/13] Ensuring Output Directory Exists")
    print("-" * 40)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"  OUTPUT_DIR: {config.OUTPUT_DIR}")

    # STEP 11 — PROCESS HURRICANE: FRANCINE (increment + cumulative rasters)
    print("\n[STEP 11/13] Processing Hurricane: FRANCINE")
    print("-" * 40)
    francine_output = rasterization.process_hurricane(
        hurricane_name='francine',
        gdf_proj=grid_params['francine_proj'],
        interval_counts=francine_interval_counts,
        time_bins=francine_time_bins,
        timestamp_dict=francine_dict,
        grid_params=grid_params,
    )
    print(f"  Francine output dir: {francine_output}")

    # STEP 12 — PROCESS HURRICANE: HELENE (increment + cumulative rasters)
    print("\n[STEP 12/13] Processing Hurricane: HELENE")
    print("-" * 40)
    helene_output = rasterization.process_hurricane(
        hurricane_name='helene',
        gdf_proj=grid_params['helene_proj'],
        interval_counts=helene_interval_counts,
        time_bins=helene_time_bins,
        timestamp_dict=helene_dict,
        grid_params=grid_params,
    )
    print(f"  Helene output dir: {helene_output}")

    # STEP 13 — SUMMARY & NEXT STEPS
    print("\n[STEP 13/13] Summary & Next Steps")
    print("-" * 40)
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nOutput directories:")
    print(f"  Francine: {francine_output}")
    print(f"  Helene:   {helene_output}")
    print(f"\nRaster types created:")
    print(f"  - increment : Tweet activity per time bin")
    print(f"  - cumulative: Accumulated tweet activity over time")
    print(f"\nNext steps:")
    print(f"  1. Run arcgis_mosaic.py in ArcGIS Pro Python environment")
    print(f"  2. Add mosaic datasets to ArcGIS Pro map")
    print(f"  3. Configure symbology and time slider")
    print(f"  4. Export animations as needed")

    return francine_output, helene_output


if __name__ == "__main__":
    try:
        # Execute full pipeline with explicit phase logging
        francine_dir, helene_dir = main()
        print("\n[SUCCESS] Pipeline execution successful!")
    except Exception as e:
        # Keep loud + re-raise for CI/ops visibility
        print(f"\n✗ Pipeline failed with error: {str(e)}")
        raise
