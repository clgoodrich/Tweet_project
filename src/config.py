"""
Tweet Project — Configuration (Step-by-Step)
===========================================

This module centralizes **all constants, file paths, and knobs** used across the
pipeline. It is read-only at runtime; other modules should import from here
instead of hard-coding values.

OVERVIEW (End-to-End Steps)
---------------------------
STEP 0 — IMPORTS
    • Use stdlib `os` for cross-platform path handling.

STEP 1 — BASE PATH DISCOVERY
    • LOCAL_PATH is derived from current working directory’s parent.
      Assumes you launch scripts from the project root or a subfolder.

STEP 2 — CORE DATA DIRECTORIES
    • Define canonical folders for inputs and outputs (data/, geojson/, etc.).

STEP 3 — INPUT FILE PATHS
    • Point to hurricane GeoJSONs and reference shapefiles.

STEP 4 — RASTER/PROJECTION SETTINGS
    • Set target CRS and cell size (meters in projected CRS).

STEP 5 — HIERARCHICAL WEIGHTS
    • Control relative influence of matches at each admin level during
      rasterization (STATE/COUNTY/CITY/FACILITY).

STEP 6 — FUZZY MATCHING & TIME BINNING
    • Thresholds for name matching; temporal bin width for aggregation.

Directory Layout (expected)
---------------------------
<project_root>/
├─ data/
│  ├─ geojson/
│  │  ├─ francine.geojson
│  │  └─ helene.geojson
│  └─ shape_files/
│     ├─ cb_2023_us_state_20m.shp
│     ├─ cb_2023_us_county_20m.shp
│     └─ US_Cities.shp
└─ rasters_output/            (created at runtime if missing)

Notes & Pitfalls
----------------
• LOCAL_PATH is computed as the **parent of the CWD**. If you run from unusual
  locations (e.g., a nested script folder), path resolution may shift. Launch
  from project root to avoid surprises, or override via env vars (see below).
• EPSG:3857 units are meters, but area/scale distort with latitude. CELL_SIZE_M
  represents grid resolution in projected meters; choose with your AOI in mind.
• Weights (WEIGHTS) are relative, not absolute. Doubling all values yields the
  same ratios; only their **proportions** matter.

Environment Overrides (optional)
--------------------------------
You may override paths/values by exporting env vars before running Python, e.g.:
    export TWEET_LOCAL_PATH="/abs/project/root"
    export TWEET_OUTPUT_DIR="/abs/outputs"
Add such logic here only if desired; current config keeps behavior static.
"""

from __future__ import annotations

# STEP 0 — IMPORTS
import os
from typing import Dict

# ------------------------------------------------------------------------------
# STEP 1 — BASE PATH DISCOVERY
# ------------------------------------------------------------------------------

# LOCAL_PATH: Project root directory (parent of src/).
# Using __file__ to find config.py location, then go up to project root.
# This works regardless of where the script is run from.
LOCAL_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------------------
# STEP 2 — CORE DATA DIRECTORIES
# ------------------------------------------------------------------------------

# DATA_DIR: Root folder for all input datasets used by the pipeline.
DATA_DIR: str = os.path.join(LOCAL_PATH, "data")

# GEOJSON_DIR: Holds hurricane-specific tweet GeoJSONs.
GEOJSON_DIR: str = os.path.join(DATA_DIR, "geojson")

# SHAPE_FILES_DIR: Holds boundary/reference shapefiles (states/counties/cities).
SHAPE_FILES_DIR: str = os.path.join(DATA_DIR, "shape_files")

# OUTPUT_DIR: Root folder where all raster products will be written.
OUTPUT_DIR: str = os.path.join(LOCAL_PATH, "rasters_output")

# ------------------------------------------------------------------------------
# STEP 3 — INPUT FILE PATHS
# ------------------------------------------------------------------------------

# Hurricane tweet datasets (GeoJSON)
FRANCINE_PATH: str = os.path.join(GEOJSON_DIR, "francine.geojson")
HELENE_PATH:   str = os.path.join(GEOJSON_DIR, "helene.geojson")

# Reference boundaries (Shapefiles; ensure accompanying .dbf/.shx/.prj exist)
STATES_PATH:   str = os.path.join(SHAPE_FILES_DIR, "cb_2023_us_state_20m.shp")
COUNTIES_PATH: str = os.path.join(SHAPE_FILES_DIR, "cb_2023_us_county_20m.shp")
CITIES_PATH:   str = os.path.join(SHAPE_FILES_DIR, "US_Cities.shp")

# ------------------------------------------------------------------------------
# STEP 4 — RASTER / PROJECTION SETTINGS
# ------------------------------------------------------------------------------

# TARGET_CRS: Output grid projection (Web Mercator).
# Units are meters; suitable for web mapping, with known high-latitude distortion.
TARGET_CRS: str = "EPSG:3857"

# CELL_SIZE_M: Pixel size (meters) in TARGET_CRS.
# Typical values: 250–2000 m. Smaller = higher detail + more files.
CELL_SIZE_M: int = 1000

# ------------------------------------------------------------------------------
# STEP 5 — HIERARCHICAL WEIGHTS
# ------------------------------------------------------------------------------

# WEIGHTS: Relative influence when aggregating tweet signals by match level.
# Example interpretation: CITY hits weigh more than COUNTY, which weigh more than STATE.
# FACILITY is provided for parity with possible facility-level matches elsewhere.
WEIGHTS: Dict[str, int] = {
    "STATE": 2,
    "COUNTY": 5,
    "CITY": 10,
    "FACILITY": 10,
}

# CITY_KERNEL_SIGMA_PIXELS: Controls Gaussian smoothing for city kernel density.
CITY_KERNEL_SIGMA_PIXELS: int = 3

# ------------------------------------------------------------------------------
# STEP 6 — FUZZY MATCHING & TIME BINNING
# ------------------------------------------------------------------------------

# FUZZY_THRESHOLD: Minimum score for a name match to be accepted (0–100).
# Higher = stricter (fewer false positives, more false negatives).
FUZZY_THRESHOLD: int = 75

# FUZZY_THRESHOLD_CONTEXTUAL: Lower threshold allowed when context supports the match
# (e.g., within already-matched county/state). Provides recall without too many spurious hits.
FUZZY_THRESHOLD_CONTEXTUAL: int = 70

# TIME_BIN_HOURS: Width (in hours) of temporal aggregation bins.
# Affects number of rasters and time slider smoothness.
TIME_BIN_HOURS: int = 4
