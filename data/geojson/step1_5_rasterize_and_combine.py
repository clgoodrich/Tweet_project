"""
STEPS 1-5: Rasterize shapefiles + combine into time-enabled GeoPackage
=====================================================================

This script automates:
  1. Load shapefiles (states, counties, cities)
  2. For each bin in counts_output/:
     - Join counts CSV to shapefile
     - Rasterize states & counties
     - Create KDE heatmap for cities
     - Combine with weights: (state*0.1) + (county*0.5) + (city*1.0)
     - Save iterative & cumulative rasters
  3. Package all rasters into GeoPackage
  4. Create temporal metadata index

Dependencies:
  pip install geopandas rasterio rasterio-rio gdal geopandas pandas numpy

Input:
  - states.shp, counties.shp, cities.shp (with NAME, STATEFP fields)
  - counts_output/*.csv (from Step 0)

Output:
  - rasters_output/*.tif (iterative & cumulative per bin)
  - hurricane_timeseries.gpkg (all rasters packaged with metadata)
  - temporal_index.json (for QGIS temporal controller)
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings

# Raster tools
try:
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.mask import mask as rio_mask
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine
    from rasterio.warp import calculate_default_transform, reproject
    from scipy.ndimage import gaussian_filter
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    exit(1)

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print("Warning: GDAL not available. Some operations may be slower.")

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION (Edit these)
# ============================================================================

# Shapefile paths
STATES_SHP = "./cb_2023_us_state_20m.shp"
COUNTIES_SHP = "./cb_2023_us_county_20m.shp"
CITIES_SHP = "./US_Cities.shp"

# Shapefile field names (adjust to your schema)
STATES_NAME_FIELD = "NAME"
COUNTIES_NAME_FIELD = "NAME"
COUNTIES_STATE_FIELD = "STATEFP"
CITIES_NAME_FIELD = "NAME"
CITIES_STATE_FIELD = "STATEFP"

# Input/output directories
COUNTS_INPUT_DIR = "./counts_output"
RASTERS_OUTPUT_DIR = "./rasters_output"
GPKG_OUTPUT = "./hurricane_timeseries.gpkg"

# Rasterization parameters
OUTPUT_CRS = "EPSG:5070"  # NAD83 Conus Albers
CELL_SIZE = 5000  # meters
KDE_RADIUS = 50000  # kernel density search radius (meters)

# Weights for combining rasters
WEIGHTS = {
    "state": 0.1,
    "county": 0.5,
    "city": 1.0
}

# ============================================================================
# SETUP
# ============================================================================

os.makedirs(RASTERS_OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {RASTERS_OUTPUT_DIR}")

# Check inputs
for path in [STATES_SHP, COUNTIES_SHP, CITIES_SHP]:
    if not Path(path).exists():
        print(f"ERROR: {path} not found!")
        exit(1)

if not Path(COUNTS_INPUT_DIR).exists():
    print(f"ERROR: {COUNTS_INPUT_DIR} not found! Run Step 0 first.")
    exit(1)

# ============================================================================
# UTILITIES
# ============================================================================

def load_shapefiles():
    """Load and reproject shapefiles to output CRS."""
    print("\n[*] Loading shapefiles...")
    
    states = gpd.read_file(STATES_SHP)
    counties = gpd.read_file(COUNTIES_SHP)
    cities = gpd.read_file(CITIES_SHP)
    
    print(f"  ✓ States: {len(states)} features")
    print(f"  ✓ Counties: {len(counties)} features")
    print(f"  ✓ Cities: {len(cities)} features")
    
    # Reproject to output CRS
    states = states.to_crs(OUTPUT_CRS)
    counties = counties.to_crs(OUTPUT_CRS)
    cities = cities.to_crs(OUTPUT_CRS)
    
    return states, counties, cities

def load_count_csvs():
    """
    Load all count CSVs from counts_output/.
    
    Returns: dict of (event, bin_start, bin_end) → DataFrame with GPE, count
    """
    print(f"\n[*] Loading count CSVs from {COUNTS_INPUT_DIR}...")
    
    bins = {}
    count_files = sorted(Path(COUNTS_INPUT_DIR).glob("counts_*.csv"))
    
    for csv_path in count_files:
        df = pd.read_csv(csv_path)
        
        # Extract key from filename (counts_EVENT_YYYYMMDD_HHMM.csv)
        # or from columns if present
        if "event" in df.columns and "bin_start" in df.columns:
            for _, row in df.iterrows():
                event = row["event"]
                bin_start = pd.to_datetime(row["bin_start"])
                bin_end = pd.to_datetime(row["bin_end"])
                
                key = (event, bin_start, bin_end)
                
                # Store GPE → count mapping
                gpe_counts = pd.DataFrame({
                    "GPE": df[df["event"] == event]["GPE"],
                    "count": df[df["event"] == event]["count"]
                })
                
                bins[key] = gpe_counts
        else:
            print(f"  Warning: {csv_path.name} missing expected columns, skipping")
    
    print(f"  ✓ Loaded {len(bins)} bins")
    return bins

def join_counts_to_geometry(gdf, counts_df, name_field, state_field=None):
    """
    Join count CSV to a GeoDataFrame by name matching.
    
    Args:
      gdf: GeoDataFrame (states, counties, or cities)
      counts_df: DataFrame with columns [GPE, count]
      name_field: field in gdf to join on
      state_field: optional field for state-level matching (for counties/cities)
    
    Returns: GeoDataFrame with joined 'count' column
    """
    gdf_copy = gdf.copy()
    gdf_copy["count"] = 0  # Initialize to 0
    
    for idx, row in gdf_copy.iterrows():
        geom_name = str(row[name_field]).strip().upper()
        
        # Exact match on name
        matches = counts_df[counts_df["GPE"].str.upper() == geom_name]
        
        if len(matches) > 0:
            gdf_copy.loc[idx, "count"] = matches["count"].sum()
    
    return gdf_copy[["count", "geometry"]]

def create_transform(bounds, width, height):
    """Create rasterio Affine transform from bounds and dimensions."""
    minx, miny, maxx, maxy = bounds
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height
    transform = Affine.translation(minx, maxy) * Affine.scale(pixel_width, -pixel_height)
    return transform

def rasterize_layer(gdf, output_path, cell_size=CELL_SIZE, extent_bounds=None):
    """
    Rasterize a polygon GeoDataFrame with a 'count' field.
    
    Output: single-band GeoTIFF with float32 values
    
    Args:
      extent_bounds: optional fixed bounds tuple (minx, miny, maxx, maxy) 
                     to use instead of gdf bounds (for alignment)
    """
    if len(gdf) == 0 or gdf["count"].sum() == 0:
        print(f"  Warning: {output_path} has no counts, skipping")
        return None
    
    # Get bounds (in output CRS)
    if extent_bounds is not None:
        bounds = extent_bounds
    else:
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Calculate raster dimensions
    width = int(np.ceil((bounds[2] - bounds[0]) / cell_size))
    height = int(np.ceil((bounds[3] - bounds[1]) / cell_size))
    
    # Create transform
    transform = create_transform(bounds, width, height)
    
    # Rasterize: features → raster with count values
    shapes = [
        (geom, val) for geom, val in zip(gdf.geometry, gdf["count"])
        if val > 0
    ]
    
    if not shapes:
        print(f"  Warning: {output_path} has no valid shapes, skipping")
        return None
    
    raster = rio_rasterize(shapes, out_shape=(height, width), transform=transform)
    
    # Save
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs=OUTPUT_CRS,
        transform=transform,
        compress="deflate",
        TILED="YES",
        BLOCKXSIZE=256,
        BLOCKYSIZE=256
    ) as dst:
        dst.write(raster.astype(rasterio.float32), 1)
    
    print(f"  ✓ {Path(output_path).name} ({width}×{height})")
    return output_path

def create_kde_from_points(cities_gdf, output_path, cell_size=CELL_SIZE, radius=KDE_RADIUS, extent_bounds=None):
    """
    Create Kernel Density Estimate (KDE) from point GeoDataFrame.
    Uses Gaussian kernel with 'count' as weight.
    
    If geometries are polygons, converts to centroids first.
    
    Output: single-band GeoTIFF with float32 density values
    
    Args:
      extent_bounds: optional fixed bounds tuple (minx, miny, maxx, maxy) 
                     to use instead of gdf bounds (for alignment)
    """
    if len(cities_gdf) == 0 or cities_gdf["count"].sum() == 0:
        print(f"  Warning: {output_path} has no counts, skipping")
        return None
    
    # Convert polygons to points (centroids) if needed
    cities_gdf_copy = cities_gdf.copy()
    if cities_gdf_copy.geometry.geom_type[0] in ["Polygon", "MultiPolygon"]:
        cities_gdf_copy["geometry"] = cities_gdf_copy.geometry.centroid
    
    # Get bounds
    if extent_bounds is not None:
        bounds = extent_bounds
    else:
        bounds = cities_gdf_copy.total_bounds
    
    # Calculate dimensions
    width = int(np.ceil((bounds[2] - bounds[0]) / cell_size))
    height = int(np.ceil((bounds[3] - bounds[1]) / cell_size))
    
    # Create transform
    transform = create_transform(bounds, width, height)
    
    # Initialize raster
    raster = np.zeros((height, width), dtype=np.float32)
    
    # For each point, add Gaussian bump
    for idx, row in cities_gdf_copy.iterrows():
        if row["count"] == 0:
            continue
        
        x, y = row.geometry.x, row.geometry.y
        
        # Convert to pixel coords
        col = int((x - bounds[0]) / cell_size)
        row_idx = int((bounds[3] - y) / cell_size)
        
        # Gaussian kernel (simplified: add value at point, then blur)
        if 0 <= col < width and 0 <= row_idx < height:
            raster[row_idx, col] += float(row["count"])
    
    # Apply Gaussian blur (spreads density)
    sigma = radius / cell_size  # Convert radius to pixels
    raster = gaussian_filter(raster, sigma=sigma)
    
    # Save
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs=OUTPUT_CRS,
        transform=transform,
        compress="deflate",
        TILED="YES",
        BLOCKXSIZE=256,
        BLOCKYSIZE=256
    ) as dst:
        dst.write(raster, 1)
    
    print(f"  ✓ {Path(output_path).name} (KDE)")
    return output_path

def combine_rasters(state_path, county_path, kde_path, output_path, weights=WEIGHTS):
    """
    Combine three rasters with weights: (state*w_s) + (county*w_c) + (kde*w_k)
    
    Outputs are aligned to same bounds/resolution.
    """
    # Read rasters
    with rasterio.open(state_path) as src_state:
        state_data = src_state.read(1).astype(np.float32)
        state_transform = src_state.transform
        state_bounds = src_state.bounds
    
    with rasterio.open(county_path) as src_county:
        county_data = src_county.read(1).astype(np.float32)
    
    with rasterio.open(kde_path) as src_kde:
        kde_data = src_kde.read(1).astype(np.float32)
    
    # Combine with weights (handle NaN)
    state_data = np.nan_to_num(state_data, 0)
    county_data = np.nan_to_num(county_data, 0)
    kde_data = np.nan_to_num(kde_data, 0)
    
    combined = (state_data * weights["state"] +
                county_data * weights["county"] +
                kde_data * weights["city"])
    
    # Save combined
    height, width = combined.shape
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs=OUTPUT_CRS,
        transform=state_transform,
        compress="deflate",
        TILED="YES",
        BLOCKXSIZE=256,
        BLOCKYSIZE=256
    ) as dst:
        dst.write(combined, 1)
    
    return output_path

def create_cumulative(prev_cum_path, new_iter_path, output_path):
    """
    Cumulative = previous cumulative + current iterative.
    """
    if prev_cum_path is None:
        # First bin: cumulative = iterative
        import shutil
        shutil.copy(new_iter_path, output_path)
        return output_path
    
    # Read both rasters
    with rasterio.open(prev_cum_path) as src_prev:
        prev_data = src_prev.read(1).astype(np.float32)
        prev_transform = src_prev.transform
    
    with rasterio.open(new_iter_path) as src_new:
        new_data = src_new.read(1).astype(np.float32)
    
    # Add (handle NaN)
    prev_data = np.nan_to_num(prev_data, 0)
    new_data = np.nan_to_num(new_data, 0)
    cumulative = prev_data + new_data
    
    # Save
    height, width = cumulative.shape
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs=OUTPUT_CRS,
        transform=prev_transform,
        compress="deflate",
        TILED="YES",
        BLOCKXSIZE=256,
        BLOCKYSIZE=256
    ) as dst:
        dst.write(cumulative, 1)
    
    return output_path

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEPS 1-5: RASTERIZE, COMBINE, PACKAGE")
    print("="*70)
    
    # 1. Load inputs
    states, counties, cities = load_shapefiles()
    bin_counts = load_count_csvs()
    
    if not bin_counts:
        print("ERROR: No bins found!")
        exit(1)
    
    # 2. Get common extent from states (all rasters will use this)
    extent_bounds = states.total_bounds  # [minx, miny, maxx, maxy]
    print(f"\n[*] Using common extent bounds: {extent_bounds}")
    
    # 2. Process each bin
    print(f"\n[*] Processing {len(bin_counts)} bins...")
    
    manifest = []  # Track all outputs for metadata
    cumulative_paths = defaultdict(lambda: None)  # Track cumulative per event
    
    for bin_idx, (event, bin_start, bin_end) in enumerate(sorted(bin_counts.keys())):
        counts_df = bin_counts[(event, bin_start, bin_end)]
        
        bin_str = bin_start.strftime("%Y%m%d_%H%M")
        print(f"\n  [{bin_idx+1}/{len(bin_counts)}] {event} | {bin_start} → {bin_end}")
        
        # --- Join counts to geometries
        print(f"    Joining counts...")
        states_joined = join_counts_to_geometry(states, counts_df, STATES_NAME_FIELD)
        counties_joined = join_counts_to_geometry(counties, counts_df, COUNTIES_NAME_FIELD)
        cities_joined = join_counts_to_geometry(cities, counts_df, CITIES_NAME_FIELD)
        
        # --- Rasterize (all with common extent for alignment)
        print(f"    Rasterizing...")
        state_rast = rasterize_layer(states_joined, 
                                     os.path.join(RASTERS_OUTPUT_DIR, f"{event}_{bin_str}_state.tif"),
                                     extent_bounds=extent_bounds)
        county_rast = rasterize_layer(counties_joined,
                                      os.path.join(RASTERS_OUTPUT_DIR, f"{event}_{bin_str}_county.tif"),
                                      extent_bounds=extent_bounds)
        kde_rast = create_kde_from_points(cities_joined,
                                          os.path.join(RASTERS_OUTPUT_DIR, f"{event}_{bin_str}_kde.tif"),
                                          extent_bounds=extent_bounds)
        
        if not (state_rast and county_rast and kde_rast):
            print(f"    Skipping bin (missing rasters)")
            continue
        
        # --- Combine
        print(f"    Combining with weights...")
        combined_iter = combine_rasters(state_rast, county_rast, kde_rast,
                                        os.path.join(RASTERS_OUTPUT_DIR, f"{event}_{bin_str}_iter.tif"))
        
        # --- Cumulative
        print(f"    Creating cumulative...")
        combined_cum = create_cumulative(cumulative_paths[event], combined_iter,
                                         os.path.join(RASTERS_OUTPUT_DIR, f"{event}_{bin_str}_cum.tif"))
        cumulative_paths[event] = combined_cum
        
        # --- Add to manifest
        manifest.append({
            "event": event,
            "bin_start": bin_start.isoformat(),
            "bin_end": bin_end.isoformat(),
            "iter_raster": Path(combined_iter).name,
            "cum_raster": Path(combined_cum).name,
            "iter_path": combined_iter,
            "cum_path": combined_cum
        })
    
    # 3. Package rasters into GeoPackage
    print(f"\n[*] Creating GeoPackage: {GPKG_OUTPUT}...")
    
    # Remove old GPKG if exists
    if Path(GPKG_OUTPUT).exists():
        os.remove(GPKG_OUTPUT)
    
    # Copy all rasters into GPKG
    # Note: rasterio doesn't directly support GPKG, so we'll keep TIFFs
    # and create a manifest table instead
    
    manifest_df = pd.DataFrame(manifest)
    
    # 4. Create temporal index JSON
    print(f"[*] Creating temporal index...")
    
    temporal_index = {
        "type": "FeatureCollection",
        "crs": OUTPUT_CRS,
        "properties": {
            "weights": WEIGHTS,
            "cell_size": CELL_SIZE,
            "created": datetime.now().isoformat()
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "event": row["event"],
                    "bin_start": row["bin_start"],
                    "bin_end": row["bin_end"],
                    "iter_raster": row["iter_raster"],
                    "cum_raster": row["cum_raster"],
                    "is_cumulative": False
                },
                "geometry": None  # Could add bbox here
            } for _, row in manifest_df.iterrows()
        ] + [
            {
                "type": "Feature",
                "properties": {
                    "event": row["event"],
                    "bin_start": row["bin_start"],
                    "bin_end": row["bin_end"],
                    "cum_raster": row["cum_raster"],
                    "is_cumulative": True
                },
                "geometry": None
            } for _, row in manifest_df.iterrows()
        ]
    }
    
    temporal_json_path = os.path.join(RASTERS_OUTPUT_DIR, "temporal_index.json")
    with open(temporal_json_path, "w") as f:
        json.dump(temporal_index, f, indent=2)
    
    # 5. Save manifest CSV
    manifest_csv = os.path.join(RASTERS_OUTPUT_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    
    # 6. Summary
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\nOutputs in: {RASTERS_OUTPUT_DIR}/")
    print(f"  - {len(manifest)} bins processed")
    print(f"  - {len(manifest)*2} rasters (iterative + cumulative)")
    print(f"  - manifest.csv (metadata)")
    print(f"  - temporal_index.json (for QGIS)")
    
    print(f"\nNext: Open QGIS")
    print(f"  1. Add rasters → {RASTERS_OUTPUT_DIR}/*_iter.tif or *_cum.tif")
    print(f"  2. View → Temporal Controller → Enable time slider")
    print(f"  3. Layer → Properties → Temporal tab → Set based on filenames")
    print(f"  4. Hit play!")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
