"""
STEP 0: Parse Hurricane GeoJSON → 4-hour binned count CSVs
=========================================================

This script:
  1. Loads your helene.geojson (and any other event GeoJSON)
  2. Extracts GPE (state/county/city mentions), timestamps, coordinates
  3. Bins events into 4-hour windows
  4. Counts mentions per GPE per bin
  5. Outputs CSVs ready to join to shapefiles in QGIS

Dependencies: pandas, geopandas (pip install pandas geopandas)

Input:  helene.geojson, francine.geojson (if available)
Output: 
  - counts_helene_*.csv (per bin with GPE → count)
  - bins_summary.csv (metadata on all bins)
  - geometries_helene_*.shp (optional: points per bin for sanity check)
"""

import json
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from pathlib import Path
from shapely.geometry import Point
import os

# ============================================================================
# CONFIG
# ============================================================================

INPUT_GEOJSONS = [
    "helene.geojson",
    # "francine.geojson",  # uncomment if you have this
]

OUTPUT_DIR = "./counts_output"
BIN_HOURS = 4

# ============================================================================
# SETUP
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# FUNCTION: Parse a single GeoJSON into a GeoDataFrame
# ============================================================================

def parse_geojson(filepath):
    """
    Load a GeoJSON and extract: event, time, GPE, coordinates.
    
    Returns: GeoDataFrame with columns:
      - event (str): name of event (e.g., 'helene')
      - time (datetime): parsed timestamp
      - GPE (str): Geopolitical Entity (state/county/city from the data)
      - geometry (Point): lon/lat
    """
    print(f"\n[*] Parsing {filepath}...")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    event_name = Path(filepath).stem  # 'helene' from 'helene.geojson'
    rows = []
    errors = 0
    
    for i, feat in enumerate(data.get("features", [])):
        try:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            
            # Extract GPE (state/county/city name)
            gpe = props.get("GPE", "").strip()
            if not gpe:
                continue  # Skip if no GPE
            
            # Parse timestamp
            time_str = props.get("time", "")
            if not time_str:
                errors += 1
                continue
            
            # Handle ISO format with timezone
            try:
                time_dt = pd.to_datetime(time_str)
            except:
                errors += 1
                continue
            
            # Extract coordinates
            if geom.get("type") == "Point":
                coords = geom.get("coordinates", [None, None])
                lon, lat = coords[0], coords[1]
            else:
                # If not a point, skip
                errors += 1
                continue
            
            if lon is None or lat is None:
                errors += 1
                continue
            
            rows.append({
                "event": event_name,
                "time": time_dt,
                "GPE": gpe,
                "latitude": lat,
                "longitude": lon,
                "geometry": Point(lon, lat)
            })
        
        except Exception as e:
            errors += 1
            if i < 5:  # Only print first few errors
                print(f"  Warning: row {i} failed: {e}")
    
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    print(f"  ✓ Loaded {len(gdf)} features, {errors} errors")
    
    return gdf

# ============================================================================
# FUNCTION: Bin and count mentions
# ============================================================================

def bin_and_count(gdf, bin_hours=4):
    """
    Bin GeoDataFrame by time, count mentions per GPE per bin.
    
    Returns: dict of (event, bin_start, bin_end) → DataFrame with (GPE, count)
    """
    print(f"\n[*] Binning into {bin_hours}-hour windows...")
    
    # Add bin column
    gdf["bin"] = gdf["time"].dt.floor(f"{bin_hours}H")
    
    # Group by event + bin, then count GPE mentions
    bin_dict = {}
    
    for (event, bin_time), group in gdf.groupby(["event", "bin"]):
        # Count each GPE in this bin
        counts = group["GPE"].value_counts().reset_index()
        counts.columns = ["GPE", "count"]
        
        # Store with readable bin end time
        bin_start = bin_time
        bin_end = bin_time + timedelta(hours=bin_hours)
        
        key = (event, bin_start, bin_end)
        bin_dict[key] = counts
        
        print(f"  ✓ {event} | {bin_start} → {bin_end}: {len(counts)} unique GPEs, {counts['count'].sum()} total mentions")
    
    return bin_dict

# ============================================================================
# FUNCTION: Export counts to CSV (for QGIS joining)
# ============================================================================

def export_counts_csv(bin_dict, output_dir, gdf):
    """
    Export one CSV per bin, ready to join to shapefiles in QGIS.
    
    CSV format:
      GPE, count, bin_start, bin_end, event
    """
    print(f"\n[*] Exporting count CSVs...")
    
    bin_metadata = []
    
    for (event, bin_start, bin_end), counts_df in sorted(bin_dict.items()):
        # Add metadata columns
        counts_df["bin_start"] = bin_start
        counts_df["bin_end"] = bin_end
        counts_df["event"] = event
        
        # Create filename
        bin_str = bin_start.strftime("%Y%m%d_%H%M")
        filename = f"counts_{event}_{bin_str}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save
        counts_df.to_csv(filepath, index=False)
        
        print(f"  ✓ {filename} ({len(counts_df)} GPEs)")
        
        # Add to metadata
        bin_metadata.append({
            "event": event,
            "bin_start": bin_start,
            "bin_end": bin_end,
            "csv_file": filename,
            "unique_gpes": len(counts_df),
            "total_mentions": counts_df["count"].sum()
        })
    
    # Export metadata summary
    metadata_df = pd.DataFrame(bin_metadata)
    metadata_path = os.path.join(output_dir, "bins_summary.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\n  ✓ Summary: {metadata_path}")
    
    return metadata_df

# ============================================================================
# FUNCTION: Create shapefiles per bin (optional, for QA)
# ============================================================================

def export_geometries_shp(gdf, bin_dict, output_dir):
    """
    Optional: Export points per bin as shapefiles for visual QA in QGIS.
    
    This lets you verify your binning and GPE extraction worked.
    """
    print(f"\n[*] Exporting geometries per bin (optional QA)...")
    
    for (event, bin_start, bin_end), counts_df in sorted(bin_dict.items()):
        # Filter GDF to just this bin
        bin_gdf = gdf[
            (gdf["event"] == event) &
            (gdf["bin"] == bin_start)
        ].copy()
        
        if len(bin_gdf) == 0:
            continue
        
        # Keep only needed columns
        bin_gdf = bin_gdf[["GPE", "latitude", "longitude", "geometry"]]
        
        # Save shapefile
        bin_str = bin_start.strftime("%Y%m%d_%H%M")
        shp_path = os.path.join(output_dir, f"geometries_{event}_{bin_str}.shp")
        bin_gdf.to_file(shp_path, driver="ESRI Shapefile")
        
        print(f"  ✓ geometries_{event}_{bin_str}.shp ({len(bin_gdf)} points)")

# ============================================================================
# FUNCTION: Print summary statistics
# ============================================================================

def print_summary(gdf, bin_dict, metadata_df):
    """Print a nice summary of what was processed."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nTotal features loaded: {len(gdf)}")
    print(f"Total bins created: {len(bin_dict)}")
    print(f"Total unique GPEs: {gdf['GPE'].nunique()}")
    print(f"\nTime range: {gdf['time'].min()} → {gdf['time'].max()}")
    print(f"Duration: {(gdf['time'].max() - gdf['time'].min()).days} days, {(gdf['time'].max() - gdf['time'].min()).seconds // 3600} hours")
    
    print(f"\nBreakdown by event:")
    for event in gdf["event"].unique():
        n = len(gdf[gdf["event"] == event])
        print(f"  {event}: {n} features")
    
    print(f"\nTop 10 most-mentioned GPEs:")
    top_gpes = gdf["GPE"].value_counts().head(10)
    for gpe, count in top_gpes.items():
        print(f"  {gpe}: {count}")
    
    print("\n" + "="*70)
    print(f"✅ DONE! Output files in: {OUTPUT_DIR}")
    print("="*70)
    print("\nNext steps:")
    print("1. Load your shapefiles (states, counties, cities) in QGIS")
    print("2. For each bin, join the corresponding counts_*.csv to the shapefile:")
    print("   - Right-click layer → Properties → Joins tab → Add Join")
    print("   - Join field: GPE (from CSV)")
    print("   - Target field: NAME (from shapefile)")
    print("3. Rasterize each joined layer using Processing → Raster → Rasterize")
    print("4. Use Raster Calculator to combine: (state*0.1) + (county*0.5) + (city*1.0)")
    print("5. Stack rasters into GeoPackage and enable Temporal Controller")
    print("\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # 1. Load all GeoJSONs
    all_gdfs = []
    for geojson_path in INPUT_GEOJSONS:
        if Path(geojson_path).exists():
            gdf = parse_geojson(geojson_path)
            all_gdfs.append(gdf)
        else:
            print(f"⚠️  Warning: {geojson_path} not found, skipping")
    
    if not all_gdfs:
        print("❌ No GeoJSON files found!")
        exit(1)
    
    # Combine all events into one GDF
    gdf = pd.concat(all_gdfs, ignore_index=True)
    print(f"\nTotal combined features: {len(gdf)}")
    
    # 2. Bin and count
    bin_dict = bin_and_count(gdf, bin_hours=BIN_HOURS)
    
    # 3. Export CSVs
    metadata_df = export_counts_csv(bin_dict, OUTPUT_DIR, gdf)
    
    # 4. Optional: Export geometries for QA
    export_geometries_shp(gdf, bin_dict, OUTPUT_DIR)
    
    # 5. Print summary
    print_summary(gdf, bin_dict, metadata_df)
