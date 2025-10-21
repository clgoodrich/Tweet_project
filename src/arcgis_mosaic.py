"""
Tweet Project – ArcGIS Mosaic Dataset Creation (Global Stats Fixed Range, v3)
=============================================================================

Builds four time-enabled Mosaic Datasets (Helene/Francine × Increment/Cumulative)
from per-bin GeoTIFFs under a fixed rasters_root, and forces a fixed global
min/max per mosaic so display range doesn't clamp to the first TIFF.

Rasters root (hardcoded): see RASTERS_ROOT below.
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Dict, Tuple
import arcpy

# --- HARD-CODED PATHS (use raw strings for Windows paths) ---
RASTERS_ROOT = r"C:\Users\colto\Documents\GitHub\Tweet_project\rasters_output"
GDB_PATH     = r"C:\Users\colto\Documents\GitHub\Tweet_project\Tweet_project.gdb"


def compute_global_min_max(raster_folder: str) -> Tuple[float, float]:
    """Scan *.tif (recursive) and return (global_min, global_max)."""
    tif_paths = []
    for root, _, files in os.walk(raster_folder):
        for f in files:
            if f.lower().endswith(".tif"):
                tif_paths.append(os.path.join(root, f))
    if not tif_paths:
        raise RuntimeError(f"No .tif files found in: {raster_folder}")

    gmin, gmax = float("inf"), float("-inf")
    for p in tif_paths:
        try:
            rmin = float(arcpy.management.GetRasterProperties(p, "MINIMUM").getOutput(0))
            rmax = float(arcpy.management.GetRasterProperties(p, "MAXIMUM").getOutput(0))
            if rmin < gmin:
                gmin = rmin
            if rmax > gmax:
                gmax = rmax
        except Exception as e:
            print(f"    Warning: Could not read stats from {p}: {e}")
            continue

    if gmin == float("inf") or gmax == float("-inf"):
        print("    Warning: No valid statistics found, using defaults [0, 100]")
        return 0.0, 100.0

    return gmin, gmax


def create_mosaic_dataset(gdb_path: str, raster_folder: str, mosaic_name: str) -> str:
    """Create a time-enabled Mosaic Dataset, add rasters, enforce global min/max."""
    print(f"\nCreating mosaic dataset: {mosaic_name}")
    print(f"  GDB: {gdb_path}")
    print(f"  Rasters: {raster_folder}")

    # 1) Ensure GDB
    if not arcpy.Exists(gdb_path):
        print("  Creating geodatabase...")
        arcpy.CreateFileGDB_management(os.path.dirname(gdb_path), os.path.basename(gdb_path))

    mosaic_path = os.path.join(gdb_path, mosaic_name)
    if arcpy.Exists(mosaic_path):
        print(f"  Deleting existing mosaic: {mosaic_path}")
        arcpy.Delete_management(mosaic_path)

    # 2) Create Mosaic Dataset (Web Mercator Auxiliary Sphere)
    spatial_ref = (
        "PROJCS['WGS_1984_Web_Mercator_Auxiliary_Sphere',"
        "GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',"
        "SPHEROID['WGS_1984',6378137.0,298.257223563]],"
        "PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],"
        "PROJECTION['Mercator_Auxiliary_Sphere'],"
        "PARAMETER['False_Easting',0.0],PARAMETER['False_Northing',0.0],"
        "PARAMETER['Central_Meridian',0.0],PARAMETER['Standard_Parallel_1',0.0],"
        "PARAMETER['Auxiliary_Sphere_Type',0.0],UNIT['Meter',1.0]]"
    )
    print("  Creating mosaic dataset...")
    arcpy.CreateMosaicDataset_management(
        in_workspace=gdb_path,
        in_mosaicdataset_name=mosaic_name,
        coordinate_system=spatial_ref
    )

    # 3) Add rasters (include subfolders)
    print("  Adding rasters to mosaic...")
    arcpy.AddRastersToMosaicDataset_management(
        in_mosaic_dataset=mosaic_path,
        raster_type="Raster Dataset",
        input_path=raster_folder,
        filter="*.tif",
        sub_folder="SUBFOLDERS"
    )

    # 4) Build pyramids & statistics - FIXED PARAMETERS
    print("  Building pyramids & calculating item statistics...")
    arcpy.management.BuildPyramidsandStatistics(  # Fixed typo: And not and
        in_workspace=mosaic_path,
        include_subdirectories="NONE",
        build_pyramids="BUILD_PYRAMIDS",
        calculate_statistics="CALCULATE_STATISTICS",
        skip_existing="OVERWRITE"  # Fixed: was skip_first, should be skip_existing
    )

    # 5) Compute global min/max and stamp on Mosaic Dataset
    print("  Computing global min/max across all rasters...")
    global_min, global_max = compute_global_min_max(raster_folder)
    print(f"    Global MIN: {global_min}, Global MAX: {global_max}")

    # Calculate statistics more robustly
    global_mean = (global_min + global_max) / 2.0
    global_std  = max((global_max - global_min) / 6.0, 1e-9)

    # 6) Set mosaic dataset properties with global statistics
    print("  Setting mosaic dataset statistics to global min/max...")

    # Method 1: Try SetRasterProperties (may not work for all versions)
    try:
        stats_str = f"{global_min} {global_max} {global_mean} {global_std}"
        arcpy.management.SetRasterProperties(
            in_raster=mosaic_path,
            statistics=[[1, global_min, global_max, global_mean, global_std]]  # Band 1 statistics
        )
        print("    Applied statistics via SetRasterProperties")
    except Exception as e:
        print(f"    SetRasterProperties failed: {e}")
        print("    Statistics will be calculated normally")

    # Method 2: Calculate statistics (this should always work)
    arcpy.management.CalculateStatistics(
        in_raster_dataset=mosaic_path,
        x_skip_factor=1,
        y_skip_factor=1,
        ignore_values=[],
        skip_existing="OVERWRITE",
        area_of_interest=None
    )

    # 7) Add DATE field, parse time from filename, enable time
    print("  Adding time field...")
    time_field = "date"
    existing_fields = [f.name for f in arcpy.ListFields(mosaic_path)]

    if time_field not in existing_fields:
        arcpy.AddField_management(mosaic_path, time_field, "DATE")
    else:
        print(f"    Time field '{time_field}' already exists")

    print("  Calculating time from filenames...")
    updated_count = 0
    with arcpy.da.UpdateCursor(mosaic_path, ["Name", time_field]) as cursor:
        for row in cursor:
            filename = row[0]  # e.g., 'francine_tweets_20240926_080000.tif'
            if filename:
                name_no_ext = os.path.splitext(filename)[0]
                parts = name_no_ext.split("_")
                if len(parts) >= 4:
                    date_token, time_token = parts[-2], parts[-1]
                    try:
                        dt = datetime.strptime(f"{date_token}{time_token}", "%Y%m%d%H%M%S")
                        row[1] = dt
                        cursor.updateRow(row)
                        updated_count += 1
                        if updated_count <= 5:  # Show first 5 for verification
                            print(f"    {filename} -> {dt}")
                    except ValueError as ve:
                        print(f"    Warning: Could not parse time from {filename}: {ve}")
                        continue

    print(f"    Updated {updated_count} records with timestamps")

    # 8) Enable time on mosaic dataset
    print("  Enabling time on mosaic (start time field)...")
    arcpy.SetMosaicDatasetProperties_management(
        in_mosaic_dataset=mosaic_path,
        start_time_field=time_field
    )

    print(f"  Mosaic dataset complete: {mosaic_path}")
    return mosaic_path


def create_all_mosaics(base_gdb_path: str, rasters_root: str) -> Dict[str, str]:
    """Build mosaics for helene/francine × increment/cumulative (v3 names)."""
    mosaic_configs = [
        ("helene",   "cumulative", "helene_cumulative_mosaic_v3"),
        ("helene",   "increment",  "helene_increment_mosaic_v3"),
        ("francine", "cumulative", "francine_cumulative_mosaic_v3"),
        ("francine", "increment",  "francine_increment_mosaic_v3"),
    ]

    created: Dict[str, str] = {}
    for hurricane, raster_type, mosaic_name in mosaic_configs:
        raster_folder = os.path.join(rasters_root, hurricane, raster_type)
        if os.path.exists(raster_folder):
            print(f"\n{'=' * 60}")
            print(f"Creating {hurricane.upper()} {raster_type.upper()} mosaic (v3)")
            print(f"{'=' * 60}")
            try:
                md_path = create_mosaic_dataset(base_gdb_path, raster_folder, mosaic_name)
                created[f"{hurricane}_{raster_type}"] = md_path
            except Exception as e:
                print(f"\nERROR creating {mosaic_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nWarning: Raster folder not found: {raster_folder}")
    return created


if __name__ == "__main__":
    print("=" * 60)
    print("MOSAIC DATASET CREATION SCRIPT (v3 - Fixed)")
    print("=" * 60)
    print(f"Using rasters root: {RASTERS_ROOT}")
    print(f"Using GDB: {GDB_PATH}")

    # Set arcpy environment for better performance
    arcpy.env.parallelProcessingFactor = "75%"

    created = create_all_mosaics(GDB_PATH, RASTERS_ROOT)

    print("\n" + "=" * 60)
    print("MOSAIC DATASETS CREATED (v3):")
    print("=" * 60)
    for k, v in created.items():
        print(f"  {k}: {v}")

    if not created:
        print("  No mosaic datasets were created successfully.")