import arcpy
import os
from datetime import datetime

# Paths
raster_folder = r"C:\Users\colto\Documents\GitHub\Tweet_project\rasters_cumulative_francine"
symbology_file = r"C:\Users\colto\Documents\ArcGIS\Projects\SAOCOM\francine symbology.lyrx"
gdb_path = r"C:\Users\colto\Documents\GitHub\Tweet_project\Tweet_project.gdb"
mosaic_name = "francine_cumulative_mosaic"

# Create geodatabase if it doesn't exist
if not arcpy.Exists(gdb_path):
    arcpy.CreateFileGDB_management(os.path.dirname(gdb_path), os.path.basename(gdb_path))

# Create mosaic dataset
mosaic_path = os.path.join(gdb_path, mosaic_name)
if arcpy.Exists(mosaic_path):
    arcpy.Delete_management(mosaic_path)

arcpy.CreateMosaicDataset_management(gdb_path, mosaic_name, "PROJCS['WGS_1984_Web_Mercator_Auxiliary_Sphere',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Mercator_Auxiliary_Sphere'],PARAMETER['False_Easting',0.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',0.0],PARAMETER['Standard_Parallel_1',0.0],PARAMETER['Auxiliary_Sphere_Type',0.0],UNIT['Meter',1.0]]")

print(f"Created mosaic dataset: {mosaic_path}")

# Add rasters to mosaic
arcpy.AddRastersToMosaicDataset_management(
    mosaic_path,
    "Raster Dataset",
    raster_folder,
    filter="*.tif"
)

print("Added rasters to mosaic dataset")

# Add time field
arcpy.AddField_management(mosaic_path, "AcquisitionDate", "DATE")

# Calculate time from filename
with arcpy.da.UpdateCursor(mosaic_path, ["Name", "AcquisitionDate"]) as cursor:
    for row in cursor:
        filename = row[0]
        # Extract time string: _20240926_080000
        time_str = filename.split("_")[-2] + filename.split("_")[-1].replace(".tif", "")
        # Parse: 20240926080000 -> datetime
        dt = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        row[1] = dt
        cursor.updateRow(row)

print("Time field populated")

# Configure mosaic properties
arcpy.SetMosaicDatasetProperties_management(
    mosaic_path,
    start_time_field="AcquisitionDate"
)

print("Mosaic dataset configured with time dimension")

print(f"\nMosaic dataset complete: {mosaic_path}")
print("To apply symbology in ArcGIS Pro:")
print(f"1. Add mosaic to map: {mosaic_path}")
print(f"2. Right-click layer > Symbology > Import")
print(f"3. Select: {symbology_file}")
print("4. Enable time slider to animate cumulative growth")