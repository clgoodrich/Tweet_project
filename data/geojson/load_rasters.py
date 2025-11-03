import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from qgis.core import QgsProject, QgsRasterLayer, QgsDateTimeRange, Qgis, QgsLayerTreeLayer, \
    QgsCoordinateReferenceSystem

RASTERS_DIR = r"C:\Users\colto\Documents\GitHub\Tweet_project\data\geojson\rasters_output"
RASTER_TYPE = "*_iter.tif"
CRS = QgsCoordinateReferenceSystem("EPSG:5070")


def parse_timestamp_from_filename(filename):
    match = re.search(r'_(\d{8})_(\d{4})_', filename)
    if not match:
        return None, None
    date_str = match.group(1)
    time_str = match.group(2)
    try:
        dt_str = f"{date_str}{time_str}"
        start_dt = datetime.strptime(dt_str, "%Y%m%d%H%M")
        end_dt = start_dt + timedelta(hours=4)
        return start_dt, end_dt
    except:
        return None, None


iface = qgis.utils.iface
project = QgsProject.instance()
root = project.layerTreeRoot()

print("\nLoading rasters with temporal...")

rasters_path = Path(RASTERS_DIR)
raster_files = sorted(rasters_path.glob(RASTER_TYPE))

layers_with_time = []

for idx, raster_path in enumerate(raster_files):
    filename = raster_path.name
    start_dt, end_dt = parse_timestamp_from_filename(filename)
    if start_dt is None:
        continue

    layer = QgsRasterLayer(str(raster_path), filename, "gdal")
    if not layer.isValid():
        continue

    layer.setCrs(CRS)
    project.addMapLayer(layer, False)

    temporal_props = layer.temporalProperties()
    temporal_props.setIsActive(True)
    temporal_props.setMode(Qgis.RasterTemporalMode.FixedRangePerBand)
    temporal_props.setFixedTemporalRange(QgsDateTimeRange(start_dt, end_dt, True, True))

    layers_with_time.append((layer, start_dt, filename))
    print(f"  {filename} {start_dt.strftime('%Y-%m-%d %H:%M')}")

event_groups = {}
for layer, start_dt, filename in sorted(layers_with_time, key=lambda x: x[1]):
    event = filename.split('_')[0]
    if event not in event_groups:
        group = root.addGroup(event)
        event_groups[event] = group
    group.addChildNode(QgsLayerTreeLayer(layer))

iface.mapCanvas().refresh()
print(f"\n✅ {len(layers_with_time)} layers loaded")