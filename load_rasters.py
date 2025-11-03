
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from qgis.core import QgsProject, QgsRasterLayer, QgsDateTimeRange, Qgis

RASTERS_DIR = r"C:\Users\colto\Documents\GitHub\Tweet_project\rasters_output"
RASTER_TYPE = "*_iter.tif"


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

print("\n" + "=" * 70)
print("Loading rasters + setting temporal...")
print("=" * 70)

rasters_path = Path(RASTERS_DIR)
raster_files = sorted(rasters_path.glob(RASTER_TYPE))
print(f"\nFound {len(raster_files)} rasters\n")

layers_with_time = []

for idx, raster_path in enumerate(raster_files):
    filename = raster_path.name
    print(f"[{idx + 1}/{len(raster_files)}] {filename}", end="")

    start_dt, end_dt = parse_timestamp_from_filename(filename)
    if start_dt is None:
        print(" - SKIP")
        continue

    layer = QgsRasterLayer(str(raster_path), filename, "gdal")
    if not layer.isValid():
        print(" - SKIP")
        continue

    project.addMapLayer(layer, False)
    temporal_props = layer.temporalProperties()
    temporal_props.setIsActive(True)
    temporal_props.setMode(Qgis.RasterTemporalMode.FixedRangePerBand)
    temporal_props.setFixedTemporalRange(QgsDateTimeRange(start_dt, end_dt))

    print(f" - OK")
    layers_with_time.append((layer, start_dt, filename))

print(f"\nOrganizing by event...")
event_groups = {}

for layer, start_dt, filename in sorted(layers_with_time, key=lambda x: x[1]):
    event = filename.split('_')[0]
    if event not in event_groups:
        group = root.addGroup(event)
        event_groups[event] = group
    else:
        group = event_groups[event]
    root.insertLayer(group, 0, layer)

iface.mapCanvas().refresh()

print(f"\n✅ Complete! {len(event_groups)} event group(s), {len(layers_with_time)} layers")
print("View → Panels → Temporal Controller, then click PLAY")
print("=" * 70 + "\n")
