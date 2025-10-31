Purpose

Guide Claude Code to produce an ArcGIS-native notebook (uses arcpy only) that converts the Tweets project inputs into iterative and cumulative time-aware rasters for Francine and Helene, ready for animation and Space-Time Pattern Mining in ArcGIS Pro.

Inputs (fixed; do not alter schemas/columns)

...\tables\cities1000.csv

...\shape_files\cb_2023_us_county_20m.shp

...\shape_files\cb_2023_us_state_20m.shp

...\geojson\francine.geojson

...\geojson\helene.geojson

Event GeoJSON schema to rely on

Properties: FAC, LOC, GPE, time, Latitude, Longitude, make_polygon

Geometry: Point with [lon, lat]

Use geometry XY; Latitude/Longitude only for QA.

Must produce (per event: francine, helene; per mode: iter, cum)

CRF time-enabled raster (<event>_<mode>.crf)

Space-Time Cube (<event>_<mode>.nc)

Stack of aligned GeoTIFF slices + manifest (GDB table + CSV)

Required ArcGIS toolchain (scripted)

CreateMosaicDataset

AddRastersToMosaicDataset (Raster Type = Table; field = RasterPath)

BuildMultidimensionalInfo (Date=Date, Variable=Var)

(Optional) BuildPyramidsandStatistics

CopyRaster → CRF with process_as_multidimensional

stpm.CreateSpaceTimeCubeFromMDLayer → .nc

The fusion of states/counties/cities into pixels is up to the notebook. Keep it general, documented in one paragraph, without naming specific methods.

Notebook skeleton

Config: events, cell_km, time_bin, weights, gdb, out_root, crs_epsg, nodata.

I/O + validation (assert all files; assert Point geometry; parse time with tz).

Time binning (contiguous, fill empty bins).

Fusion & slice creation (one raster per bin, iterative and cumulative).

Mosaic → MD fields → CRF → Cube.

QA: JSON summary; quick checks for consistent extent/CRS/pixel size and slice counts.

Style & robustness

Succinct helpers; early asserts; deterministic filenames.

Logs compact; errors actionable (NEED_INFO: ...).

No GeoPandas/Rasterio/online deps—arcpy only.

Success criteria

For both hurricanes, you can add the CRFs to a map and scrub the time slider; .nc works with Space-Time Pattern Mining and 3D visualization.

All slices share the same CRS, extent, pixel size; mosaic footprints show Standard Time / Dimensions / Variable after building multidimensional info.