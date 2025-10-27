from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling

in_dir   = Path(r"PATH\TO\tiles")
inputs   = sorted(in_dir.glob("*.tif"))
assert inputs, "No input rasters found."

# Inspect first raster for metadata “template”
with rasterio.open(inputs[0]) as src0:
    profile = src0.profile.copy()
    dtype   = src0.dtypes[0]
    nodata  = src0.nodata if src0.nodata is not None else -9999

# Open all rasters (must share CRS/resolution/transform)
srcs = [rasterio.open(str(p)) for p in inputs]
mosaic, out_transform = merge(srcs, nodata=nodata, method="first")  # or method="max"/"sum"

# Ensure dtype and nodata are preserved
mosaic = mosaic.astype(dtype, copy=False)

profile.update(
    driver="GTiff",
    height=mosaic.shape[1],
    width=mosaic.shape[2],
    transform=out_transform,
    nodata=nodata,
    tiled=True,
    compress="DEFLATE",
    predictor=2,
    BIGTIFF="YES"
)

out_tif = r"PATH\TO\mosaic.tif"
with rasterio.open(out_tif, "w", **profile) as dst:
    dst.write(mosaic)
for s in srcs: s.close()
