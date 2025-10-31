# Hurricane Tweet Analysis - ArcGIS Time-Aware Rasters

## Quick Start

### 1. Run the Notebook
```bash
jupyter notebook spatiotemporal_raster.ipynb
```
- Edit CONFIG cell if needed (event, cell size, time bins)
- Kernel → Restart & Run All
- Wait ~2 minutes

### 2. Find Your Outputs
`rasters_output/helene/` (or `rasters_output/francine/`)

**Files:**
- `slice_000_iterative.tif` through `slice_020_iterative.tif`
- `slice_000_cumulative.tif` through `slice_020_cumulative.tif`
- `metadata.json`

### 3. Load in ArcGIS Pro

**Simple way:**
1. Add Data → Add Multiple Files
2. Select all `*_iterative.tif` files
3. Right-click group → Properties → Time
4. Enable time from file metadata
5. View → Time Slider → Play

**Symbology:**
- Stretched renderer
- White → Yellow → Orange → Red
- Set 0 as transparent

---

## What It Does

Takes hurricane tweets (coordinates + timestamps) and creates:
- Heat map rasters showing tweet intensity
- Time slices (one per time bin)
- Cumulative slices (growing over time)

**Iterative** = activity per time bin
**Cumulative** = total activity from start to that time

---

## Configuration

Edit the CONFIG cell:
```python
CONFIG = {
    'event': 'helene',          # or 'francine'
    'cell_size_km': 5,          # grid resolution
    'time_bin_hours': 2,        # time slice interval
    'crs': 'EPSG:5070',        # Albers Equal Area
    'weights': {
        'coordinates': 0.60,    # direct tweet locations
        'county': 0.25,         # county aggregation
        'state': 0.15           # state aggregation
    }
}
```

---

That's it. Run notebook → Load in ArcGIS → Animate with time slider.
