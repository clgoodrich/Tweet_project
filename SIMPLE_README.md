# Hurricane Tweet Analysis - Quick Start

## What You Have

Two analysis outputs ready to load in ArcGIS Pro:

### 1. Vector Time Series (RECOMMENDED)
**File:** `output/hurricane_analysis_output.gpkg`

**Layers:**
- `helene_states_timeseries` - State polygons with tweet counts over time
- `helene_counties_timeseries` - County polygons with tweet counts over time
- `francine_states_timeseries`
- `francine_counties_timeseries`

**Load in ArcGIS Pro:**
1. Add Data → Browse to `output/hurricane_analysis_output.gpkg`
2. Add any layer (e.g., `helene_counties_timeseries`)
3. Right-click layer → Properties → Time tab
4. Enable time:
   - Start Time Field: `time_start`
   - End Time Field: `time_end`
5. View → Time Slider
6. Click Play to animate

**Symbology:**
- Graduated Colors
- Field: `tweet_count` or `tweets_per_1000sqkm`
- Color: Yellow-Orange-Red

---

### 2. Raster Heat Maps (OPTIONAL)
**Location:** `rasters_output/helene/`

**Files:**
- `slice_000_iterative.tif` through `slice_020_iterative.tif` (21 time slices)
- `slice_000_cumulative.tif` through `slice_020_cumulative.tif` (cumulative)

**Load in ArcGIS Pro:**
1. Add Data → Add Multiple Files
2. Select all `slice_*_iterative.tif` files
3. Right-click group → Properties → Time
4. Enable time from file metadata
5. Symbology: Stretched, heat colors, 0 = transparent

---

## Which One to Use?

- **Vector (GeoPackage)**: Administrative boundaries, clean stats, easy to use
- **Raster (GeoTIFFs)**: Heat map visualization, hotspot detection

**Start with the vector GeoPackage** - it's simpler and already configured.

---

## Re-run Analysis

### Vector Analysis
```bash
jupyter notebook hurricane_spatiotemporal_analysis.ipynb
# Kernel → Restart & Run All
```

### Raster Analysis
```bash
jupyter notebook hurricane_raster_fusion.ipynb
# Edit CONFIG cell to change event/resolution
# Kernel → Restart & Run All
```

---

That's it! The vector GeoPackage is your quickest path to time-enabled visualization in ArcGIS Pro.
