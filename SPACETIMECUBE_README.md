# Space Time Cube - Installation & Usage

## Install Required Package

```bash
pip install netCDF4
```

OR if using conda:
```bash
conda install -c conda-forge netcdf4
```

## Run the Notebook

```bash
jupyter notebook create_spacetimecube.ipynb
```

- Opens automatically
- Click: Kernel → Restart & Run All
- Wait ~2-3 minutes

## Output

**Location:** `spacetimecube_output/`

**4 NetCDF files created:**
- `helene_iterative.nc` - Per-bin tweet counts (pulses)
- `helene_cumulative.nc` - Growing totals over time
- `francine_iterative.nc` - Per-bin tweet counts
- `francine_cumulative.nc` - Growing totals over time

## Load in ArcGIS Pro

**Method 1 (Easiest):**
1. Open Catalog Pane
2. Navigate to `spacetimecube_output/` folder
3. Right-click on .nc file (e.g., `helene_iterative.nc`)
4. Select **Add to Current Map**
5. ArcGIS will prompt to select variable → Choose **COUNT**
6. Time dimension auto-detected
7. **View** → **Time Slider** to animate

**Method 2:**
1. **Map** tab → **Add Data** → **Data**
2. Browse to `spacetimecube_output/`
3. Select .nc file
4. Choose **COUNT** variable
5. Use **Time Slider**

**Method 3 (Drag & Drop):**
- Drag .nc file from File Explorer directly onto map

## What's the Difference?

**Iterative** = Shows activity per time bin (temporal pulses)
- Good for: Finding peak activity times, event patterns

**Cumulative** = Shows growing spatial footprint
- Good for: Seeing spread over time, total impact

## Key Time Encoding Fix

This version uses `date2num()` with `'hours since 1970-01-01 00:00:00'` which is CF-compliant and works with ArcGIS Pro's time slider. The time values are stored as numeric hours since epoch, not datetime strings.
