import json
from pathlib import Path

# Create notebook structure
nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Define cells
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "# Hurricane Raster Fusion Analysis\n\n**PLAN:** Load tweets → Create grid → Temporal bins → Fuse signals → Export rasters\n\n**DO:** Execute pipeline\n\n**VERIFY:** Check outputs"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "CONFIG = {\n    'event': 'helene',\n    'cell_size_km': 5,\n    'time_bin_hours': 2,\n    'weights': {'coordinates': 0.50, 'city': 0.30, 'county': 0.15, 'state': 0.05},\n    'crs': 'EPSG:5070',\n    'export_slices': True\n}\nassert abs(sum(CONFIG['weights'].values()) - 1.0) < 0.001\nprint(f\"Config: {CONFIG['event']}, {CONFIG['cell_size_km']}km\")"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "import geopandas as gpd\nimport pandas as pd\nimport numpy as np\nimport rasterio\nfrom rasterio.transform import from_bounds\nfrom pathlib import Path\nfrom datetime import datetime\nimport warnings, json\nwarnings.filterwarnings('ignore')\nprint('Imports ready')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "DATA = Path(r'C:\\Users\\colto\\Documents\\GitHub\\Tweet_project\\data')\nOUT = Path(r'C:\\Users\\colto\\Documents\\GitHub\\Tweet_project\\rasters_output')\nOUT.mkdir(exist_ok=True)\nEVENT_DIR = OUT / CONFIG['event']\nEVENT_DIR.mkdir(exist_ok=True)\nprint(f'Output: {EVENT_DIR}')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "print('Loading tweets...')\ntweets = gpd.read_file(DATA / 'geojson' / f\"{CONFIG['event']}.geojson\")\ntweets['time'] = pd.to_datetime(tweets['time'])\nif tweets.crs is None:\n    tweets.set_crs('EPSG:4326', inplace=True)\ntweets = tweets.to_crs(CONFIG['crs'])\nprint(f'{len(tweets)} tweets, {tweets[\"time\"].min()} to {tweets[\"time\"].max()}')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "buffer_m = 100000\ncell_m = CONFIG['cell_size_km'] * 1000\nbounds = tweets.total_bounds\nminx = bounds[0]-buffer_m\nminy = bounds[1]-buffer_m\nmaxx = bounds[2]+buffer_m\nmaxy = bounds[3]+buffer_m\nwidth = int(np.ceil((maxx-minx)/cell_m))\nheight = int(np.ceil((maxy-miny)/cell_m))\ntransform = from_bounds(minx, miny, maxx, maxy, width, height)\nprint(f'Grid: {width}x{height} = {width*height:,} cells')\nx_coords = np.linspace(minx+cell_m/2, maxx-cell_m/2, width)\ny_coords = np.linspace(miny+cell_m/2, maxy-cell_m/2, height)\nX, Y = np.meshgrid(x_coords, y_coords)"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "min_t = tweets['time'].min()\nmax_t = tweets['time'].max()\nstart_t = min_t.floor(f\"{CONFIG['time_bin_hours']}h\")\nend_t = max_t.ceil(f\"{CONFIG['time_bin_hours']}h\")\ntime_bins = pd.date_range(start_t, end_t, freq=f\"{CONFIG['time_bin_hours']}h\")\ntweets['bin'] = pd.cut(tweets['time'], bins=time_bins, labels=range(len(time_bins)-1), include_lowest=True)\nprint(f'{len(time_bins)-1} bins of {CONFIG[\"time_bin_hours\"]}h each')\nbin_meta = pd.DataFrame({'idx': range(len(time_bins)-1), 'start': time_bins[:-1], 'end': time_bins[1:]})"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "def fuse_intensity(tweet_subset, grid_x, grid_y, sigma_m=3000):\n    intensity = np.zeros(grid_x.shape, dtype=np.float32)\n    for idx, row in tweet_subset.iterrows():\n        if row.geometry and row.geometry.is_valid:\n            px, py = row.geometry.x, row.geometry.y\n            dist_sq = (grid_x - px)**2 + (grid_y - py)**2\n            kernel = np.exp(-dist_sq / (2*sigma_m**2))\n            intensity += kernel\n    if intensity.max() > 0:\n        intensity = intensity / intensity.max()\n    return intensity\n\nprint('Fusion ready')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "print('Generating slices...')\nslices_iter = []\nslices_cum = []\nfor idx, row in bin_meta.iterrows():\n    tweets_in_bin = tweets[tweets['bin'] == idx]\n    tweets_cumulative = tweets[tweets['bin'] <= idx]\n    print(f'Bin {idx}: {len(tweets_in_bin)} tweets')\n    if len(tweets_in_bin) > 0:\n        intensity_i = fuse_intensity(tweets_in_bin, X, Y)\n    else:\n        intensity_i = np.zeros((height, width), dtype=np.float32)\n    if len(tweets_cumulative) > 0:\n        intensity_c = fuse_intensity(tweets_cumulative, X, Y)\n    else:\n        intensity_c = np.zeros((height, width), dtype=np.float32)\n    slices_iter.append(intensity_i)\n    slices_cum.append(intensity_c)\n    if CONFIG['export_slices']:\n        for name, data in [('iterative', intensity_i), ('cumulative', intensity_c)]:\n            path = EVENT_DIR / f'slice_{idx:03d}_{name}.tif'\n            with rasterio.open(path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=np.float32, crs=CONFIG['crs'], transform=transform, compress='lzw') as dst:\n                dst.write(data, 1)\n                dst.update_tags(time_start=str(row['start']), time_end=str(row['end']))\nprint(f'{len(slices_iter)} slices generated')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "meta = {\n    'pipeline': 'Raster Fusion',\n    'timestamp': datetime.now().isoformat(),\n    'config': CONFIG,\n    'grid': {'width': width, 'height': height},\n    'slices': len(slices_iter)\n}\nprint(json.dumps(meta, indent=2))\nwith open(EVENT_DIR / 'metadata.json', 'w') as f:\n    json.dump(meta, f, indent=2)\nprint(f'Metadata saved')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "print('VERIFICATION:')\nassert len(slices_iter) == len(bin_meta)\nprint('  ✓ Count OK')\nassert slices_iter[0].shape == (height, width)\nprint('  ✓ Shape OK')\nprint(f'  ✓ Slices: {sum(1 for s in slices_iter if s.max() > 0)}/{len(slices_iter)} non-empty')\nprint(f'\\nOUTPUTS: {EVENT_DIR}')\nprint('COMPLETE')"
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## README\n\n### Fusion Strategy\nGaussian kernel density from tweet coordinates. Normalized intensity [0,1].\n\n### ArcGIS Pro\n1. Add GeoTIFF slices\n2. Enable time on layer\n3. Use time slider\n4. Symbology: Stretched, heat colors\n\n### Files\n- `slice_NNN_iterative.tif`\n- `slice_NNN_cumulative.tif`\n- `metadata.json`"
    }
]

nb["cells"] = cells

# Write
output_path = Path(r'C:\users\colto\documents\github\tweet_project\hurricane_raster_fusion.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Created: {output_path}')
print(f'Size: {output_path.stat().st_size} bytes')
