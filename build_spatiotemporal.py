import json
from pathlib import Path

# Build minimal clean notebook
cells = []

# Title
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": "# Space-Time Raster Fusion\n\n**PLAN:** Load → Join → Bin → Fuse → Export\n\n**DO:** Execute\n\n**VERIFY:** Check outputs"
})

# Config
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": "CONFIG = {'event': 'helene', 'cell_size_km': 5, 'time_bin_hours': 2, 'crs': 'EPSG:5070', 'weights': {'coordinates': 0.60, 'county': 0.25, 'state': 0.15}}\nassert abs(sum(CONFIG['weights'].values()) - 1.0) < 0.001\nprint(f\"Event: {CONFIG['event']}, {CONFIG['cell_size_km']}km, {CONFIG['time_bin_hours']}h\")"
})

# Imports
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": "import geopandas as gpd\nimport pandas as pd\nimport numpy as np\nimport rasterio\nfrom rasterio.transform from_bounds\nfrom rasterio.features import rasterize\nfrom pathlib import Path\nfrom datetime import datetime\nimport json, warnings\nwarnings.filterwarnings('ignore')\nprint('Ready')"
})

# Paths
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": "BASE = Path(r'C:\\Users\\colto\\Documents\\GitHub\\Tweet_project')\nDATA = BASE / 'data'\nOUT = BASE / 'spatiotemporal_output' / CONFIG['event']\nOUT.mkdir(parents=True, exist_ok=True)\nprint(f'Output: {OUT}')"
})

# Load
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": "tweets = gpd.read_file(DATA / 'geojson' / f\"{CONFIG['event']}.geojson\")\ntweets['time'] = pd.to_datetime(tweets['time'])\nif tweets.crs is None:\n    tweets.set_crs('EPSG:4326', inplace=True)\ntweets = tweets.to_crs(CONFIG['crs'])\nstates = gpd.read_file(DATA / 'shape_files' / 'cb_2023_us_state_20m.shp').to_crs(CONFIG['crs'])\ncounties = gpd.read_file(DATA / 'shape_files' / 'cb_2023_us_county_20m.shp').to_crs(CONFIG['crs'])\nprint(f'{len(tweets)} tweets, {len(states)} states, {len(counties)} counties')"
})

# More cells would go here but let me simplify even further
# Actually, let me just copy the working raster fusion notebook and modify it

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = Path(r'C:\users\colto\documents\github\tweet_project\spatiotemporal_raster.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Created: {output_path}')
