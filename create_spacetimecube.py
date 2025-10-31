import json
from pathlib import Path

# Simple Space Time Cube notebook
nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Space Time Cube Generator\n\nCreates ArcGIS-compatible Space Time Cube (NetCDF) from hurricane tweets."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Configuration\nevent = 'helene'  # or 'francine'\ncell_size_km = 10  # Grid cell size\ntime_step_hours = 4  # Time bin size\n\nprint(f'Creating Space Time Cube for {event}')\nprint(f'Cell: {cell_size_km}km, Time: {time_step_hours}h')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "import geopandas as gpd\nimport pandas as pd\nimport numpy as np\nfrom netCDF4 import Dataset\nfrom pathlib import Path\nimport warnings\nwarnings.filterwarnings('ignore')\n\nprint('Ready')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Load tweets\ndata_path = Path(r'C:\\Users\\colto\\Documents\\GitHub\\Tweet_project\\data\\geojson')\ntweets = gpd.read_file(data_path / f'{event}.geojson')\ntweets['time'] = pd.to_datetime(tweets['time'])\n\n# Reproject to meters (Albers Equal Area)\nif tweets.crs is None:\n    tweets.set_crs('EPSG:4326', inplace=True)\ntweets = tweets.to_crs('EPSG:5070')\n\ntweets['x'] = tweets.geometry.x\ntweets['y'] = tweets.geometry.y\n\nprint(f'Loaded {len(tweets)} tweets')\nprint(f'Time range: {tweets.time.min()} to {tweets.time.max()}')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Create spatial grid\ncell_m = cell_size_km * 1000\n\nxmin, ymin, xmax, ymax = tweets.total_bounds\nxmin = np.floor(xmin / cell_m) * cell_m\nymin = np.floor(ymin / cell_m) * cell_m\nxmax = np.ceil(xmax / cell_m) * cell_m\nymax = np.ceil(ymax / cell_m) * cell_m\n\nx_bins = np.arange(xmin, xmax + cell_m, cell_m)\ny_bins = np.arange(ymin, ymax + cell_m, cell_m)\n\nprint(f'Grid: {len(x_bins)-1} x {len(y_bins)-1} cells')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Create time bins\ntime_min = tweets.time.min().floor(f'{time_step_hours}h')\ntime_max = tweets.time.max().ceil(f'{time_step_hours}h')\ntime_bins = pd.date_range(time_min, time_max, freq=f'{time_step_hours}h')\n\nprint(f'Time: {len(time_bins)-1} bins of {time_step_hours}h')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Assign to bins\ntweets['x_bin'] = pd.cut(tweets.x, bins=x_bins, labels=False, include_lowest=True)\ntweets['y_bin'] = pd.cut(tweets.y, bins=y_bins, labels=False, include_lowest=True)\ntweets['t_bin'] = pd.cut(tweets.time, bins=time_bins, labels=False, include_lowest=True)\n\n# Drop NaN (tweets outside grid)\ntweets_binned = tweets.dropna(subset=['x_bin', 'y_bin', 't_bin']).copy()\ntweets_binned['x_bin'] = tweets_binned.x_bin.astype(int)\ntweets_binned['y_bin'] = tweets_binned.y_bin.astype(int)\ntweets_binned['t_bin'] = tweets_binned.t_bin.astype(int)\n\nprint(f'{len(tweets_binned)} tweets in grid ({len(tweets_binned)/len(tweets)*100:.1f}%)')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Create 3D cube\nnx = len(x_bins) - 1\nny = len(y_bins) - 1\nnt = len(time_bins) - 1\n\ncube = np.zeros((nt, ny, nx), dtype=np.int32)\n\n# Count tweets per cell\nfor idx, row in tweets_binned.iterrows():\n    cube[row.t_bin, row.y_bin, row.x_bin] += 1\n\nprint(f'Cube shape: {cube.shape} (time, y, x)')\nprint(f'Total tweets in cube: {cube.sum()}')\nprint(f'Non-empty cells: {(cube > 0).sum()}')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Export as NetCDF Space Time Cube\nout_dir = Path(r'C:\\Users\\colto\\Documents\\GitHub\\Tweet_project\\spacetimecube_output')\nout_dir.mkdir(exist_ok=True)\nout_file = out_dir / f'{event}_spacetimecube.nc'\n\nwith Dataset(out_file, 'w', format='NETCDF4') as nc:\n    # Dimensions\n    nc.createDimension('x', nx)\n    nc.createDimension('y', ny)\n    nc.createDimension('time', nt)\n    \n    # Coordinate variables\n    x_var = nc.createVariable('x', 'f8', ('x',))\n    x_var[:] = (x_bins[:-1] + x_bins[1:]) / 2  # Cell centers\n    x_var.units = 'meters'\n    x_var.long_name = 'x coordinate (Albers Equal Area)'\n    \n    y_var = nc.createVariable('y', 'f8', ('y',))\n    y_var[:] = (y_bins[:-1] + y_bins[1:]) / 2\n    y_var.units = 'meters'\n    y_var.long_name = 'y coordinate (Albers Equal Area)'\n    \n    t_var = nc.createVariable('time', 'f8', ('time',))\n    # Store as hours since epoch\n    epoch = pd.Timestamp('1970-01-01')\n    t_var[:] = [(t - epoch).total_seconds() / 3600 for t in time_bins[:-1]]\n    t_var.units = 'hours since 1970-01-01 00:00:00'\n    t_var.calendar = 'gregorian'\n    t_var.long_name = 'time'\n    \n    # Data variable\n    count_var = nc.createVariable('COUNT', 'i4', ('time', 'y', 'x'), \n                                   zlib=True, complevel=4,\n                                   fill_value=-1)\n    count_var[:] = cube\n    count_var.long_name = 'Tweet count per space-time bin'\n    count_var.units = 'count'\n    \n    # Global attributes\n    nc.title = f'{event.title()} Hurricane Space Time Cube'\n    nc.institution = 'Tweet Project'\n    nc.source = 'Twitter/X hurricane data'\n    nc.Conventions = 'CF-1.6'\n    nc.crs = 'EPSG:5070'\n    nc.cell_size_meters = cell_m\n    nc.time_step_hours = time_step_hours\n    nc.creation_date = pd.Timestamp.now().isoformat()\n\nprint(f'\\nSaved: {out_file}')\nprint(f'Size: {out_file.stat().st_size / 1024:.1f} KB')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Summary\nprint('\\n' + '='*60)\nprint('SPACE TIME CUBE CREATED')\nprint('='*60)\nprint(f'File: {out_file.name}')\nprint(f'Event: {event}')\nprint(f'Tweets: {cube.sum()} in {(cube>0).sum()} non-empty cells')\nprint(f'Dimensions: {nx}x × {ny}y × {nt}t')\nprint(f'Resolution: {cell_size_km}km, {time_step_hours}h')\nprint('\\nLoad in ArcGIS Pro:')\nprint('  1. Add Multidimensional Raster Layer')\nprint('  2. Browse to .nc file')\nprint('  3. Select COUNT variable')\nprint('  4. Use Time Slider to animate')"
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = Path(r'C:\users\colto\documents\github\tweet_project\space_time_cube.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Created: {output_path}')
