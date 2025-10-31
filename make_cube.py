import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "# Space Time Cube for ArcGIS Pro\n\nProper CF-compliant NetCDF with time encoding ArcGIS can read."
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "event = 'helene'\ncell_km = 10\ntime_hours = 4\nprint(f'{event}, {cell_km}km, {time_hours}h')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "import geopandas as gpd\nimport pandas as pd\nimport numpy as np\nfrom netCDF4 import Dataset, date2num\nfrom pathlib import Path\nimport warnings, json\nwarnings.filterwarnings('ignore')\nprint('Ready')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "data = Path(r'C:\\Users\\colto\\Documents\\GitHub\\Tweet_project\\data')\ntweets = gpd.read_file(data / 'geojson' / f'{event}.geojson')\ntweets['time'] = pd.to_datetime(tweets['time'])\nif tweets.crs is None:\n    tweets.set_crs('EPSG:4326', inplace=True)\ntweets = tweets.to_crs('EPSG:5070')\ntweets['x'] = tweets.geometry.x\ntweets['y'] = tweets.geometry.y\nprint(f'{len(tweets)} tweets')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "cell_m = cell_km * 1000\nxmin, ymin, xmax, ymax = tweets.total_bounds\nxmin = np.floor(xmin / cell_m) * cell_m\nymin = np.floor(ymin / cell_m) * cell_m\nxmax = np.ceil(xmax / cell_m) * cell_m\nymax = np.ceil(ymax / cell_m) * cell_m\nx_edges = np.arange(xmin, xmax + cell_m, cell_m)\ny_edges = np.arange(ymin, ymax + cell_m, cell_m)\nx_centers = (x_edges[:-1] + x_edges[1:]) / 2\ny_centers = (y_edges[:-1] + y_edges[1:]) / 2\nprint(f'Grid: {len(x_edges)-1} x {len(y_edges)-1}')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "tmin = tweets.time.min().floor(f'{time_hours}h')\ntmax = tweets.time.max().ceil(f'{time_hours}h')\ntime_edges = pd.date_range(tmin, tmax, freq=f'{time_hours}h')\ntime_centers = time_edges[:-1] + (time_edges[1:] - time_edges[:-1]) / 2\nprint(f'Time: {len(time_edges)-1} bins')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "tweets['x_bin'] = pd.cut(tweets.x, bins=x_edges, labels=False, include_lowest=True)\ntweets['y_bin'] = pd.cut(tweets.y, bins=y_edges, labels=False, include_lowest=True)\ntweets['t_bin'] = pd.cut(tweets.time, bins=time_edges, labels=False, include_lowest=True)\ntweets_binned = tweets.dropna(subset=['x_bin', 'y_bin', 't_bin']).copy()\ntweets_binned[['x_bin', 'y_bin', 't_bin']] = tweets_binned[['x_bin', 'y_bin', 't_bin']].astype(int)\nprint(f'{len(tweets_binned)} binned')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "nx = len(x_edges) - 1\nny = len(y_edges) - 1\nnt = len(time_edges) - 1\ncube = np.zeros((nt, ny, nx), dtype=np.int32)\nfor _, row in tweets_binned.iterrows():\n    cube[row.t_bin, row.y_bin, row.x_bin] += 1\nprint(f'Cube: {nt}t x {ny}y x {nx}x = {cube.sum()} tweets')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "out = Path(r'C:\\Users\\colto\\Documents\\GitHub\\Tweet_project\\spacetimecube_output')\nout.mkdir(exist_ok=True)\nncfile = out / f'{event}_cube.nc'\nwith Dataset(ncfile, 'w', format='NETCDF4') as nc:\n    nc.createDimension('x', nx)\n    nc.createDimension('y', ny)\n    nc.createDimension('time', nt)\n    x_var = nc.createVariable('x', 'f8', ('x',))\n    x_var[:] = x_centers\n    x_var.units = 'meters'\n    x_var.standard_name = 'projection_x_coordinate'\n    y_var = nc.createVariable('y', 'f8', ('y',))\n    y_var[:] = y_centers\n    y_var.units = 'meters'\n    y_var.standard_name = 'projection_y_coordinate'\n    t_var = nc.createVariable('time', 'f8', ('time',))\n    time_units = 'hours since 1970-01-01 00:00:00'\n    t_var.units = time_units\n    t_var.calendar = 'gregorian'\n    t_var.standard_name = 'time'\n    t_var[:] = date2num(time_centers.to_pydatetime(), units=time_units, calendar='gregorian')\n    count_var = nc.createVariable('COUNT', 'i4', ('time', 'y', 'x'), zlib=True, complevel=4)\n    count_var[:] = cube\n    count_var.long_name = 'Tweet count'\n    count_var.units = 'count'\n    nc.Conventions = 'CF-1.6'\n    nc.title = f'{event.title()} Space Time Cube'\nprint(f'Saved: {ncfile}')\nprint(f'Size: {ncfile.stat().st_size/1024:.1f} KB')"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "summary = {'file': str(ncfile), 'event': event, 'dims': [nt, ny, nx], 'tweets': int(cube.sum())}\nprint(json.dumps(summary, indent=2))\nprint('\\nArcGIS Pro: Insert → Add Multidimensional Raster → Select .nc → Choose COUNT variable')"
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output = Path(r'C:\users\colto\documents\github\tweet_project\create_spacetimecube.ipynb')
with open(output, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Created: {output}')
