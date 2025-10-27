import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os

# Configuration
base_path = r"C:\Users\colto\Documents\GitHub\Tweet_project"
output_base = r"C:\Users\colto\Documents\GitHub\Tweet_project\rasters_all"
cell_size_m = 5000  # 5km in meters

# Dataset configurations
datasets = [
    {'name': 'helene_freq', 'file': 'helene_cities_freq_over_time.shp', 'value_field': 'tweets'},
    {'name': 'helene_cumulative', 'file': 'helene_cities_cumulative.shp', 'value_field': 'cum'},
    {'name': 'francine_freq', 'file': 'francine_cities_freq_over_time.shp', 'value_field': 'tweets'},
    {'name': 'francine_cumulative', 'file': 'francine_cities_cumulative.shp', 'value_field': 'cum'}
]


def create_kernel_density(points_gdf, value_field, width, height, bounds_proj, cell_size_m, sigma_factor=2):
    """Create kernel density raster from points"""
    coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])
    values = points_gdf[value_field].values

    if len(coords) == 0:
        return np.zeros((height, width))

    x_coords = np.linspace(bounds_proj[0], bounds_proj[2], width)
    y_coords = np.linspace(bounds_proj[1], bounds_proj[3], height)
    xx, yy = np.meshgrid(x_coords, y_coords[::-1])

    density = np.zeros((height, width))
    sigma = cell_size_m * sigma_factor

    for (px, py), weight in zip(coords, values):
        dist_sq = (xx - px) ** 2 + (yy - py) ** 2
        kernel = weight * np.exp(-dist_sq / (2 * sigma ** 2))
        density += kernel

    density = density / (2 * np.pi * sigma ** 2)
    return density


# Process each dataset
for dataset in datasets:
    print(f"\n{'=' * 50}")
    print(f"Processing: {dataset['name']}")
    print('=' * 50)

    # Create output folder
    output_folder = os.path.join(output_base, dataset['name'])
    os.makedirs(output_folder, exist_ok=True)

    # Load shapefile
    input_file = os.path.join(base_path, dataset['file'])
    print(f"Loading: {input_file}")
    gdf = gpd.read_file(input_file)

    # Get unique time periods
    time_periods = sorted(gdf['time_str'].unique())
    print(f"Found {len(time_periods)} time periods")
#
    # Project and get bounds
    gdf_proj = gdf.to_crs(epsg=3857)
    bounds_proj = gdf_proj.total_bounds
    minx_proj, miny_proj, maxx_proj, maxy_proj = bounds_proj

    # Calculate grid dimensions
    width = int(np.ceil((maxx_proj - minx_proj) / cell_size_m))
    height = int(np.ceil((maxy_proj - miny_proj) / cell_size_m))
    print(f"Grid dimensions: {width} x {height} cells")

    # Process each time period
    for i, time_period in enumerate(time_periods):
        print(f"  Processing {i + 1}/{len(time_periods)}: {time_period}")

        time_data = gdf_proj[gdf_proj['time_str'] == time_period]

        if len(time_data) == 0:
            continue

        # Create density raster
        density = create_kernel_density(time_data, dataset['value_field'], width, height, bounds_proj, cell_size_m)

        # Create transform
        transform = from_bounds(minx_proj, miny_proj, maxx_proj, maxy_proj, width, height)

        # Save as GeoTIFF
        safe_time = time_period.replace(":", "").replace(" ", "_").replace("-", "")
        output_path = os.path.join(output_folder, f"{dataset['name']}_{safe_time}.tif")

        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=density.dtype,
                crs='EPSG:3857',
                transform=transform,
                compress='lzw'
        ) as dst:
            dst.write(density, 1)

    print(f"  Complete! Saved to: {output_folder}")

print(f"\n{'=' * 50}")
print("ALL DATASETS COMPLETE")
print('=' * 50)
print(f"Output location: {output_base}")
print("\nFolders created:")
for dataset in datasets:
    print(f"  - {dataset['name']}")