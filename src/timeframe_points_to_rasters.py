import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
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
    if points_gdf.empty:
        return np.zeros((height, width), dtype="float32")

    density = np.zeros((height, width), dtype="float32")

    minx, miny, maxx, maxy = bounds_proj
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height

    coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])
    values = points_gdf[value_field].to_numpy(dtype="float32", copy=False)

    cols = ((coords[:, 0] - minx) / pixel_width).astype(int)
    rows = ((maxy - coords[:, 1]) / pixel_height).astype(int)

    valid_mask = (
        (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )

    rows = rows[valid_mask]
    cols = cols[valid_mask]
    values = values[valid_mask]

    np.add.at(density, (rows, cols), values)

    sigma_meters = cell_size_m * float(sigma_factor)
    sigma_pixels = max(1.0, sigma_meters / max(pixel_width, pixel_height))
    density = gaussian_filter(density, sigma=sigma_pixels, mode="constant", cval=0.0)

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