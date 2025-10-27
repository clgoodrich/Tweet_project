import os
from scipy.ndimage import gaussian_filter
from rasterio.features import rasterize
from rasterio.features import geometry_mask
# ==============================================================================
# STEP 2: MAIN RASTERIZATION LOOP - TIME ITERATION
# ==============================================================================

# Create output directories



def create_hierarchical_rasters(data, grid_params, time_bin):
    """Create hierarchically weighted rasters with automatic parent state inclusion"""
    print(f"    Creating hierarchical raster for time {time_bin}...")

    output_grid = np.zeros((grid_params['height'], grid_params['width']), dtype=np.float32)
    states_to_include = set()  # Track which states need base layers

    # 1. First pass: identify all states that need base layers
    state_data = data[data['scale_level'] == 'STATE']
    if len(state_data) > 0:
        states_to_include.update(state_data['matched_name'].unique())

    # Check counties - add their parent states
    county_data = data[data['scale_level'] == 'COUNTY']
    for county_name in county_data['matched_name'].unique():
        if county_name in county_lookup_proj:
            # Find parent state by spatial containment
            county_geom = county_lookup_proj[county_name]
            for state_name, state_geom in state_lookup_proj.items():
                if state_geom.contains(county_geom.centroid):
                    states_to_include.add(state_name)
                    break

    # Check cities - add their parent states
    city_data = data[data['scale_level'] == 'CITY']
    for city_name in city_data['matched_name'].unique():
        if city_name in cities_lookup_proj:
            city_geom = cities_lookup_proj[city_name]
            for state_name, state_geom in state_lookup_proj.items():
                if state_geom.contains(city_geom.centroid):
                    states_to_include.add(state_name)
                    break

    # 2. Rasterize all states that need inclusion
    for state_name in states_to_include:
        if state_name in state_lookup_proj:
            state_geom = state_lookup_proj[state_name]
            mask = rasterize(
                [(state_geom, 1)],
                out_shape=(grid_params['height'], grid_params['width']),
                transform=grid_params['transform'],
                fill=0, dtype=np.float32, all_touched=True
            )

            # Get tweet count if state was mentioned, else use minimal base
            if state_name in state_data['matched_name'].values:
                tweet_count = state_data[state_data['matched_name'] == state_name]['count'].sum()
            else:
                tweet_count = 1  # Minimal base for implied states

            base_value = np.log1p(tweet_count) * 2
            output_grid += mask * base_value

    # 3. Add counties (same as before)
    if len(county_data) > 0:
        county_counts = county_data.groupby('matched_name')['count'].sum()
        for county_name, tweet_count in county_counts.items():
            if county_name in county_lookup_proj:
                mask = rasterize(
                    [(county_lookup_proj[county_name], 1)],
                    out_shape=(grid_params['height'], grid_params['width']),
                    transform=grid_params['transform'],
                    fill=0, dtype=np.float32, all_touched=True
                )
                output_grid += mask * np.log1p(tweet_count) * 5

    # 4. Add cities (same as before)
    if len(city_data) > 0:
        city_counts = city_data.groupby('matched_name')['count'].sum()
        for city_name, tweet_count in city_counts.items():
            if city_name in cities_lookup_proj:
                mask = rasterize(
                    [(cities_lookup_proj[city_name], 1)],
                    out_shape=(grid_params['height'], grid_params['width']),
                    transform=grid_params['transform'],
                    fill=0, dtype=np.float32, all_touched=True
                )
                output_grid += mask * np.log1p(tweet_count) * 10

    # 5. Add facilities
    facility_data = data[data['scale_level'] == 'FACILITY']
    if len(facility_data) > 0:
        output_grid += create_facility_raster(data, grid_params)

    return output_grid

def process_hurricane(hurricane_name, gdf_proj, interval_counts, time_bins, timestamp_dict, grid_params):
    """
    Process a single hurricane through all time bins
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {hurricane_name.upper()}")
    print(f"{'=' * 60}")
    print()
    print(gdf_proj)
    # Create hurricane-specific output directory
    hurricane_dir = os.path.join(output_dir, hurricane_name.lower())
    os.makedirs(hurricane_dir, exist_ok=True)

    # Initialize cumulative grid (persists across time bins)
    cumulative_grid = np.zeros((grid_params['height'], grid_params['width']), dtype=np.float32)

    # Loop through each time bin chronologically
    for idx, time_bin in enumerate(time_bins):
        # print(f"\n--- Time Bin {idx+1}/{len(time_bins)}: {time_bin} ---")

        # Filter data for current time bin
        current_data = interval_counts[interval_counts['unix_timestamp'] == time_bin]
        tweet_count = len(current_data)
        # print(f"  Tweets in this bin: {tweet_count}")

        # WITH THIS:
        incremental_grid = create_hierarchical_rasters(current_data, grid_params, time_bin)

        # === END PLACEHOLDERS ===

        # Update cumulative grid
        cumulative_grid += incremental_grid
        # Save rasters
        save_raster(incremental_grid, hurricane_dir, hurricane_name, time_bin, 'increment', timestamp_dict)
        save_raster(cumulative_grid, hurricane_dir, hurricane_name, time_bin, 'cumulative', timestamp_dict)

        print(f"  Incremental max value: {np.max(incremental_grid):.2f}")
        print(f"  Cumulative max value: {np.max(cumulative_grid):.2f}")

    print(f"\n{hurricane_name.upper()} processing complete!")
    print(f"Output saved to: {hurricane_dir}")
    return

# ==============================================================================
# PLACEHOLDER FUNCTIONS (TO BE IMPLEMENTED)
# ==============================================================================

def create_facility_raster(data, grid_params):
    """Create KDE raster for facility points with strong hotspot multiplier"""
    print("    [FACILITY] Creating facility raster...")

    # Initialize empty raster
    facility_grid = np.zeros((grid_params['height'], grid_params['width']), dtype=np.float32)

    # Filter for FACILITY-level tweets only
    facility_data = data[data['scale_level'] == 'FACILITY']

    if len(facility_data) == 0:
        print("      No facility-level tweets in this time bin")
        return facility_grid

    # Group by facility coordinates (using matched_name as proxy) and sum counts
    facility_counts = facility_data.groupby('matched_name')['count'].sum()

    print(f"      Processing {len(facility_counts)} unique facilities")

    # HOTSPOT PARAMETERS for facilities
    sigma_meters = 2 * grid_params['cell_size']  # 10 km for 5km cells
    sigma_pixels = sigma_meters / grid_params['cell_size']  # Convert to pixel units
    facility_multiplier = 10  # Make facilities 10x more prominent (strongest hotspots)

    # Process each facility
    facilities_processed = 0
    for facility_name, tweet_count in facility_counts.items():
        # Get facility data to extract geometry
        facility_rows = facility_data[facility_data['matched_name'] == facility_name]

        if len(facility_rows) > 0:
            # Get the point geometry (should be from the tweet's geocoded location)
            facility_point = facility_rows.iloc[0]['matched_geom']

            # Project point to grid CRS if needed
            if hasattr(facility_point, 'x') and hasattr(facility_point, 'y'):
                # Create GeoSeries to handle projection
                point_geoseries = gpd.GeoSeries([facility_point], crs='EPSG:4326')
                point_proj = point_geoseries.to_crs(grid_params['crs']).iloc[0]

                # Convert point coordinates to pixel indices
                px = (point_proj.x - grid_params['bounds'][0]) / grid_params['cell_size']
                py = (grid_params['bounds'][3] - point_proj.y) / grid_params['cell_size']

                # Check if point is within grid bounds
                if 0 <= px < grid_params['width'] and 0 <= py < grid_params['height']:
                    # Create point raster with tweet count at location
                    point_grid = np.zeros((grid_params['height'], grid_params['width']), dtype=np.float32)
                    point_grid[int(py), int(px)] = tweet_count

                    # Apply Gaussian filter to create kernel density
                    kernel_grid = gaussian_filter(point_grid, sigma=sigma_pixels, mode='constant', cval=0)

                    # FIXED: Only add once with proper multiplier
                    facility_grid += kernel_grid * facility_multiplier

                    facilities_processed += 1
                    effective_value = tweet_count * facility_multiplier
                else:
                    print(f"      WARNING: Facility '{facility_name}' outside grid bounds")
            else:
                print(f"      WARNING: Invalid geometry for facility '{facility_name}'")

    print(f"      Processed {facilities_processed} facilities with sigma={sigma_pixels:.2f} pixels")

    total_value = np.sum(facility_grid)
    max_value = np.max(facility_grid)
    # print(f"      Total facility grid value: {total_value:.2f}, Max pixel: {max_value:.2f}")

    return facility_grid

def save_raster(grid, output_dir, hurricane_name, time_bin, raster_type, timestamp_dict):
    """Save raster as GeoTIFF in type-specific subdirectory"""
    # Create subdirectory for raster type
    type_dir = os.path.join(output_dir, raster_type)
    os.makedirs(type_dir, exist_ok=True)
    print('max grid', np.max(grid))
    # Convert unix timestamp (microseconds) back to datetime
    time_str = timestamp_dict[time_bin].strftime('%Y%m%d_%H%M%S')
    # time_str = pd.Timestamp(time_bin, unit='us').strftime('%Y%m%d_%H%M%S')
    print([time_str])
    filename = f"{hurricane_name}_tweets_{time_str}.tif"
    filepath = os.path.join(type_dir, filename)
    print(grid_params)
    with rasterio.open(
        filepath, 'w',
        driver='GTiff',
        height=grid_params['height'],
        width=grid_params['width'],
        count=1,
        dtype=grid.dtype,
        crs=grid_params['crs'],
        transform=grid_params['transform'],
        compress='lzw'
    ) as dst:
        dst.write(grid, 1)

    print(f"    Saved: {raster_type}/{filename}")

def rasterize_process(proj, interval_counts, time_bines, time_stamp_dict, local_path, grid_params):
    rasters_dir = r"\rasters_output"
    output_dir = f"{local_path}{rasters_dir}"
    # output_dir = os.path.join(local_path, 'rasters_output')
    # output_dir = r"C:\Users\colto\Documents\GitHub\Tweet_project\rasters_output"
    os.makedirs(output_dir, exist_ok=True)
    # Process Francine
    process_hurricane('francine', proj, interval_counts, time_bines, time_stamp_dict, grid_params)

    # process_hurricane('francine', francine_proj, francine_interval_counts, francine_time_bins, francine_timestamp_dict)

    # Process Helene
    # process_hurricane('helene', helene_proj, helene_interval_counts, helene_time_bins, helene_timestamp_dict)
