"""
FIXED DATA LOADING FUNCTION FOR CELL 6
Copy and paste this into your notebook to replace Cell 6
"""

def load_tweets_geojson(workspace="in_memory"):
    """
    Load helene.geojson tweets as feature class (FIXED for in_memory)
    Returns: Feature class path
    """
    print("Loading tweets from helene.geojson...")
    geojson_path = get_data_file_path('data', 'geojson', 'helene.geojson')
    print(f"  Path: {geojson_path}")

    # FIX: JSONToFeatures doesn't work with in_memory, use temp GDB first
    if workspace == "in_memory":
        # Create temp feature class in scratch GDB
        tweets_fc_temp = arcpy.management.CreateScratchName("tweets_helene", "", "FeatureClass", arcpy.env.scratchGDB)
        arcpy.conversion.JSONToFeatures(geojson_path, tweets_fc_temp)

        # Check spatial reference and project if needed
        sr = arcpy.Describe(tweets_fc_temp).spatialReference
        if sr.factoryCode != 4326:
            tweets_fc_proj = arcpy.management.CreateScratchName("tweets_helene_wgs84", "", "FeatureClass", arcpy.env.scratchGDB)
            arcpy.management.Project(tweets_fc_temp, tweets_fc_proj, arcpy.SpatialReference(4326))
            arcpy.management.Delete(tweets_fc_temp)
            tweets_fc_temp = tweets_fc_proj

        # Copy to in_memory
        tweets_fc = os.path.join(workspace, "tweets_helene")
        arcpy.management.CopyFeatures(tweets_fc_temp, tweets_fc)
        arcpy.management.Delete(tweets_fc_temp)
    else:
        # Use workspace directly
        tweets_fc = os.path.join(workspace, "tweets_helene")
        arcpy.conversion.JSONToFeatures(geojson_path, tweets_fc)

        # Project to WGS84 if needed
        sr = arcpy.Describe(tweets_fc).spatialReference
        if sr.factoryCode != 4326:
            tweets_fc_wgs84 = os.path.join(workspace, "tweets_helene_wgs84")
            arcpy.management.Project(tweets_fc, tweets_fc_wgs84, arcpy.SpatialReference(4326))
            arcpy.management.Delete(tweets_fc)
            tweets_fc = tweets_fc_wgs84

    count = int(arcpy.management.GetCount(tweets_fc).getOutput(0))
    print(f"  ✓ Loaded {count} tweet features")

    return tweets_fc


def load_cities_csv(workspace="in_memory"):
    """
    Load cities1000.csv and convert to point feature class
    Returns: Feature class path
    """
    print("Loading cities from CSV...")
    csv_path = get_data_file_path('data', 'tables', 'cities1000.csv')
    print(f"  Path: {csv_path}")

    cities_fc = os.path.join(workspace, "us_cities")

    try:
        arcpy.management.MakeXYEventLayer(
            csv_path,
            "longitude",
            "latitude",
            "cities_layer",
            arcpy.SpatialReference(4326)
        )

        arcpy.management.SelectLayerByAttribute(
            "cities_layer",
            "NEW_SELECTION",
            "country_code = 'US' AND feature_class = 'P' AND population IS NOT NULL"
        )

        arcpy.management.CopyFeatures("cities_layer", cities_fc)
        arcpy.management.Delete("cities_layer")

    except Exception as e:
        print(f"  Error loading cities: {e}")
        raise

    count = int(arcpy.management.GetCount(cities_fc).getOutput(0))
    print(f"  ✓ Loaded {count} US city features")

    return cities_fc


def load_states_shapefile(workspace="in_memory"):
    """
    Load states shapefile and project to WGS84
    Returns: Feature class path
    """
    print("Loading states shapefile...")
    shp_path = get_data_file_path('data', 'shape_files', 'cb_2023_us_state_20m.shp')
    print(f"  Path: {shp_path}")

    states_fc = os.path.join(workspace, "us_states")

    try:
        arcpy.management.Project(shp_path, states_fc, arcpy.SpatialReference(4326))
    except Exception as e:
        print(f"  Error loading states: {e}")
        raise

    count = int(arcpy.management.GetCount(states_fc).getOutput(0))
    print(f"  ✓ Loaded {count} state features")

    return states_fc


def load_counties_shapefile(workspace="in_memory"):
    """
    Load counties shapefile and project to WGS84
    Returns: Feature class path
    """
    print("Loading counties shapefile...")
    shp_path = get_data_file_path('data', 'shape_files', 'cb_2023_us_county_20m.shp')
    print(f"  Path: {shp_path}")

    counties_fc = os.path.join(workspace, "us_counties")

    try:
        arcpy.management.Project(shp_path, counties_fc, arcpy.SpatialReference(4326))
    except Exception as e:
        print(f"  Error loading counties: {e}")
        raise

    count = int(arcpy.management.GetCount(counties_fc).getOutput(0))
    print(f"  ✓ Loaded {count} county features")

    return counties_fc


print("✓ Data loading functions defined (FIXED for in_memory).")
