"""
ALTERNATIVE APPROACH: Load GeoJSON as table, then convert to points
This avoids the JSONToFeatures + in_memory issue entirely
"""

import json

def load_tweets_geojson(workspace="in_memory"):
    """
    Load helene.geojson by reading JSON and creating points from coordinates
    Returns: Feature class path
    """
    print("Loading tweets from helene.geojson...")
    geojson_path = get_data_file_path('data', 'geojson', 'helene.geojson')
    print(f"  Path: {geojson_path}")

    # Read GeoJSON file
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    # Create feature class
    tweets_fc = os.path.join(workspace, "tweets_helene")

    # Create empty point feature class
    arcpy.management.CreateFeatureclass(
        os.path.dirname(tweets_fc) if workspace != "in_memory" else workspace,
        os.path.basename(tweets_fc),
        "POINT",
        spatial_reference=arcpy.SpatialReference(4326)
    )

    # Add fields for attributes
    arcpy.management.AddField(tweets_fc, "GPE", "TEXT", field_length=500)
    arcpy.management.AddField(tweets_fc, "FAC", "TEXT", field_length=500)
    arcpy.management.AddField(tweets_fc, "LOC", "TEXT", field_length=500)
    arcpy.management.AddField(tweets_fc, "time", "TEXT", field_length=50)
    arcpy.management.AddField(tweets_fc, "Latitude", "DOUBLE")
    arcpy.management.AddField(tweets_fc, "Longitude", "DOUBLE")

    # Insert features
    fields = ['SHAPE@XY', 'GPE', 'FAC', 'LOC', 'time', 'Latitude', 'Longitude']

    with arcpy.da.InsertCursor(tweets_fc, fields) as cursor:
        for feature in geojson_data['features']:
            props = feature['properties']
            coords = feature['geometry']['coordinates']

            # GeoJSON is [lon, lat]
            lon = coords[0]
            lat = coords[1]

            cursor.insertRow([
                (lon, lat),  # SHAPE@XY
                props.get('GPE', ''),
                props.get('FAC', ''),
                props.get('LOC', ''),
                props.get('time', ''),
                lat,
                lon
            ])

    count = int(arcpy.management.GetCount(tweets_fc).getOutput(0))
    print(f"  ✓ Loaded {count} tweet features")

    return tweets_fc


def load_cities_csv(workspace="in_memory"):
    """Load cities1000.csv and convert to point feature class"""
    print("Loading cities from CSV...")
    csv_path = get_data_file_path('data', 'tables', 'cities1000.csv')
    print(f"  Path: {csv_path}")

    cities_fc = os.path.join(workspace, "us_cities")

    arcpy.management.MakeXYEventLayer(
        csv_path, "longitude", "latitude", "cities_layer",
        arcpy.SpatialReference(4326)
    )

    arcpy.management.SelectLayerByAttribute(
        "cities_layer", "NEW_SELECTION",
        "country_code = 'US' AND feature_class = 'P' AND population IS NOT NULL"
    )

    arcpy.management.CopyFeatures("cities_layer", cities_fc)
    arcpy.management.Delete("cities_layer")

    count = int(arcpy.management.GetCount(cities_fc).getOutput(0))
    print(f"  ✓ Loaded {count} US city features")

    return cities_fc


def load_states_shapefile(workspace="in_memory"):
    """Load states shapefile and project to WGS84"""
    print("Loading states shapefile...")
    shp_path = get_data_file_path('data', 'shape_files', 'cb_2023_us_state_20m.shp')
    print(f"  Path: {shp_path}")

    states_fc = os.path.join(workspace, "us_states")
    arcpy.management.Project(shp_path, states_fc, arcpy.SpatialReference(4326))

    count = int(arcpy.management.GetCount(states_fc).getOutput(0))
    print(f"  ✓ Loaded {count} state features")

    return states_fc


def load_counties_shapefile(workspace="in_memory"):
    """Load counties shapefile and project to WGS84"""
    print("Loading counties shapefile...")
    shp_path = get_data_file_path('data', 'shape_files', 'cb_2023_us_county_20m.shp')
    print(f"  Path: {shp_path}")

    counties_fc = os.path.join(workspace, "us_counties")
    arcpy.management.Project(shp_path, counties_fc, arcpy.SpatialReference(4326))

    count = int(arcpy.management.GetCount(counties_fc).getOutput(0))
    print(f"  ✓ Loaded {count} county features")

    return counties_fc


print("✓ Data loading functions defined (using JSON parser - works with in_memory!).")
