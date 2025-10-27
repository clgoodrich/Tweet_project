import geodatasets
import geopandas
import matplotlib.pyplot as plt
# gdf_path = get_data_file_path('data', 'shape_files', "cb_2023_us_state_20m.shp")
gdf = geopandas.read_file(r"C:\Users\colto\Documents\GitHub\Tweet_project\data\shape_files\cb_2023_us_state_20m.shp")
ax = gdf.plot(figsize=(10, 10))
print(gdf)