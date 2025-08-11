import os
import geopandas as gpd

def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_gpkg(gdf: gpd.GeoDataFrame, path: str, layer: str):
    ensure_dir_for(path)
    gdf.to_file(path, driver="GPKG", layer=layer)
