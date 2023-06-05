import os
import geopandas as gpd

"""
Function creates a zone.geojson which contains a zone defined by a GeoJSON file containing the boundary and if provided by other GeoJSON's some holes.
"""

if __name__ == '__main__':
    folder = 'data/managed_data_files/shapefiles/modelling_zones/antwerp'

    # the boundary is the exterior shape and the holes that we are going to cut out
    # are called the exclusion zones
    boundary_filepath = os.path.join(folder, 'boundary/boundary.geojson')
    exclusion_zones_folder = os.path.join(folder, 'exclusion_zones')
    result_filepath = os.path.join(folder, 'zone.geojson')

    # let's load our boundary
    boundary = gpd.read_file(boundary_filepath).unary_union

    # let's load all the exclusion zones
    exclusion_zones = []
    for root, dirs, files in os.walk(exclusion_zones_folder):
        for file in files:
            gdf = gpd.read_file(os.path.join(root, file))
            exclusion_zones.extend(gdf.geometry.tolist())

    # let's create the resulting polygon
    zone = boundary
    for ez in exclusion_zones:
        zone = zone.difference(ez)

    gdf = gpd.GeoDataFrame(geometry=[zone])
    gdf.to_file(result_filepath, driver='GeoJSON')
