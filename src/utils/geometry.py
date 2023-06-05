from shapely.geometry import LineString, MultiLineString
from shapely.wkb import dumps, loads

def from_3D_to_2D(geom):
    return loads(dumps(geom, output_dimension=2))


def convert_to_linestring(geometry):
    if isinstance(geometry, LineString):
        return geometry

    if isinstance(geometry, MultiLineString):
        if len(geometry.geoms) != 1:
            raise Exception(
                'MultiLineString has more than 1 geometry and cannot be safely converted to LineString')
        return geometry.geoms[0]

    raise Exception('Not a LineString or MultiLineString')
