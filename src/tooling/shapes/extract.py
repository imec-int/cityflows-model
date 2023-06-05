import argparse
from csv import DictReader, DictWriter
import shapely
import json
from shapely.geometry import shape, GeometryCollection
import shapely.wkt
import csv

OUTPUT_FIELD_NAMES = ['data_source', 'index', 'locationrange']


def extract_shapes(input_file_path, output_file_path, region_shape_file_path=None):

    # keys are (data_source, index)
    processed_shape_keys = set()
    full_geometry = None

    if region_shape_file_path != 'None':
        # extract shape for use
        with open(region_shape_file_path) as f:
            features = json.load(f)["features"]

        # NOTE: buffer(0) is a trick for fixing scenarios where polygons have overlapping coordinates
        full_geometry = GeometryCollection(
            [shape(feature["geometry"]).buffer(0) for feature in features])

    with open(input_file_path, 'r') as input_file:
        with open(output_file_path, 'w') as output_file:
            csv_reader = DictReader(input_file)

            csv_writer = DictWriter(output_file, fieldnames=OUTPUT_FIELD_NAMES)
            csv_writer.writeheader()

            if region_shape_file_path == 'None':
                for row in csv_reader:
                    shape_key = (row['data_source'], row['index'])
                    if shape_key not in processed_shape_keys:
                        csv_writer.writerow({
                            'data_source': row['data_source'],
                            'index': row['index'],
                            'locationrange': row['locationrange'],
                        })
                        processed_shape_keys.add(shape_key)
            else:  # check for each line if it intersects to global full shape
                print(region_shape_file_path)
                for row in csv_reader:
                    local_shape = shapely.wkt.loads(row['locationrange'])
                    if (local_shape.intersects(full_geometry)):
                        shape_key = (row['data_source'], row['index'])
                        if shape_key not in processed_shape_keys:
                            csv_writer.writerow({
                                'data_source': row['data_source'],
                                'index': row['index'],
                                'locationrange': row['locationrange'],
                            })
                            processed_shape_keys.add(shape_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python -m src.tooling.extract_shapes.extract',
        description='Extract all unique shapes from an all_data.csv like file'
    )
    parser.add_argument('--input_file_path', help='Input file path')
    parser.add_argument('--output_file_path', help='Output file path')
    parser.add_argument('--region_shape_file_path',
                        help='Region of modelling geojson file path', default='None')
    args = vars(parser.parse_args())

    extract_shapes(args['input_file_path'],
                   args['output_file_path'], args['region_shape_file_path'])
