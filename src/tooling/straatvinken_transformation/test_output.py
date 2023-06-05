import os

from src.tooling.validation.score import read_validated_data

if __name__ == '__main__':
    folder = 'data/managed_data_files/mobiele_stad/validation_straatvinken/transformed'
    file = 'SV2020_DataVVR-Antwerp_20210422_transformed.csv'
    filepath = os.path.join(folder, file)
    df = read_validated_data(filepath)
    print(df.head())
