# Scripts

The intent of the scripts in this folder is to transform the StraatVinken data into a format that can be used by our validation script.

To that end, we have 3 scripts
| script name | purpose | command |
| --------------- | ----------- | ------------- |
| `clip_shapefile.py` | produce a subset of the Wegenregister shapefile with streets within the bounding box of StraatVinken data, this is a performance pre-processing step of the next script | `time python -m src.tooling.straatvinken_transformation.clip_shapefile` |
| `transform.py` | transform the StraatVinken into appropriate format | `time python -m src.tooling.straatvinken_transformation.transform` |
| `test_output.py` | test the validity of the produced csv file | `time python -m src.tooling.straatvinken_transformation.test_output` |

## `clip_shapefile.py`

If interested in running this script/step, you will need

- [Wegenregister street segments](https://www.geopunt.be/catalogus/datasetfolder/a8ea3875-e233-4d3c-8c15-f39c960fb938)
- [StraatVinken data](https://portal.azure.com/#blade/Microsoft_Azure_Storage/BlobPropertiesBladeV2/storageAccountId/%2Fsubscriptions%2Fe9dd7c35-d295-4043-b597-bdee67913b24%2FresourceGroups%2Fdigital-twin-cityflows%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fcityflowsdev/path/data-files%2Fmobiele_stad%2Fvalidation_straatvinken%2FSV2020_DataVVR-Antwerp_20210422.csv/isDeleted//tabToload/0)

The output has been uploaded [here](https://portal.azure.com/#blade/Microsoft_Azure_Storage/ContainerMenuBlade/overview/storageAccountId/%2Fsubscriptions%2Fe9dd7c35-d295-4043-b597-bdee67913b24%2FresourceGroups%2Fdigital-twin-cityflows%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fcityflowsdev/path/data-files/etag/%220x8D937B41A48F9B6%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride//defaultId//publicAccessVal/None), then `shapefiles/Wegenregister_straatvinken`

## `transform.py`

If interested in running this script/step, you will need

- The output of `clip_shapefile.py`, you need all files
- [StraatVinken data](https://portal.azure.com/#blade/Microsoft_Azure_Storage/BlobPropertiesBladeV2/storageAccountId/%2Fsubscriptions%2Fe9dd7c35-d295-4043-b597-bdee67913b24%2FresourceGroups%2Fdigital-twin-cityflows%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fcityflowsdev/path/data-files%2Fmobiele_stad%2Fvalidation_straatvinken%2FSV2020_DataVVR-Antwerp_20210422.csv/isDeleted//tabToload/0)

This script iterates on all StraatVinken locations, map-matches them and in case of impossible/inconfident matches, then some locations are dropped. With current loogic, 95% of locations are kept.

There are 3 output files, located [here](https://portal.azure.com/#blade/Microsoft_Azure_Storage/ContainerMenuBlade/overview/storageAccountId/%2Fsubscriptions%2Fe9dd7c35-d295-4043-b597-bdee67913b24%2FresourceGroups%2Fdigital-twin-cityflows%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fcityflowsdev/path/data-files/etag/%220x8D937B41A48F9B6%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride//defaultId//publicAccessVal/None), then `mobiele_stad/validation_straatvinken` :
| file name | use |
| ------------ | ----- |
| SV2020_DataVVR-Antwerp_20210422_transformed.csv | actual output to be used in the validation script |
| SV2020_DataVVR-Antwerp_20210422_transformed_enriched.csv | file containing the kept StraatVinken locations, useful for debugging with QGIS |
| SV2020_DataVVR-Antwerp_20210422_transformed_dropped.csv | file containing the dropped StraatVinken locations, useful for debugging with QGIS |

## `test_output.py`

If interested in running this script/step, you will need

- The output of `transform.py`, only `SV2020_DataVVR-Antwerp_20210422_transformed.csv`

This script doesn't output any new file, on success it just logs the head of a dataframe.
