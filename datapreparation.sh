datapath=data/managed_data_files/mobiele_stad/2020_analysis/data_preparation
outputpath=data/managed_data_files/mobiele_stad/2020_analysis/input

for month in jan feb mar apr may jun jul aug sep okt nov dec
do
python -m src.tooling.merge.main --input_files $datapath/anpr/counts_anpr_$month.csv $datapath/cropland/counts_cropland_$month.csv \
    $datapath/signco/counts_signco_$month.csv $datapath/telraam/counts_telraam_$month.csv $datapath/velo2/counts_velo_$month.csv \
    --output_file $outputpath/input_$month/all_data_2.csv \
    --telco_data_source cropland --area data/managed_data_files/shapefiles/modelling_zones/antwerp/zone.geojson
echo $month 'finished'
done