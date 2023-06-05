ID=test_set_bigger

# clean up outdates files
cd $HOME/dev/imec/mobility-and-logistics/cityflows-data-model
rm -rf data/managed_data_files/$ID

# compute a new counts data file for the model and for the road cutting
# can use bounds or indices for filtering the AAA counts data file
python src/model/create_test_set.py --longitudes 4.3 4.306 --latitudes 51.29475 51.30 # bounds filtering
# python src/model/create_test_set.py --indices 145542 145543 145544 146042 146043 146044 # indices filtering
cp data/managed_data_files/$ID/input/counts/data.csv ../cityflows-street-cutting-postgis/data

# perform road cutting
cd $HOME/dev/imec/mobility-and-logistics/cityflows-street-cutting-postgis
yarn start
cp output/* ../cityflows-data-model/data/managed_data_files/$ID/input

# run the model
cd $HOME/dev/imec/mobility-and-logistics/cityflows-data-model
python -m src.model.Main_batch --blobs_dir $ID --counts_prefixes data.csv --compute_on_local_files --solver gurobi --densities_folder_suffix gurobi
python -m src.model.Main_batch --blobs_dir $ID --counts_prefixes data.csv --compute_on_local_files --solver osqp --densities_folder_suffix osqp
