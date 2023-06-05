# Files

The json files in this folder will contain some information about the data_sources and the modalities used in the model.

## count_constraints_bounds.json

For every data source which bounds of the count constraints are taken into account.

## modality_mapping.json

For evey modality some information:
- "max_density": the value which is the highest density (people/meter) allowed.
- "modal_split": percentage in the modal split !DEPRECIATED: not used anymore!
- "average_speed": the estimated average speed of this modality.
- "raw_modalities": The names given to this modality by all the sources.