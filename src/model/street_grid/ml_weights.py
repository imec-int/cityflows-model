import pandas as pd
from ..data_mappings import mapping_from_dict


def load_ml_weights(input_path, modality_mapping):
    """Maps and groups weights of each modality from machine learning according to the modality mapping

    Args:
        input path: location on disk where csv file is located
        modality_mapping: modality information dictionary

    Returns:
        mapped ML weights based on given modality mapping dataframe per street_object_id
    Raises:
        --
    """
    try:
        ml_weights = pd.read_csv(input_path, index_col=0).set_index(
            'WS_OIDN_left').rename_axis('street_object_id')
    except:        
        raise Exception(f'ML weights not found in {input_path}')
    ml_weights = ml_weights.rename(
        columns=lambda x: mapping_from_dict(x, modality_mapping))
    ml_weights = ml_weights.groupby(ml_weights.columns, axis=1).sum()

    #dropping doubled entries    
    ml_weights.reset_index(inplace=True)
    ml_weights.drop_duplicates(inplace=True)
    ml_weights.set_index('street_object_id',inplace=True)

    # adding bg modality setting default to 1
    ml_weights['bg_density'] = [1]*len(ml_weights.index)
    ## for ignoring could be set to 0

    # join separate modalities under same 'modality' named column
    ml_weights=ml_weights.melt(var_name='modality', value_name='weight',ignore_index=False)

    ml_weights = ml_weights.groupby(
        ['street_object_id', 'modality']).mean().reset_index()
    return ml_weights

def get_default_weights(street_segments,modalities_list):
    """Returns a dataframe of 1's in the same format as the ml_weights

    Args:
        street_segments: dataframe with street info
        modalities_list: list with all the modalities

    Returns:
        1's for every street for every modality
    Raises:
        --    
    """
    default_weights = pd.DataFrame(street_segments['street_object_id'].drop_duplicates())
    default_weights = default_weights.merge(pd.DataFrame(modalities_list,columns=['modality']),how='cross')
    default_weights['weight'] = 1.
    return default_weights