import json

def get_modality_list(modality_mapping):
    """Gives back a list of all used modalities, based on a modality mapping object

    Args:
        modality_mapping: modality_mapping dict as gotten from get_modality_objects

    Returns:
        modalities, a list of the modeled = mapped modalities
    Raises:
        --
    """
    
    modalities = list(modality_mapping.keys())
    return modalities

def get_modality_objects(input_path):
    """Reads in modality input from disk. Returns relevant modality information dictionary.

    Args:
        input path: location on disk where csv file is located

    Returns:
        modality mapping dataframe, with columns raw_modality (index), mapped_modality, max_density, modal_split
    Raises:
        --
    """

    with open(input_path) as file:
        modality_mapping = json.load(file)

    return modality_mapping
