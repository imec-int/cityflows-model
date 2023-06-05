import pickle

def load_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def write_pickle_file(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
