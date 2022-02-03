import pickle

def save_object(filenname: str, data: object):
    with open(filenname, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filenname: str) -> object:
    with open(filenname, "rb") as input_file:
        return pickle.load(input_file)