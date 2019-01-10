import os
import pickle


def cache(dataset, file_name):
    print("Caching dataset to {} ... ".format(file_name), end='', flush=True)
    pickle.dump(dataset, open(file_name, 'wb'))
    print("done")


def retreive_from_cache(file_name):
    if os.path.exists(file_name):
        print("Retrieving {} from cache... ".format(file_name), end='', flush=True)
        dataset = pickle.load(open(file_name, 'rb'))
        print("done")
        return dataset
    else:
        print("Couldn't find a dataset at {}".format(file_name))
        return False
