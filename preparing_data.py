import gzip
import pickle
import os
import urllib.request
import numpy as np
from tqdm import tqdm

def download_binarirzed_mnist():
    main_path = os.getcwd() # Current directory
    path_original_data = os.path.join("..", "Data", "Original_binarized_mnist") # Directory of the original data
    filenames = tqdm(["train", "valid", "test"]) # Datasets
    for filename in filenames:
        filenames.set_description("Downloading: {}".format(filename))
        filename_path = os.path.join(main_path, path_original_data, "binarized_mnist_" + filename) # path + filename
        url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(filename)
        urllib.request.urlretrieve(url, filename_path) # Download and save to the filename_path
    return 0

def read_original_data_and_write_pickled_data_gzip():
    path_original_data = os.path.join("..", "Data", "Original_binarized_mnist")
    path_pickled_data = os.path.join("..", "Data", "Pickle_binarized_mnist")
    filenames = tqdm(os.listdir(path_original_data))
    dict_names = {}
    for filename in filenames:
        dataset = filename.split("_")[-1].split(".")[0]
        filenames.set_description("Reading original: {}".format(dataset))
        with open(os.path.join(path_original_data, filename), "r") as fl:
            dict_names[dataset] = []
            for line in fl:
                dict_names[dataset].append(list(map(float, line.split())))
        dict_names[dataset] = np.array(dict_names[dataset])
        with gzip.open(os.path.join(path_pickled_data, filename) + ".pickle.gzip", "wb") as fl:
            pickle.dump(dict_names[dataset], fl)
    return dict_names

def read_pickled_data_gzip():
    path_pickled_data = os.path.join("..", "Data", "Pickle_binarized_mnist")
    filenames = tqdm(os.listdir(path_pickled_data))
    dict_names = {}
    for filename in filenames:
        dataset = filename.split("_")[-1].split(".")[0]
        filenames.set_description("Reading compressed pickled: {}".format(dataset))
        with gzip.open(os.path.join(path_pickled_data, filename), "rb") as fl:
            dict_names[dataset] = pickle.load(fl)
    train_data, valid_data, test_data = [dict_names[key] for key in ["train", "valid", "test"]]
    return train_data, valid_data, test_data

if __name__ == "__main__":
    download_bool = False
    if download_bool:
        download_binarirzed_mnist()
        read_original_data_and_write_pickled_data_gzip()
    train_data, valid_data, test_data = read_pickled_data_gzip()
    print("\nShapes for train:{} valid:{} test:{}\n".format(train_data.shape, valid_data.shape, test_data.shape))

