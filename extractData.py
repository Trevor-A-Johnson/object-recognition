import os
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_images(dataset_path="cifar-10-batches-py"):
    """
    Retrieves all the data from the cifar-10-batches-py folder and returns an array with all the images.
    """
    images = []
    
    batch_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
    
    for batch_file in batch_files:
        file_path = os.path.join(dataset_path, batch_file)
        if os.path.exists(file_path):
            batch_dict = unpickle(file_path)
            batch_images = batch_dict[b'data']
            images.append(batch_images)
            
    if len(images) > 0:
        return np.concatenate(images)
    else:
        return np.array([])
