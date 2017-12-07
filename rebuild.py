import csv
import json
import pickle
import gzip
import sys

from scipy.io import loadmat, whosmat
import numpy as np

def extract(data_set):
    images = np.array(
        [image.reshape(28,28) # Convert image from column-major to row-major
              .transpose()              
              .reshape(784) 
         for image 
         in data_set[0][0][0]])

    labels = np.array([label[0] for label in data_set[0][0][1]])

    return images, labels

if __name__ == "__main__":
    mat = loadmat("emnist-digits.mat")

    training_images, training_labels = extract(mat['dataset'][0][0][0])
    test_images,     test_labels     = extract(mat['dataset'][0][0][1])

    # Save as pickle
    dataset = ((training_images, training_labels), (test_images, test_labels))
    with gzip.open('emnist-digits.pkl.gz', 'wb') as f:
        pickle.dump(dataset, f)

    # Save as csv
    with gzip.open('emnist-digits-train.csv.gz', 'wt') as f:
        writer = csv.writer(f)
        for image, label in zip(training_images, training_labels):
            writer.writerow([label] + image.tolist())

    with gzip.open('emnist-digits-test.csv.gz', 'wt') as f:
        writer = csv.writer(f)
        for image, label in zip(test_images, test_labels):
            writer.writerow([label] + image.tolist())
