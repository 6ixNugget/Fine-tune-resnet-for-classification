from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import hashlib
from PIL import Image
from shutil import copyfile
import random

DATA_DIR = "./img_align_celeba"
TRAIN_DIR = "./data/training"
TEST_DIR = "./data/test"
VALIDATION_DIR = "./data/validation"
LABEL_PATH = "./list_attr_celeba.txt"

TRAIN_NUM = 10000
VALIDATION_NUM = 1000
TEST_NUM = 1000
### Examples with glasses: 13193  Examples without glasses: 189406

feature_to_split = "Eyeglasses"

def parse_labels():
    with open(LABEL_PATH, 'r') as label_file:
        label_lines = label_file.read().splitlines()
        all_feat = list(feat for feat in label_lines[1].split())
        label_lines = label_lines[2:]
    
    labels = []
    for line in label_lines:
        name = line.split()[0]
        features = np.array(line.split()[1:])
        feat_idx = all_feat.index(feature_to_split)
        features = features[feat_idx]
        labels.append((name, features))

    return labels


def make_dir_except(DIR_PATH):
    try:
        os.makedirs(DIR_PATH)
    except FileExistsError:
        pass

def copy_files(labels, target_dir):
    for filename, label in labels:
        make_dir_except(os.path.join(target_dir, label))
        copyfile(os.path.join(DATA_DIR, filename), os.path.join(target_dir, label, filename))

def get_label_list(labels):
    label_list = list(set([i[1] for i in labels]))
    return label_list

def split_dataset(labels):
    """ Split dataset into training, test, and validation sets. Support multiclass labelling.
    
    Args:
        labels: list of labels, each label is a tuple with the format of (filename, class_label)
    """
    if os.path.isdir(TRAIN_DIR) or os.path.isdir(TEST_DIR) or os.path.isdir(VALIDATION_DIR):
        print ("Training/Test/Validation directories already exist. Abort splitting")
        return

    if TRAIN_NUM + VALIDATION_NUM + TEST_NUM > len(labels):
        print ("Number of training samples is smaller than the sum of split targets. Abort splitting")

    os.makedirs(TRAIN_DIR)
    os.makedirs(TEST_DIR)
    os.makedirs(VALIDATION_DIR)

    label_list = get_label_list(labels)

    print(label_list)
    for label in label_list:
        labelled_examples = [x for x in labels if x[1] == label]
        random.shuffle(labelled_examples)
        test_labels = labelled_examples[:TEST_NUM]
        validation_labels = labelled_examples[TEST_NUM:TEST_NUM+VALIDATION_NUM]
        training_labels = labelled_examples[TEST_NUM+VALIDATION_NUM:TEST_NUM+VALIDATION_NUM+TRAIN_NUM]

        copy_files(test_labels, TEST_DIR)
        copy_files(validation_labels, VALIDATION_DIR)
        copy_files(training_labels, TRAIN_DIR)

if __name__ == "__main__":
    split_dataset(parse_labels())