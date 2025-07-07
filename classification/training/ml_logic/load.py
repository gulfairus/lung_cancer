import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import PIL
from PIL import Image
import os
import numpy as np
import pandas as pd
import pickle
import tarfile
from tqdm import tqdm_notebook as tqdm
#tqdm().pandas()
import tensorflow
import pydicom
import cv2
from classification.params import *
from google.cloud import storage, bigquery
import io
import tensorflow as tf
import time
import tensorflow_io as tfio
from colorama import Fore, Style

def load_data():
    preprocessed_path = '/home/gulfairus/.database/lung_cancer/data/processed'
    def parse_tfrecord(example_proto):
        features = {
            'images': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.FixedLenFeature([], tf.string),
            #'id': tf.io.FixedLenFeature([], tf.string),
        }
        parsed = tf.io.parse_single_example(example_proto, features)
        images = tf.io.parse_tensor(parsed['images'], out_type=tf.float32)
        labels = tf.io.parse_tensor(parsed['labels'], out_type=tf.float32)
        #id = tf.io.parse_tensor(parsed['id'], out_type=tf.string)
        return images, labels


    # Load dataset back

    train_ds = tf.data.TFRecordDataset(os.path.join(preprocessed_path, "train_dataset.tfrecord"))
    train_ds = train_ds.map(parse_tfrecord)

    valid_ds = tf.data.TFRecordDataset(os.path.join(preprocessed_path, "valid_dataset.tfrecord"))
    valid_ds = valid_ds.map(parse_tfrecord)

    #test_ds = tf.data.TFRecordDataset(os.path.join(preprocessed_path, "test_dataset.tfrecord"))
    #test_ds = test_ds.map(parse_tfrecord)


    return train_ds, valid_ds
