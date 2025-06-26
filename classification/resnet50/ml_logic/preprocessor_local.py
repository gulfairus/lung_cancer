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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pydicom
import cv2
from classification.params import *
from google.cloud import storage
import io
import tensorflow as tf


def preprocess_data():

    # generate training,testing and validation batches
    image_dir = DICOM_DATA_PATH

    #load dataframe

    train_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_train.csv"))
    valid_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_val.csv"))
    test_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_test.csv"))

    #changing png to dcm

    id = train_df['id'].apply(lambda x: x.split('.')[0] + '.dcm')
    train_df['id'] = id
    id = valid_df['id'].apply(lambda x: x.split('.')[0] + '.dcm')
    valid_df['id'] = id
    id = test_df['id'].apply(lambda x: x.split('.')[0] + '.dcm')
    test_df['id'] = id

    train_id = list(train_df["id"].values)

    labels = train_df.drop(columns=['id', 'subj_id'])
    labels = labels.apply(lambda x: x.to_list(), axis=1)
    num_classes = len(labels[0])


    bucket_name = 'lung_cancer1'
    image_size = (320, 320)

    client = storage.Client()
    bucket = client.bucket(bucket_name)


    def read_dicom_from_gcs(blob_path, df):
        if blob_path.split('/')[2] in df:
            blob = bucket.blob(blob_path)
            dicom_bytes = blob.download_as_bytes()
            ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
            arr = ds.pixel_array.astype(np.float32)
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))  # Normalize to [0,1]
            arr = np.stack([arr] * 3, axis=-1)  # Make 3-channel RGB
        return arr

    def load_image_tf(dicom_path, label, image_size=image_size):
        def _load(path_str, label_arr):
            image = read_dicom_from_gcs(path_str.numpy().decode('utf-8'))
            image = tf.image.resize(image, image_size)
            label_tensor = tf.convert_to_tensor(label_arr, dtype=tf.float32)
            return image, label_tensor

        image, label = tf.py_function(
            func=_load,
            inp=[dicom_path, label],
            Tout=(tf.float32, tf.float32)
        )
        image.set_shape([*image_size, 3])
        label.set_shape([num_classes])
        return image, label



    blobs = bucket.list_blobs(prefix='dicom/dicom')
    dicom_paths = [blob.name for blob in blobs if blob.name.endswith(".dcm")]
    print(dicom_paths)

    label_array = np.array(labels.tolist(), dtype=np.float32)
    #filename_tensor = tf.constant(train_df["id"].values)
    label_tensor = tf.constant(label_array)

    dataset = tf.data.Dataset.from_tensor_slices((dicom_paths, label_tensor))
    dataset = dataset.map(lambda path, label: load_image_tf(path, label), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)


    return None

iterator = iter(preprocess_data())
print(next(iterator))
