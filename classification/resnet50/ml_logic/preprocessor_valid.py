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
from preprocessor_train import *
from google.cloud import storage
import io
import tensorflow as tf
import time


def valid_data(train_main, train_std):
    start_time = time.time()

    # generate training,testing and validation batches
    #image_dir = DICOM_DATA_PATH

    #load dataframe

    #train_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_train.csv"))
    valid_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_val.csv"))
    #test_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_test.csv"))
    print(f"valid_df {valid_df.shape}")

    #changing png to dcm

    id = valid_df['id'].apply(lambda x: x.split('.')[0] + '.dcm')
    valid_df['id'] = id

    valid_id = list(valid_df["id"].values)

    labels = valid_df.drop(columns=['id', 'subj_id'])
    labels = labels.apply(lambda x: x.to_list(), axis=1)
    num_classes = len(labels[0])


    bucket_name = 'lung_cancer1'
    image_size = (224, 224)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    def read_dicom_from_gcs(blob_path, train_main, train_std):

        blob = bucket.blob(blob_path)
        dicom_bytes = blob.download_as_bytes()
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))  # Normalize to [0,1]
        arr = np.stack([arr] * 3, axis=-1)  # Make 3-channel RGB
        arr = (arr - train_main) / train_std
        #return arr.astype(np.float32), np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)
        return arr



    def load_image_tf(dicom_path, label, image_size=image_size, num_classes=num_classes):
        def _load(path_str, label_arr):
            image = read_dicom_from_gcs(path_str.numpy().decode('utf-8'), train_main, train_std)
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
    dicom_paths = [blob.name for blob in blobs if blob.name.split('/')[2] in valid_id]

    label_array = np.array(labels.tolist(), dtype=np.float32)
    #filename_tensor = tf.constant(train_df["id"].values)
    label_tensor = tf.constant(label_array)

    dataset = tf.data.Dataset.from_tensor_slices((dicom_paths, label_tensor))
    dataset = dataset.map(lambda path, label: load_image_tf(path, label, image_size=image_size, num_classes=num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time {elapsed_time}")


    return dataset
