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
import time
import tensorflow_io as tfio

start_time = time.time()

def valid_data():



    # generate training,testing and validation batches
    #image_dir = DICOM_DATA_PATH

    #load dataframe

    #train_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_train.csv"))
    valid_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_val.csv"))
    #test_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_test.csv"))
    print(f"valid_df {valid_df.shape}")

    #changing png to dcm

    #id = train_df['id'].apply(lambda x: x.split('.')[0] + '.dcm')
    #train_df['id'] = id
    id = valid_df['id'].apply(lambda x: x.split('.')[0] + '.dcm')
    valid_df['id'] = id
    #id = test_df['id'].apply(lambda x: x.split('.')[0] + '.dcm')
    #test_df['id'] = id

    valid_id = list(valid_df["id"].values)

    labels = valid_df.drop(columns=['id', 'subj_id'])
    labels = labels.apply(lambda x: x.to_list(), axis=1)
    num_classes = len(labels[0])
    #print(num_classes)


    bucket_name = 'lung_cancer1'
    image_size = (224, 224)
    MEAN_TRAIN = 0.53306305
    STD_TRAIN = 0.24305601


    def read_dicom_from_gcs2(path, label):

        image_bytes = tf.io.read_file(path)
        image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16, scale="auto")
        image = tf.squeeze(image, axis=0)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        image = image / tf.reduce_max(image)
        # Standardize: (x - mean) / std
        #mean, variance = tf.nn.moments(image, axes=[0, 1])
        #stddev = tf.sqrt(variance)
        #mean = tf.reduce_mean(image)
        #stddev = tf.math.reduce_std(image)
        image = (image - MEAN_TRAIN) / (STD_TRAIN + 1e-6)  # add epsilon for stability
        # Expand grayscale to 3 channels if needed
        #image = tf.expand_dims(image, -1)
        #image = tf.image.grayscale_to_rgb(image)

        return image, tf.cast(label, tf.float32)

    dicom_paths = [f"gs://{bucket_name}/dicom/dicom/"+ id for id in valid_id]
    #print(dicom_paths)

    label_array = np.array(labels.tolist(), dtype=np.float32)
    #filename_tensor = tf.constant(train_df["id"].values)
    label_tensor = tf.constant(label_array)
    #print(labels.tolist()[:5])

    dataset = tf.data.Dataset.from_tensor_slices((dicom_paths, label_tensor))
    dataset = dataset.map(read_dicom_from_gcs2, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time {elapsed_time}")

    return dataset


dataset = valid_data()
for img, lbl in dataset:
    images = img
    labels = lbl

np.save('/home/gulfairus/.database/lung_cancer/data/processed/valid_dicom.npy', images)
np.save('/home/gulfairus/.database/lung_cancer/data/processed/valid_label.npy', labels)

#images = np.load('/home/gulfairus/.database/lung_cancer/data/processed/train_dicom.npy')
end_time = time.time()
elapsed_time = end_time - start_time

print(f"elapsed_time {elapsed_time}")
print(f"✅ Data saved locally")

def load_data():
    images = np.load('/home/gulfairus/.database/lung_cancer/data/processed/valid_dicom.npy')
    labels = np.load('/home/gulfairus/.database/lung_cancer/data/processed/valid_label.npy')


    return images, labels

images, labels = load_data()
print(images.shape)
print(labels.shape)
