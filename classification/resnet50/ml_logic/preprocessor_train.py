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
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pydicom
import cv2
from classification.params import *
from google.cloud import storage, bigquery
import io
import tensorflow as tf
import time
import tensorflow_io as tfio
from colorama import Fore, Style

train_main = []
train_std = []
#start_time = time.time()
bucket_name = "lung_cancer1"

def train_data():
    start_time = time.time()


    train_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_train.csv"))
    #valid_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_val.csv"))
    #test_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_test.csv"))
    print(f"train_df {train_df.shape}")

    #changing png to dcm

    id = train_df['id'].apply(lambda x: x.split('.')[0] + '.dcm')
    train_df['id'] = id
    #id = valid_df['id'].apply(lambda x: x.split('.')[0] + '.dcm')
    #valid_df['id'] = id
    #id = test_df['id'].apply(lambda x: x.split('.')[0] + '.dcm')
    #test_df['id'] = id

    train_id = list(train_df["id"].values)

    labels = train_df.drop(columns=['id', 'subj_id'])
    labels = labels.apply(lambda x: x.to_list(), axis=1)
    num_classes = len(labels[0])
    #print(num_classes)


    bucket_name = "lung_cancer1"
    image_size = (224, 224)
    MEAN_TRAIN = 0.53306305
    STD_TRAIN = 0.24305601

    #client = storage.Client()
    #bucket = client.bucket(bucket_name)
    #train_main = []
    #train_std = []

    #def read_dicom_from_gcs(blob_path):

        #blob = bucket.blob(blob_path)
        ##print(blob.name)
        #dicom_bytes = blob.download_as_bytes()
        #ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        #arr = ds.pixel_array.astype(np.float32)
        #arr = np.resize(arr, image_size)
        #arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))  # Normalize to [0,1]
        #arr = np.stack([arr] * 1, axis=-1)  # Make 3-channel RGB
        #mean = np.mean(arr)
        #std = np.std(arr) # add epsilon to avoid div by 0
        #arr = (arr - mean) / std
        #return arr.astype(np.float32), np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)
        #return arr, mean, std



    def read_dicom_from_gcs2(path, label):

        image_bytes = tf.io.read_file(path)
        image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16, scale="auto")
        image = tf.squeeze(image, axis=0)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        #image = image / tf.reduce_max(image)
        image_min = tf.reduce_min(image)
        image_max = tf.reduce_max(image)
        image = (image - image_min) / (image_max - image_min + 1e-8)
        # Standardize: (x - mean) / std
        #mean, variance = tf.nn.moments(image, axes=[0, 1])
        #stddev = tf.sqrt(variance)
        mean = MEAN_TRAIN
        stddev = STD_TRAIN
        #mean = tf.reduce_mean(image)
        #stddev = tf.math.reduce_std(image)
        image = (image - mean) / (stddev + 1e-6)  # add epsilon for stability
        # Expand grayscale to 3 channels if needed
        #image = tf.expand_dims(image, -1)
        #image = tf.image.grayscale_to_rgb(image)
        img = tf.cast(path, tf.string)
        #print(img)

        #return image, tf.cast(label, tf.float32), mean, stddev
        return image, tf.cast(label, tf.float32)


    #def load_image_tf(dicom_path, label, image_size=image_size, num_classes=num_classes):
    #    def _load(path_str, label_arr):
    #        image, mean, std = read_dicom_from_gcs(path_str.numpy().decode('utf-8'))
    #        #image = tf.image.resize(image, image_size)
    #        label_tensor = tf.convert_to_tensor(label_arr, dtype=tf.float32)
    #        return image, label_tensor, mean, std

    #    image, label, mean, std = tf.py_function(
    #        func=_load,
    #        inp=[dicom_path, label],
    #        Tout=(tf.float32, tf.float32, tf.float32, tf.float32)
    #    )
    #    image.set_shape([*image_size, 1])
    #    label.set_shape([num_classes])
    #    mean.set_shape([])
    #    std.set_shape([])

    #    return image, label, mean, std



    #blobs = bucket.list_blobs(prefix='dicom/dicom')
    #dicom_paths = [blob.name for blob in blobs if blob.name.split('/')[2] in train_id][:5]
    #dicom_paths = [f"gs://{bucket_name}/"+ blob.name for blob in blobs if blob.name.split('/')[2] in train_id][:5]
    dicom_paths = [f"gs://{bucket_name}/dicom/dicom/"+ id for id in train_id][:5]
    #print(dicom_paths)

    #print(dicom_paths)
    #for blob in dicom_paths:
    #    mean, std = read_dicom_from_gcs(blob)
    #    train_main.append(mean)
    #    train_std.append(std)


    label_array = np.array(labels.tolist()[:5], dtype=np.float32)
    #filename_tensor = tf.constant(train_df["id"].values)
    label_tensor = tf.constant(label_array)
    #print(labels.tolist()[:5])

    #dataset = tf.data.Dataset.from_tensor_slices((dicom_paths, label_tensor))
    #dataset = dataset.map(lambda path, label: load_image_tf(path, label, image_size=image_size, num_classes=num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    #dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.from_tensor_slices((dicom_paths, label_tensor))
    dataset = dataset.map(read_dicom_from_gcs2, num_parallel_calls=tf.data.AUTOTUNE)
    #ds_for_training = dataset.map(lambda x, y: (x, y['label']))
    dataset = dataset.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time {elapsed_time}")


    return dataset

#iterator = iter(preprocess_data())
dataset = train_data()
for img, lbl in dataset:
    images = img
    labels = lbl

np.save('/home/gulfairus/.database/lung_cancer/data/processed/train_dicom.npy', images)
np.save('/home/gulfairus/.database/lung_cancer/data/processed/train_label.npy', labels)

#print(images, labels, ids)
#end_time = time.time()
#elapsed_time = end_time - start_time
#print(f"elapsed_time {elapsed_time}")
#iterator = iter(dataset)
#print(iterator.next())

def serialize_batch(images, labels):
    # Flatten the 4D tensor to 1D byte string
    images_bytes = tf.io.serialize_tensor(images)
    labels_bytes = tf.io.serialize_tensor(labels)
    #id_bytes = tf.io.serialize_tensor(id)

    features = {
        'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images_bytes.numpy()])),
        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels_bytes.numpy()])),
        #'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id_bytes.numpy()])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()

#output = f"gs://{bucket_name}/dicom/preprocessed_data1.tfrecord"

#with tf.io.TFRecordWriter(output) as writer:
#    for images, labels in dataset:
#        serialized = serialize_batch(images, labels)
#        writer.write(serialized)

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

# Load dataset back from GCS
#reloaded_ds = tf.data.TFRecordDataset(output)
#reloaded_ds = reloaded_ds.map(parse_tfrecord)

#for img, lbl, id in reloaded_ds:
 #   images = img
#    labels = lbl
#    ids = id
#print(images, labels, ids)
