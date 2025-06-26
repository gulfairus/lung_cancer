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
from google.cloud import storage
import io
from io import StringIO


def preprocess_data():

    # generate training,testing and validation batches

    #image_dir = DICOM_DATA_PATH


    bucket_name = 'lung_cancer1'
    image_size = (320, 320)

    def read_dicom_images_from_gcs(bucket_name, prefix='dicom/dicom/', image_size=image_size):

        #train_df = pd.read_csv(os.path.join('/home/gulfairus/.database/lung_cancer/data/raw', "miccai2023_nih-cxr-lt_labels_train.csv"))
        #val_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_val.csv"))
        #test_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_test.csv"))
        #print(train_df.shape)
        #df = train_df[['id']]

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        train_df = bucket.blob('dicom/miccai2023_nih-cxr-lt_labels_train.csv')
        train_df = train_df.download_as_text()
        train_df = pd.read_csv(StringIO(train_df))
        print(train_df.shape)
        df = list(train_df['id'])

        blobs = bucket.list_blobs(prefix=prefix)
        print(blobs)

        dicom_data = {}

        for blob in blobs:
            if not blob.name.lower().endswith('.dcm'):
                continue  # Skip non-DICOM files
            nam = blob.name.split('/')[2]
            nam = nam.split('.')[0]
            nam = nam + '.png'

            if nam in df:
                dcm_bytes = blob.download_as_bytes()
                dcm_file = pydicom.dcmread(io.BytesIO(dcm_bytes))
                dcm_file = dcm_file.pixel_array.astype(np.float32)
                img = cv2.resize(dcm_file, image_size)  # Resize to (H, W)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = img / 255.0
                dicom_data[blob.name] = img
                print(len(dicom_data.keys()))


            #if len(dicom_data.keys())==78506:
            if len(dicom_data.keys())==100:
                pixels = np.stack(dicom_data.values())
                print(pixels.shape)
                mean = np.mean(pixels)
                std = np.std(pixels)
                print(mean, std)
            if len(dicom_data.keys())==200:
                pixels = np.stack(dicom_data.values())
                print(pixels.shape)
                mean = np.mean(pixels)
                std = np.std(pixels)
                print(mean, std)
            if len(dicom_data.keys())==300:
                pixels = np.stack(dicom_data.values())
                print(pixels.shape)
                mean = np.mean(pixels)
                std = np.std(pixels)
                print(mean, std)
            if len(dicom_data.keys())==400:
                pixels = np.stack(dicom_data.values())
                print(pixels.shape)
                mean = np.mean(pixels)
                std = np.std(pixels)
                print(mean, std)
            if len(dicom_data.keys())==500:
                pixels = np.stack(dicom_data.values())
                print(pixels.shape)
                mean = np.mean(pixels)
                std = np.std(pixels)
                print(mean, std)



        #return dicom_data
        return None

    #dicom_data = read_dicom_images_from_gcs(bucket_name, prefix='dicom/dicom/', image_size=image_size)
    read_dicom_images_from_gcs(bucket_name, prefix='dicom/dicom/', image_size=image_size)
    #pixels = np.stack(dicom_data.values())
    #print(pixels.shape)
    #mean = np.mean(pixels)
    #std = np.std(pixels)

    #return mean, std


print(preprocess_data())
