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


def preprocess_data():

    # generate training,testing and validation batches

    image_dir = DICOM_DATA_PATH


    train_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_train.csv"))
    #val_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_val.csv"))
    #test_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_test.csv"))

    def read_dicom_images_from_gcs(bucket_name, prefix='dicom/dicom/', image_size=image_size):
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=prefix)

        dicom_data = {}
        for blob in blobs[:10]:
            if not blob.name.lower().endswith('.dcm'):
                continue  # Skip non-DICOM files
            print(blob.name)
            dcm_bytes = blob.download_as_bytes()
            dcm_file = pydicom.dcmread(io.BytesIO(dcm_bytes))
            dcm_file = dcm_file.pixel_array.astype(np.float32)
            img = cv2.resize(dcm_file, image_size)  # Resize to (H, W)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img / 255.0
            dicom_data[blob.name] = img

        return dicom_data

    dicom_data = read_dicom_images_from_gcs('lung_cancer1', prefix='dicom/dicom/', image_size=(320, 320))

    print(len(dicom_data.keys()))
    mean = np.mean(dicom_data.values(), axis=(0, 1, 2))
    std = np.std(dicom_data.values(), axis=(0, 1, 2))

    return mean, std

print(preprocess_data())
