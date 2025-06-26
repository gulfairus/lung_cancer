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
    val_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_val.csv"))
    test_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "miccai2023_nih-cxr-lt_labels_test.csv"))


    def load_dicom_image(path, image_size):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.resize(img, image_size)  # Resize to (H, W)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#        if len(img.shape) == 2:
#            img = np.expand_dims(img, axis=-1)
#        img = np.repeat(img, 3, axis=-1)  # Convert to 3 channels
        return img

    def compute_mean_std_train(df, image_dir, image_size):
        pixels = []
        df = df[['id']][:1000]
        for i, row in df.iterrows():
            #print(row)
            im = row['id'].split('.')[0]
            dic = im + '.dcm'
            img_path = f"{image_dir}/{dic}"
            img = load_dicom_image(img_path, image_size)
            #img = image.load_img(img_path, target_size=image_size)
            #img_array = image.img_to_array(img) / 255.0  # Scale to [0, 1]
            img_array = img / 255.0
            pixels.append(img_array)
        pixels = np.stack(pixels)
        print(pixels.shape)
        mean = np.mean(pixels)
        std = np.std(pixels)
        return mean, std

    mean, std = compute_mean_std_train(train_df, image_dir, image_size=(320, 320))


    return mean, std

print(preprocess_data())
