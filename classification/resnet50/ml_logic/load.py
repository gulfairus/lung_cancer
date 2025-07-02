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

def load_data():
    images = np.load('/home/gulfairus/.database/lung_cancer/data/processed/train_dicom.npy')
    labels = np.load('/home/gulfairus/.database/lung_cancer/data/processed/train_label.npy')


    return images, labels

images, labels = load_data()
print(images.shape)
print(labels.shape)
