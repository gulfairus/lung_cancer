import os
import PIL
import time
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import tarfile
#from tqdm import tqdm_notebook as tqdm
from google.cloud import storage
#tqdm().pandas()

from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input

#main_dir = "/content/gdrive/My Drive/data/lung_cancer/"
#main_dir_cloud= "/home/gulfairus/.database/lung_cancer/data/"
#tar = tarfile.open(os.path.join(main_dir_cloud, "images_001.tar.gz"))
#model = DenseNet121(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

def preprocess_features_001():
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    def extract_features():
        main_dir_cloud= "/home/gulfairus/.database/lung_cancer/data/"
        #model = DenseNet121(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
        tar = tarfile.open(os.path.join(main_dir_cloud, "raw", "images_001.tar.gz"))
        features = {}
        for member in tar.getmembers():
            if len(member.name.split('/'))==2:
                img = member.name.split('/')[1]
                image1 = Image.open(tar.extractfile(member))
                image2 = image1.convert('RGB')
                image3 = image2.resize((224,224))
                image4 = np.expand_dims(image3, axis=0)
                image5 = preprocess_input(image4)
                image6 = image5/255
            #    image = image - 1.0
                # image = image.flatten()
                # print(image.shape)
                #feature = model.predict(image6)
                features[img] = image6
                #timestamp = time.strftime("%Y%m%d-%H%M%S")
                print("✅ Features extracted")
        return features

    def save_results(features: dict) -> None:
        if features is not None:
            feature_path = os.path.join(main_dir_cloud, "processed/features/densenet121", "images_001.pickle")
            with open(feature_path, "wb") as file:
                pickle.dump(features, file)

        print("✅ Features saved")
    return None
