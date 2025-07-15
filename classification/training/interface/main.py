import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

import pandas as pd
import os
# from skimage.transform import resize
# from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from PIL import Image
from google.cloud import storage
import requests
from io import BytesIO
import random
from classification.params import *
import tensorflow as tf
import tensorflow_io as tfio
from classification.training.ml_logic.data import load_data_to_bq
from classification.training.ml_logic.model_scratch import initialize_model, compile_model, train_model, evaluate_model
from classification.training.ml_logic.registry_scratch import load_model, save_model, save_results, save_history
from classification.training.ml_logic.registry import mlflow_run, mlflow_transition_model
import tensorflow_addons as tfa

# def preprocess() -> None:
#     #storage_client = storage.Client(GCP_PROJECT)
#     #bucket = storage_client.get_bucket(BUCKET_NAME)


#     # load_data_to_bq(
#     #     df_processed,
#     #     gcp_project=GCP_PROJECT,
#     #     bq_dataset=BQ_DATASET,
#     #     table=f'df_processed',
#     #     truncate=True
#     # )

#     print("✅ preprocess() done \n")
#     return None

@mlflow_run
def train(
        learning_rate=0.0001,
        batch_size = 32,
        patience = 2,
        epochs=100, image_size = (224, 224)
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\generating data..." + Style.RESET_ALL)

    def parse_tfrecord(example_proto):
        features = {
            'images': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.FixedLenFeature([], tf.string),
            #'id': tf.io.FixedLenFeature([], tf.string),
        }
        parsed = tf.io.parse_single_example(example_proto, features)
        images = tf.io.parse_tensor(parsed['images'], out_type=tf.float32)
        labels = tf.io.parse_tensor(parsed['labels'], out_type=tf.float32)
        images = tf.reshape(images, [224, 224,1])
        labels = tf.reshape(labels, [20,])
        #id = tf.io.parse_tensor(parsed['id'], out_type=tf.string)
        return images, labels


    # Load dataset back
    train_path = os.path.join('/home/gulfairus/.database/lung_cancer/data/processed', "train_dataset2.tfrecord")
    validation_path = os.path.join('/home/gulfairus/.database/lung_cancer/data/processed', "valid_dataset2.tfrecord")
     #test_dataset = pd.read_csv(os.path.join(PREPROCESSED_DATA_PATH, "test_dataset.tfrecord"))

    reloaded_ds = tf.data.TFRecordDataset(train_path)
    train_dataset = reloaded_ds.map(parse_tfrecord)
    '''
    def reshape_fn(image, label):
        image = tf.reshape(image, (224,224,1))
        label = tf.reshape(label, (20,1))
        return image, label
    '''

    def augment_fn(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        #image = tf.image.random_rotate(image, angles=tf.random.uniform([], -0.1, 0.1))  # radians
        image = tfa.image.rotate(image, angles=tf.random.uniform([], -0.1, 0.1))  # radians
        return image, label

    #train_dataset = train_dataset.map(reshape_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)

    #iterator = iter(train_dataset)
    #print(iterator.next())

    reloaded_ds = tf.data.TFRecordDataset(validation_path)
    validation_dataset = reloaded_ds.map(parse_tfrecord)
    #validation_dataset = validation_dataset.map(reshape_fn, num_parallel_calls=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)


    print(Fore.BLUE + "\data loaded" + Style.RESET_ALL)

    # Train model using `model.py`
    #model = load_model()

    #if model is None:
    #    model = initialize_model(input_shape=(224,224,1))
    model = initialize_model(input_shape=(224,224,1))
    model = compile_model(model, learning_rate=0.0001)
    model, history = train_model(
        model, train_data=train_dataset, batch_size=batch_size,
        patience=patience,validation_data=validation_dataset, epochs=epochs
    )

    val_accuracy = np.min(history.history['auroc'])

    params = dict(
        context="train",
        #training_set_size=DATA_SIZE,
        #row_count=len(X_train_processed),
    )

    save_results(params=params, metrics=dict(accuracy=val_accuracy))
    save_history(history=history)

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # The latest model should be moved to staging
    # $CHA_BEGIN
    #if MODEL_TARGET == 'mlflow':
    #    mlflow_transition_model(current_stage="None", new_stage="Staging")
    # $CHA_END

    print("✅ train() done \n")

    return val_accuracy


@mlflow_run
def evaluate(
        batch_size = 32,
        # min_date:str = '2014-01-01',
        # max_date:str = '2015-01-01',
        stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None

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


    test_path = os.path.join(PREPROCESSED_DATA_PATH, "test_dataset2.tfrecord")

    reloaded_ds = tf.data.TFRecordDataset(test_path)
    test_dataset = reloaded_ds.map(parse_tfrecord)

    metrics_dict = evaluate_model(model=model, test_data=test_dataset, batch_size=batch_size)
    accuracy = metrics_dict['auroc']

    params = dict(
        context="evaluate", # Package behavior
        #training_set_size=DATA_SIZE,
        #row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return accuracy


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    from google.colab import files
    from keras.preprocessing import image
    uploaded = files.upload()
    print("\n⭐️ Use case: predict")

    image_size = (224, 224)
    MEAN_TRAIN = 0.53306305
    STD_TRAIN = 0.24305601

    model = load_model()
    assert model is not None

    for filename in uploaded.keys():
        img_path = os.getcwd+filename
        #img = image.load_img(img_path, target_size=(224,224))

        image_bytes = tf.io.read_file(img_path)
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
        image = (image - mean) / (stddev + 1e-6)

        image = image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)

    print("\n✅ prediction done: ", prediction, prediction.shape, "\n")
    return prediction


if __name__ == '__main__':
    #preprocess()
    train()
    evaluate()
    pred()
