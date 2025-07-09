import numpy as np
import time
import pandas as pd

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

import os
import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.losses import binary_crossentropy
import tensorflow_addons as tfa

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model(input_shape) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3), padding="SAME", activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Conv2D(64,kernel_size=(3,3), padding="SAME", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Conv2D(128,kernel_size=(3,3), padding="SAME", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Conv2D(256,kernel_size=(3,3), padding="SAME", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(20, activation='sigmoid'))

    print("✅ Model initialized")
    print(model.summary)

    return model

#Compile the CNN


def compile_model(model: Model, learning_rate) -> Model:
    """
    Compile the Neural Network
    """
    data_dir = "/home/gulfairus/.database/lung_cancer/data/raw"
    train_df = pd.read_csv(os.path.join(data_dir, "miccai2023_nih-cxr-lt_labels_train.csv"))
    labels = train_df.drop(columns=['id', 'subj_id'])
    labels = labels.apply(lambda x: x.to_list(), axis=1)
    labels = labels.to_list()
    label_counts = np.sum(labels, axis=0)
    TF_pos = label_counts / len(labels)
    TF_neg = 1 - TF_pos
    IDF = tf.keras.backend.log((1 + len(labels)) / (1 + label_counts)) + 1  # Smoothing
    IDF = tf.cast(IDF, tf.float32)
    TF_IDF_pos = TF_pos * IDF
    weights_pos = 1.0 / TF_IDF_pos  # Invert to give higher weights to rarer labels
    weights_pos = weights_pos / tf.reduce_max(weights_pos)  # Normalize to [0, 1]
    TF_IDF_neg = TF_neg * IDF
    weights_neg = 1.0 / TF_IDF_neg  # Invert to give higher weights to rarer labels
    weights_neg = weights_neg / tf.reduce_max(weights_neg)  # Normalize to [0, 1]


    def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
        def weighted_loss(y_true, y_pred):
            loss = 0.0
            for i in range(len(pos_weights)):
                loss += tf.keras.backend.mean(-(pos_weights[i] * y_true[:, i] * tf.keras.backend.log(y_pred[:, i]+epsilon) + neg_weights[i] * (1-y_true[:, i]) * tf.keras.backend.log(1-y_pred[:, i]+epsilon)))
            return loss
        return weighted_loss

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    f1_score = tfa.metrics.F1Score(num_classes=20, average='macro', threshold=0.5)

    model.compile(optimizer=optimizer, loss=get_weighted_loss(weights_pos, weights_neg), metrics=[tf.keras.metrics.AUC(
            name='auroc',
            multi_label=True,
            num_labels=20,
            from_logits=False
        )])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        train_data,
        batch_size,
        patience,
        validation_data=None,
        epochs=None) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=.01,
        mode='auto',
        restore_best_weights=True,
        verbose=1,
        start_from_epoch = 10
    )

    rlr = ReduceLROnPlateau( monitor="val_loss",
                            factor=0.2,
                            patience=patience,
                            verbose=0,
                            mode="auto",
                            min_delta=0.001)

    #steps = len(train_names)//batch_size

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, rlr],
        verbose=1
    )


    print(f"✅ Model trained ")

    return model, history


def evaluate_model(
        model: Model,
        test_data,
        batch_size
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model ..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        test_data = test_data,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics[tf.keras.metrics.AUC(
            name='auroc',
            multi_label=True,
            num_labels=20,
            from_logits=False
        )]

    print(f"✅ Model evaluated, accuracy: {round(accuracy, 2)}")

    return metrics
