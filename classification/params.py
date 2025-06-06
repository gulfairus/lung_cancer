import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
BUCKET_URL = os.environ.get("BUCKET_URL")
CATEGORIES = os.environ.get("CATEGORIES")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")
#DATA_SOURCE = os.environ.get("DATA_SOURCE")
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".database", "lung_cancer", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".database", "lung_cancer", "training_outputs")#


TRAIN_DATA_PATH_CLOUD = "/home/user/Covid19/raw_data/cloud/train_test/train_test/train"
TEST_DATA_PATH_CLOUD = "/home/user/Covid19/raw_data/cloud/train_test/train_test/test"

TRAIN_DATA_PATH = "/home/user/code/gulfairus/Covid19/raw_data/cloud/train_test/train_val/train"
TEST_DATA_PATH = "/home/user/code/gulfairus/Covid19/raw_data/cloud/train_test/test"
#TRAIN_NORM_PATH = "/home/user/code/gulfairus/Covid19/raw_data/cloud/train_test/train_val/train_val_norm"
#TRAIN_OTHER_PATH = "/home/user/code/gulfairus/Covid19/raw_data/cloud/train_test/train_val/train"
VAL_DATA_PATH = "/home/user/code/gulfairus/Covid19/raw_data/cloud/train_test/train_val/val"

TF_ENABLE_ONEDNN_OPTS = os.environ.get("TF_ENABLE_ONEDNN_OPTS")






################## VALIDATIONS #################

env_valid_options = dict(
    DATA_SIZE=["1k", "200k", "all"],
    MODEL_TARGET=["local", "gcs", "mlflow"],
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
