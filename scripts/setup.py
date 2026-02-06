#Set env variables for wandb and kaggle
#download dataset
import os

from utils.env import load_and_setup_env

load_and_setup_env("secrets.json", set_kaggle=True, set_wandb=False)

import kagglehub
download_path = "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation"

# Set the KAGGLEHUB_CACHE environment variable
os.environ['KAGGLEHUB_CACHE'] = download_path

kagglehub.dataset_download("bhuvaneshsingla/synthrad2025-hn")
