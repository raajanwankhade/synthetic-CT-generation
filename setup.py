#Set env variables for wandb and kaggle
#download dataset
import os
import json
secrets_path = 'secrets.json'
with open(secrets_path, 'r') as f:
    secrets = json.load(f)
os.environ["KAGGLE_USERNAME"] = secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = secrets["KAGGLE_KEY"]

import kagglehub
download_path = "/workspace/awesome-synthetic-CT-generation/bhuvanesh"

# Set the KAGGLEHUB_CACHE environment variable
os.environ['KAGGLEHUB_CACHE'] = download_path

kagglehub.dataset_download("bhuvaneshsingla/synthrad2025-hn")
