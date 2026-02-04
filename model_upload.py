import os
import json
import argparse
secrets_path = '/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/secrets.json'
with open(secrets_path, 'r') as f:
    secrets = json.load(f)
os.environ["KAGGLE_USERNAME"] = secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = secrets["KAGGLE_KEY"]

import kagglehub

def parse_args():
    parser = argparse.ArgumentParser(description="Upload a model file to Kaggle")
    parser.add_argument("--kaggle_model", type=str,
                        help="Kaggle model path: 'model-name/framework/variant' (username auto-prepended)")
    parser.add_argument("--model_file", type=str,
                        help="Path to the model file to upload")
    return parser.parse_args()

args = parse_args()
KAGGLE_MODEL = args.kaggle_model
kaggle_handle = f"{secrets['KAGGLE_USERNAME']}/{args.kaggle_model}"
MODEL_FILE = args.model_file
print(f"Uploading\n{MODEL_FILE}\nto Kaggle model:\n{kaggle_handle}")
kagglehub.model_upload(
    kaggle_handle,
    MODEL_FILE
)
print("Upload complete.")