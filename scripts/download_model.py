import os
import requests
import tarfile
from pathlib import Path

def download_model():
    model_dir = os.getenv("MODEL_DIR", "model_save")
    model_url = os.getenv("MODEL_DOWNLOAD_URL")
    
    if not model_url:
        raise ValueError("MODEL_DOWNLOAD_URL environment variable is not set")
    
    if not Path(model_dir).exists():
        print(f"Downloading model from {model_url}")
        response = requests.get(model_url, stream=True)
        with open("model_save.tar.gz", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting model files...")
        with tarfile.open("model_save.tar.gz", "r:gz") as tar:
            tar.extractall(".")
        os.remove("model_save.tar.gz")
        print("Model downloaded and extracted successfully")

if __name__ == "__main__":
    download_model()
