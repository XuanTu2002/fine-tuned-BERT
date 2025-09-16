import tarfile
import os

def compress_model():
    if not os.path.exists("model_save"):
        print("Error: model_save directory not found!")
        return
        
    print("Compressing model_save directory...")
    with tarfile.open("model_save.tar.gz", "w:gz") as tar:
        tar.add("model_save")
    print("Compression complete! Created model_save.tar.gz")

if __name__ == "__main__":
    compress_model()
