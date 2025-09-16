import hashlib
import os

def calculate_hash():
    file_path = "model_save.tar.gz"
    if not os.path.exists(file_path):
        print("Error: model_save.tar.gz not found!")
        return
    
    # Calculate SHA256 hash
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    print("\nFile details for GitHub Release:")
    print("---------------------------------")
    print(f"File name: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print(f"SHA256: {sha256_hash.hexdigest()}")
    print("\nPlease include these details in your GitHub release notes!")

if __name__ == "__main__":
    calculate_hash()
