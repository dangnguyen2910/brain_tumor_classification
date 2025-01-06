import subprocess 
import zipfile
import os

zip_file = 'brain-tumor-multimodal-image-ct-and-mri.zip'
subprocess.run("curl -L -o brain-tumor-multimodal-image-ct-and-mri.zip https://www.kaggle.com/api/v1/datasets/download/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri", shell=True, check=True)

with zipfile.ZipFile(zip_file, 'r') as f:
    f.extractall()

os.remove(zip_file)

os.rename("Dataset", "data")

