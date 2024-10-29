'''
#zip_cropped_Lara = ["CrossValidation/Cropped.zip"]
#zip_cropped_Meri = ["C:/Users/merit/OneDrive/Escritorio/UNIVERSITAT/3-A ASSIGNATURES/VISION & LEARNING/PROJECT 2 - HELICOBACTER DETECTION/CrossValidation.zip/CrossValidation/Cropped.zip"]
#zip_cropped_Miguel = ["C:/Users/migue/OneDrive/Escritorio/UAB INTELIGENCIA ARTIFICIAL/Tercer Any/3A/Vision and Learning/Challenge 2/Cropped.zip"]

# INICI DEL UNZIP (LARA)
with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for file_name in zip_file.namelist():
            with zip_file.open(file_name) as file
                with zipfile.ZipFile(file, 'r') as zip_file:
                    if file_name.endswith('.zip'):
                        with zip_file.open(file_name) as file:
'''

import zipfile
import os
import random
import shutil

# Path to CrossValidation.zip
crossvalidation_zip_path = "C:/Users/merit/OneDrive/Escritorio/UNIVERSITAT/3-A ASSIGNATURES/VISION & LEARNING/PROJECT 2 - HELICOBACTER DETECTION/CrossValidation.zip"
extraction_dir = 'C:/Users/merit/OneDrive/Escritorio/UNIVERSITAT/3-A ASSIGNATURES/VISION & LEARNING/PROJECT 2 - HELICOBACTER DETECTION/EXTRACT'

# Step 1: Extract CrossValidation.zip and locate Annotated.zip within it
def extract_crossvalidation_zip(cross_zip_path, extract_to):
    with zipfile.ZipFile(cross_zip_path, 'r') as cross_zip:
        cross_zip.extractall(extract_to)
        
        # Locate Annotated.zip within the extracted directory
        annotated_zip_path = os.path.join(extract_to, 'CrossValidation', 'Annotated.zip')
        
        if os.path.exists(annotated_zip_path):
            print(f"Extracting {annotated_zip_path}...")
            extract_nested_zip(annotated_zip_path, extract_to)
        else:
            print("Annotated.zip not found inside CrossValidation.zip")

# Step 2: Function to extract nested zip files (Annotated.zip in this case)
def extract_nested_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        for item in outer_zip.namelist():
            if item.endswith('.zip'):
                with outer_zip.open(item) as nested_zip_file:
                    nested_zip_path = os.path.join(extract_to, item)
                    with open(nested_zip_path, 'wb') as temp_zip:
                        temp_zip.write(nested_zip_file.read())
                    with zipfile.ZipFile(nested_zip_path, 'r') as inner_zip:
                        inner_zip.extractall(extract_to)
                    os.remove(nested_zip_path)
            else:
                outer_zip.extract(item, extract_to)

# Step 3: Extract CrossValidation.zip, and within it, Annotated.zip
extract_crossvalidation_zip(crossvalidation_zip_path, extraction_dir)