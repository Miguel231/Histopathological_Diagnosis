from PIL import Image
import zipfile
import os
import random
import shutil

#zip_cropped_Lara = ["CrossValidation/Cropped.zip"]
#zip_cropped_Meri = ["C:/Users/merit/OneDrive/Escritorio/UNIVERSITAT/3-A ASSIGNATURES/VISION & LEARNING/PROJECT 2 - HELICOBACTER DETECTION/CrossValidation.zip/CrossValidation/Cropped.zip"]
#zip_cropped_Miguel = ["C:/Users/migue/OneDrive/Escritorio/UAB INTELIGENCIA ARTIFICIAL/Tercer Any/3A/Vision and Learning/Challenge 2/Cropped.zip"]

#zip_annot_Lara = [""]
#zip_annot_Meri = ["C:/Users/merit/OneDrive/Escritorio/UNIVERSITAT/3-A ASSIGNATURES/VISION & LEARNING/PROJECT 2 - HELICOBACTER DETECTION/CrossValidation.zip/CrossValidation/Annotated.zip"]
zip_annot_Miguel = ["C:/Users/migue/OneDrive/Escritorio/UAB INTELIGENCIA ARTIFICIAL/Tercer Any/3A/Vision and Learning/Challenge 2/Annotated.zip"]

''' # INICI DEL UNZIP (LARA)
with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for file_name in zip_file.namelist():
            with zip_file.open(file_name) as file
                with zipfile.ZipFile(file, 'r') as zip_file:
                    if file_name.endswith('.zip'):
                        with zip_file.open(file_name) as file:
'''                             

# Directory paths
extraction_dir = 'C:/Users/merit/OneDrive/Escritorio/UNIVERSITAT/3-A ASSIGNATURES/VISION & LEARNING/PROJECT 2 - HELICOBACTER DETECTION/EXTRACT' # Directory to store extracted files
subset_dir = 'C:/Users/merit/OneDrive/Escritorio/UNIVERSITAT/3-A ASSIGNATURES/VISION & LEARNING/PROJECT 2 - HELICOBACTER DETECTION/SUBSET DATA' # Directory to store subset
subset_fraction = 0.1                 # Fraction of dataset to sample, e.g., 10%

# Function to extract nested zip files
def extract_nested_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        for item in outer_zip.namelist():
            # Check if the item is a zip file inside the zip
            if item.endswith('.zip'):
                # Create a temporary zip file for nested zip
                with outer_zip.open(item) as nested_zip_file:
                    nested_zip_path = os.path.join(extract_to, item)
                    with open(nested_zip_path, 'wb') as temp_zip:
                        temp_zip.write(nested_zip_file.read())
                    # Extract the nested zip
                    with zipfile.ZipFile(nested_zip_path, 'r') as inner_zip:
                        inner_zip.extractall(extract_to)
                    # Clean up
                    os.remove(nested_zip_path)
            else:
                # Extract other files directly
                outer_zip.extract(item, extract_to)

# Extract files for each user
for zip_path in zip_annot_Miguel:
    if zip_path:  # Ensure path is not empty
        print(f'Extracting {zip_path}...')
        extract_nested_zip(zip_path, extraction_dir)

