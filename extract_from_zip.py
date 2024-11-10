#input = i want patchlevel o patientlevel
#if patch --> annotated folder (extract)
#if pacient --> cropped (extract)

import zipfile
import os

# Paths
crossvalidation_zip_path = "C:/Users/merit/OneDrive/Escritorio/UNIVERSITAT/3-A ASSIGNATURES/VISION & LEARNING/PROJECT 2 - HELICOBACTER DETECTION/CrossValidation.zip"
extraction_dir_annotated = 'C:/Users/merit/OneDrive/Escritorio/UNIVERSITAT/3-A ASSIGNATURES/VISION & LEARNING/PROJECT 2 - HELICOBACTER DETECTION/EXTRACT_annotated'
extraction_dir_cropped = 'C:/Users/merit/OneDrive/Escritorio/UNIVERSITAT/3-A ASSIGNATURES/VISION & LEARNING/PROJECT 2 - HELICOBACTER DETECTION/EXTRACT_cropped'

# Get user input
#level_choice = input("Do you want to analyze by Patch level or Patient level? ").strip().lower()
level_choice = "patient level"

# Extract CrossValidation.zip and locate Annotated.zip or Cropped.zip within it
def extract_crossvalidation_zip(cross_zip_path, extract_to_annotated, extract_to_cropped):
    with zipfile.ZipFile(cross_zip_path, 'r') as cross_zip:
        cross_zip.extractall(extract_to_annotated if level_choice == "patch level" else extract_to_cropped)
        
        # Paths to the nested zips
        annotated_zip_path = os.path.join(extract_to_annotated, 'CrossValidation', 'Annotated.zip')
        cropped_zip_path = os.path.join(extract_to_cropped, 'CrossValidation', 'Cropped.zip')
        
        # Extract based on the user input
        if level_choice == "patch level" and os.path.exists(annotated_zip_path):
            print(f"Extracting {annotated_zip_path}...")
            extract_nested_zip(annotated_zip_path, extract_to_annotated)
        elif level_choice == "patient level" and os.path.exists(cropped_zip_path):
            print(f"Extracting {cropped_zip_path}...")
            extract_nested_zip(cropped_zip_path, extract_to_cropped)
            rename_patient_images(extract_to_cropped)
        else:
            print("Selected zip file not found inside CrossValidation.zip")

# Extract nested zip files
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

# Rename images inside each patient folder
def rename_patient_images(base_dir):
    for root, dirs, _ in os.walk(base_dir):
        for patient_folder in dirs:
            patient_folder_path = os.path.join(root, patient_folder)
            for i, image_file in enumerate(os.listdir(patient_folder_path)):
                image_path = os.path.join(patient_folder_path, image_file)
                if os.path.isfile(image_path):
                    new_name = os.path.join(patient_folder_path, f"{patient_folder}_{i+1}.png")
                    os.rename(image_path, new_name)

# Run extraction
extract_crossvalidation_zip(crossvalidation_zip_path, extraction_dir_annotated, extraction_dir_cropped)