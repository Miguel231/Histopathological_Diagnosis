import zipfile
import os
import shutil

# Paths
crossvalidation_zip_path = r"C:\Users\larar\OneDrive\Documentos\Escritorio\CHALLENGE#2_VISION\CrossValidation.zip"
holdout_zip_path = r"C:\Users\larar\OneDrive\Documentos\Escritorio\CHALLENGE#2_VISION\HoldOut.zip"
extraction_dir_annotated = r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis-5\EXTRACT_annotated"
extraction_dir_cropped = r"C:\Users\larar\OneDrive\Documentos\Escritorio\EXTRACT_cropped"
extract_dir_holdout = r"C:\Users\larar\OneDrive\Documentos\Escritorio\EXTRACT_holdout"

# User input
level_choice = "patient level"  # or "patch level"

# Extract CrossValidation.zip and locate Annotated.zip or Cropped.zip within it
def extract_crossvalidation_zip(cross_zip_path, extract_to_annotated, extract_to_cropped):
    with zipfile.ZipFile(cross_zip_path, 'r') as cross_zip:
        cross_zip.extractall(extract_to_annotated if level_choice == "patch level" else extract_to_cropped)
        
        # Paths to nested zips
        annotated_zip_path = os.path.join(extract_to_annotated, 'CrossValidation', 'Annotated.zip')
        cropped_zip_path = os.path.join(extract_to_cropped, 'CrossValidation', 'Cropped.zip')
        
        # Extract based on user choice
        if level_choice == "patch level" and os.path.exists(annotated_zip_path):
            print(f"Extracting {annotated_zip_path}...")
            extract_nested_zip(annotated_zip_path, extract_to_annotated)
        elif level_choice == "patient level" and os.path.exists(cropped_zip_path):
            print(f"Extracting {cropped_zip_path}...")
            extract_nested_zip(cropped_zip_path, extract_to_cropped)
            rename_and_move_patient_images(extract_to_cropped)
        else:
            print("Selected zip file not found inside CrossValidation.zip")

# Extract HoldOut.zip and locate HoldOut.zip inside it
def extract_holdout_zip(holdout_zip_path, extract_to_holdout):
    with zipfile.ZipFile(holdout_zip_path, 'r') as holdout_zip:
        holdout_zip.extractall(extract_to_holdout)
        
        # Path to nested HoldOut.zip inside the extracted HoldOut folder
        holdout_inner_zip_path = os.path.join(extract_to_holdout, 'HoldOut', 'HoldOut.zip')
        
        if os.path.exists(holdout_inner_zip_path):
            print(f"Extracting {holdout_inner_zip_path}...")
            extract_nested_zip(holdout_inner_zip_path, extract_to_holdout)
            rename_and_move_patient_images(extract_to_holdout)
        else:
            print("HoldOut.zip not found inside HoldOut.zip")

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

# Rename images inside each patient folder and move them to EXTRACT_cropped or EXTRACT_holdout
def rename_and_move_patient_images(base_dir):
    for root, dirs, _ in os.walk(base_dir):
        for patient_folder in dirs:
            patient_folder_path = os.path.join(root, patient_folder)

            # Extract patient name up until the first underscore (_)
            patient_name = patient_folder.split('_')[0]

            # Initialize a list to store the renamed images
            renamed_images = []

            for i, image_file in enumerate(os.listdir(patient_folder_path)):
                image_path = os.path.join(patient_folder_path, image_file)
                if os.path.isfile(image_path):
                    # Construct the new name based on patient name and sequence number
                    new_name = os.path.join(base_dir, f"{patient_name}_{i+1}.png")
                    
                    # Rename and move the image to the respective extraction directory
                    shutil.move(image_path, new_name)
                    renamed_images.append(new_name)
                    print(f"Moved and renamed {image_path} to {new_name}")
            
            # Clean up: delete all ZIP files (but keep renamed images)
            for item in os.listdir(patient_folder_path):
                item_path = os.path.join(patient_folder_path, item)
                if item.endswith('.zip') and os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"Deleted {item_path}")

# Run extraction for CrossValidation.zip
extract_crossvalidation_zip(crossvalidation_zip_path, extraction_dir_annotated, extraction_dir_cropped)

# Run extraction for HoldOut.zip
extract_holdout_zip(holdout_zip_path, extract_dir_holdout)
