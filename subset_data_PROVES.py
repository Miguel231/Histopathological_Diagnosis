from PIL import Image
import zipfile
import os
import random
import shutil

import zipfile
import os
import shutil

# Define paths
main_zip_path = "C:/Users/larar/OneDrive/Documentos/Escritorio/CHALLENGE#2_VISION/CrossValidation.zip"
output_dir = "cropped_Sample"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Step 1: Extract the main zip file
with zipfile.ZipFile(main_zip_path, 'r') as main_zip:
    main_zip.extractall("temp_main_unzipped")

# Step 2: Look for the Cropped.zip file inside the extracted content
cropped_zip_path = "temp_main_unzipped/CrossValidation/Cropped.zip"

# Verify the Cropped.zip file exists
if not os.path.exists(cropped_zip_path):
    print(f"Error: {cropped_zip_path} not found.")
else:
    # Step 3: Extract Cropped.zip
    with zipfile.ZipFile(cropped_zip_path, 'r') as cropped_zip:
        cropped_zip.extractall("temp_cropped_unzipped")

    # Step 4: Process images in each patient folder
    root_dir = "temp_cropped_unzipped"
    counter = 1

    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)
        
        if os.path.isdir(patient_path):
            images = sorted(os.listdir(patient_path))
            
            for i in range(0, len(images), 200):
                image_file = images[i]
                image_path = os.path.join(patient_path, image_file)
                
                new_name = f"{counter}_{patient_folder}.jpg"
                new_path = os.path.join(output_dir, new_name)
                
                shutil.copy(image_path, new_path)
                counter += 1

    # Cleanup
    shutil.rmtree("temp_main_unzipped")
    shutil.rmtree("temp_cropped_unzipped")

print(f"Images processed and saved in '{output_dir}'")
