import os
import shutil
import random
from collections import defaultdict

# Define directories
data_dir = r"C:\Users\larar\OneDrive\Documentos\Escritorio\EXTRACT_cropped"  # Replace with the path to your data_dir
output_dir = 'USABLE_cropped'  # Path to store the cropped images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store images by CODI tag
codi_dict = defaultdict(list)

# Traverse the data directory to categorize images by their CODI tag
for filename in os.listdir(data_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add more extensions if needed
        # Extract the CODI tag (part before the first underscore)
        codi_tag = filename.split('_')[0]
        # Store the file path under the corresponding CODI tag
        codi_dict[codi_tag].append(filename)

# For each CODI tag, select the first 3 images and copy them
for codi_tag, filenames in codi_dict.items():
    # Ensure there are at least 3 images to select from
    if len(filenames) >= 3:
        # Shuffle the list of filenames for random selection
        random.shuffle(filenames)
        
        # Select the first 3 random images
        selected_files = filenames[:10]

        # Copy the selected images to the output directory
        for selected_file in selected_files:
            # Define the source file path
            src_file = os.path.join(data_dir, selected_file)
            # Define the destination file path
            dst_file = os.path.join(output_dir, selected_file)
            
            # Copy the file to the new directory
            shutil.copy(src_file, dst_file)

print(f"3 random images per CODI tag have been copied to {output_dir}")