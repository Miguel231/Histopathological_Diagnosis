import os
import shutil
import pandas as pd

# Path to the CSV file
csv_path = r"C:\Users\larar\OneDrive\Documentos\Escritorio\PROJECT_VISION2\PatientDiagnosis_cropped.csv"
extract_folder = r"C:\Users\larar\OneDrive\Documentos\Escritorio\EXTRACT_cropped"
usable_folder = r'C:\Users\larar\OneDrive\Documentos\Escritorio\USABLE_cropped'

# Load the CSV file into a DataFrame
csv_df = pd.read_csv(csv_path)

# Ensure the 'CODI' and 'DENSITAT' columns are present
if not all(col in csv_df.columns for col in ['CODI', 'DENSITAT']):
    raise ValueError("The CSV file must contain 'CODI' and 'DENSITAT' columns.")

# Define a mapping for 'DENSITAT' values
density_mapping = {
    'NEGATIVA': 0,
    'BAIXA': 1,
    'ALTA': 1
}

# Apply the mapping to the 'DENSITAT' column
csv_df['DENSITAT'] = csv_df['DENSITAT'].map(density_mapping)

# Prepare the output file path for the new CSV before filtering
train_output_path = r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis-5\DATA_TRAIN_cropped.csv"

# Save the updated DataFrame to a new CSV file (before filtering)
csv_df[['CODI', 'DENSITAT']].to_csv(train_output_path, index=False)

print(f"DATA_TRAIN_cropped.csv has been saved with the new DENSITAT mapping.")

# Calculate the number of patients removed
total_patients_before_filtering = len(csv_df)
print(total_patients_before_filtering)
filtered_df = csv_df[csv_df['DENSITAT'] == 0]
total_patients_after_filtering = len(filtered_df)

# Calculate how many patients were removed (erased)
patients_erased = total_patients_before_filtering - total_patients_after_filtering
print(f"Number of patients erased: {patients_erased}")


# Create a list of valid CODI values (those with DENSITAT == 0)
valid_codes = filtered_df['CODI'].astype(str).tolist()

# List all files in the extract folder
image_files = os.listdir(extract_folder)

# Check if the usable folder exists, if not, create it
if not os.path.exists(usable_folder):
    os.makedirs(usable_folder)
    print(f"Created the directory: {usable_folder}")

# Iterate over the image files
for image in image_files:
    # Extract the part of the image name before the first underscore
    image_code = image.split('_')[0]

    # Check if the image code is in the list of valid CODI values
    if image_code in valid_codes:
        # Construct the source and destination file paths
        source_path = os.path.join(extract_folder, image)
        destination_path = os.path.join(usable_folder, image)

        # Copy the image to the usable folder
        shutil.copy(source_path, destination_path)
        print(f"Copied: {image} to usable folder")
