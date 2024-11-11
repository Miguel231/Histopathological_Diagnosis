import pandas as pd

# Load both CSV files
annotations_file = pd.read_csv(r"TRAIN_DATA_annotated.csv")
annotations_file_cropped = pd.read_csv(r"TRAIN_DATA_cropped.csv")

# Step 1: Identify patients based on DENSITY in the cropped file
# Filter patients with DENSITY = 1 and DENSITY = 0
positive_patients = annotations_file_cropped[annotations_file_cropped['DENSITAT'] == 1]['CODI'].unique()
negative_patients = annotations_file_cropped[annotations_file_cropped['DENSITAT'] == 0]['CODI'].unique()

# Step 2: Remove rows without a valid patch-level Presence value in the annotations file
# Filter patches to include only those with non-null Presence values
annotations_file = annotations_file.dropna(subset=['Presence'])

# Step 3: Filter patches from annotations_file for each patient group
# Select patches from positive patients and negative patients
positive_patches = annotations_file[annotations_file['Pat_ID'].isin(positive_patients)]
negative_patches = annotations_file[annotations_file['Pat_ID'].isin(negative_patients)]

# Step 4: Save the filtered patches to separate CSV files
positive_patches.to_csv("positive_patches.csv", index=False)
negative_patches.to_csv("negative_patches.csv", index=False)

print("CSV files created:")
print(" - positive_patches.csv: Patches from patients with DENSITY = 1 and valid Presence values")
print(" - negative_patches.csv: Patches from patients with DENSITY = 0 and valid Presence values")

