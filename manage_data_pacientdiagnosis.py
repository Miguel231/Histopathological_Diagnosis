#patint csv (taules) --> BAIX (0) /ALT (1)

#NO ESTÀ FET, NOMÉS HE COPIAT LO DEL PATHLEVEL MANAGE DATA I DSP MODIFICO

import os
import shutil
import pandas as pd

# Paths to your folders and Excel file
extract_folder = r'C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\EXTRACT_cropped'
excel_path = #add path and change colums name (presence)

# Load the Excel file into a DataFrame
excel_df = pd.read_excel(excel_path)
excel_df['Pat_ID'] = excel_df['Pat_ID'].astype(str)

# Ensure the 'Pat_ID', 'Window_ID', and 'Presence' columns are present
if not all(col in excel_df.columns for col in ['Pat_ID', 'Presence']):
    raise ValueError("The Excel file must contain 'Pat_ID' and 'Presence' columns.")

# Prepare a list to hold records for the TRAIN_DATA_cropped.csv file
train_data_records = []

counter = 0
# Iterate through each file in the EXTRACT folder
for file_name in os.listdir(extract_folder):
    # Skip if not an image file (e.g., in this case we can assume JPEG or PNG files)
    if not (file_name.endswith('.jpg') or file_name.endswith('.png')):
        continue
    
    base_name = os.path.splitext(file_name)[0]
    
    # Split filename by underscore to get Pat_ID and Window_ID
    parts = base_name.split('_')

    # Extract Pat_ID (before the first underscore) and Window_ID (between the first and second underscores)
    pat_id = parts[0]
    
    # Filter the Excel DataFrame for matching Pat_ID and Window_ID
    match = excel_df[(excel_df['Pat_ID'] == pat_id)]
    
    # If a match is found, and it's in the same row, proceed
    if not match.empty:
        presence_value = match['Presence'].iloc[0]
        # Add the matched data to the train_data_records list
        if presence_value == -1:
            train_data_records.append({
                'Pat_ID': pat_id,
                'Presence': 0
            })
        else:
            # Add the matched data to the train_data_records list
            train_data_records.append({
                'Pat_ID': pat_id,
                'Presence': 1
            })
        counter += 1


# Save the train_data_records to a CSV file named TRAIN_DATA_cropped.csv
train_data_df = pd.DataFrame(train_data_records)
train_data_df.to_csv(os.path.join(r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis", 'TRAIN_DATA_cropped.csv'), index=False)

print(f"Total images copied to the USABLE folder: {counter}") 
print("TRAIN_DATA_cropped.csv has been saved with matching data records.")