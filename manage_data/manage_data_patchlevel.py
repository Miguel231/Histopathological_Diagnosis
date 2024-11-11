import os
import shutil
import pandas as pd

# Paths to your folders and Excel file
extract_folder = r'C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\EXTRACT_annotated'
usable_folder = r'C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\USABLE_annotated'
excel_path = 'C:/Users/larar/OneDrive/Documentos/Escritorio/Histopathological_Diagnosis-3/Excel Files/HP_WSI-CoordAnnotatedPatches.xlsx'

# Load the Excel file into a DataFrame
excel_df = pd.read_excel(excel_path)
excel_df['Pat_ID'] = excel_df['Pat_ID'].astype(str)
excel_df['Window_ID'] = excel_df['Window_ID'].astype(str)

# Ensure the 'Pat_ID', 'Window_ID', and 'Presence' columns are present
if not all(col in excel_df.columns for col in ['Pat_ID', 'Window_ID', 'Presence']):
    raise ValueError("The Excel file must contain 'Pat_ID', 'Window_ID', and 'Presence' columns.")

# Create the USABLE folder if it doesn't already exist
os.makedirs(usable_folder, exist_ok=True)

# Prepare a list to hold records for the TRAIN_DATA.csv file
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
    window_id_str = parts[1]
    # Remove leading zeros from Window_ID by converting it to an integer and back to string
    window_id = str(int(window_id_str))  # This removes all leading zeros
    
    # Filter the Excel DataFrame for matching Pat_ID and Window_ID
    match = excel_df[(excel_df['Pat_ID'] == pat_id) & (excel_df['Window_ID'] == window_id)]
    
    # If a match is found, and it's in the same row, proceed
    if not match.empty:
        if len(parts) == 2:
            presence_value = match['Presence'].iloc[0]
            if presence_value != 0:
                # Copy the file to the USABLE folder
                src_path = os.path.join(extract_folder, file_name)
                dst_path = os.path.join(usable_folder, file_name)
                shutil.copy(src_path, dst_path)
                
                # Add the matched data to the train_data_records list
                train_data_records.append({
                    'Pat_ID': pat_id,
                    'Window_ID': window_id,
                    'Presence': presence_value
                })
                counter += 1
        else:
            src_path = os.path.join(extract_folder, file_name)
            dst_path = os.path.join(usable_folder, file_name)
            shutil.copy(src_path, dst_path)
            
            # Add the matched data to the train_data_records list
            train_data_records.append({
                'Pat_ID': pat_id,
                'Window_ID': window_id,
                'Presence': -1
            })
            counter += 1

    else:
        # Copy the file to the USABLE folder
        src_path = os.path.join(extract_folder, file_name)
        dst_path = os.path.join(usable_folder, file_name)
        shutil.copy(src_path, dst_path)
        
        # Add the matched data to the train_data_records list
        train_data_records.append({
            'Pat_ID': pat_id,
            'Window_ID': window_id,
            'Presence': -1
        })
        counter += 1


# Save the train_data_records to a CSV file named TRAIN_DATA_annotated.csv
train_data_df = pd.DataFrame(train_data_records)
train_data_df.to_csv(os.path.join(r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis", 'TRAIN_DATA_annotated.csv'), index=False)

# Print completion message and total count
print("Filtered images have been copied to the USABLE folder.")
print(f"Total images copied to the USABLE folder: {counter}") 
print("TRAIN_DATA_annotated.csv has been saved with matching data records.")