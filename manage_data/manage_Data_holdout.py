#patint csv (taules) --> BAIX (0) /ALT (1)

#mirar file de patient diagnosis: NEGATIVA = 0 (VALOR NEGATIU), BAIXA O POSITIVA = 1 (VALOR POSITIU)

import pandas as pd


# Path to the CSV file
csv_path = r"C:\Users\larar\OneDrive\Documentos\Escritorio\PROJECT_VISION2\PatientDiagnosis_holdout.csv"

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

# Prepare the output file path
output_path = r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis-5\TRAIN_DATA_holdout.csv"

# Save the modified DataFrame to a new CSV file
csv_df[['CODI', 'DENSITAT']].to_csv(output_path, index=False)

print(f"TRAIN_DATA_holdout.csv has been saved with modified DENSITAT values.")