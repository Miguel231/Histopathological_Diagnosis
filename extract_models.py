import zipfile
import os

# Ruta del archivo .zip (especifica la ruta completa si está en otra carpeta)
zip_path = 'ruta_al_archivo/archivo.zip'

# Carpeta de destino para los modelos descomprimidos
output_folder = 'save_trainmodels'

# Crear la carpeta de destino si no existe
os.makedirs(output_folder, exist_ok=True)

# Extraer solo los archivos .pth
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for file in zip_ref.namelist():
        if file.endswith('.pth'):
            zip_ref.extract(file, output_folder)
            print(f'{file} extraído en {output_folder}')
