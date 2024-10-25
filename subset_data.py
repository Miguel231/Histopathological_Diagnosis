from PIL import Image
import zipfile
zip_path = ["CrossValidation/Cropped.zip"]
with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for file_name in zip_file.namelist():
            with zip_file.open(file_name) as file
                with zipfile.ZipFile(file, 'r') as zip_file:
                    if file_name.endswith('.zip'):
                        with zip_file.open(file_name) as file:
                    