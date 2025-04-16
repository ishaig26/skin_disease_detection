import zipfile
import os

def unzip_file(zip_path, extract_to):
    # Ensure the target directory exists
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted '{zip_path}' to '{extract_to}'")

# Example usage
zip_path = '"C:/Users/User/Downloads/archive_data.zip"'       # Replace with your .zip file path
extract_to = 'C:/Users/User/OneDrive/Desktop/vs/Skin/data/raw'  # Replace with your destination folder path

unzip_file(zip_path, extract_to)
