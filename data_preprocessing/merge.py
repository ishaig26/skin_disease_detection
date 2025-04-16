import os, shutil

src_1 = 'C:/Users/User/OneDrive/Desktop/vs/Skin/data_total/raw/HAM10000_images_part_1'
src_2 = 'C:/Users/User/OneDrive/Desktop/vs/Skin/data_total/raw/HAM10000_images_part_2'
dest = 'data/all_images'

os.makedirs(dest, exist_ok=True)

for folder in [src_1, src_2]:
    for img in os.listdir(folder):
        src_path = os.path.join(folder, img)
        dest_path = os.path.join(dest, img)
        shutil.copyfile(src_path, dest_path)

print(f"Images merged into {dest}")
