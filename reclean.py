import os
import cv2
import pandas as pd

# Update this path to where your images are stored
BASE_DIR = "E:/vs/Skin/data_total/raw/data/all_images"

def is_valid_image(image_id):
    img_path = os.path.join(BASE_DIR, image_id + ".jpg")
    img = cv2.imread(img_path)
    return img is not None

def clean_csv(csv_path, output_path):
    df = pd.read_csv(csv_path)

    # Check if 'image_id' column is available
    if 'image_id' not in df.columns:
        raise ValueError("CSV does not contain 'image_id' column.")

    # Apply validity check
    df['is_valid'] = df['image_id'].apply(is_valid_image)

    # Filter out invalid images
    cleaned_df = df[df['is_valid']].copy()
    cleaned_df.drop(columns=['is_valid'], inplace=True)

    # Save cleaned CSV
    cleaned_df.to_csv(output_path, index=False)
    print(f"✅ Cleaned CSV saved to {output_path}. Removed {len(df) - len(cleaned_df)} corrupted entries.")

# Clean both train and val CSVs
clean_csv(
    "E:/vs/Skin/data_total/raw/data/train.csv",
    "E:/vs/Skin/data_total/raw/data/train_clean.csv"
)

clean_csv(
    "E:/vs/Skin/data_total/raw/data/val.csv",
    "E:/vs/Skin/data_total/raw/data/val_clean.csv"
)


print(is_valid_image("ISIC_0033319"))
