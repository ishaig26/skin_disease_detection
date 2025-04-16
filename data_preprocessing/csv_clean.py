import pandas as pd

df = pd.read_csv('E:/vs/Skin/data_total/raw/HAM10000_metadata.csv')

# Add full image path
df['image_path'] = df['image_id'].apply(lambda x: f"data/all_images/{x}.jpg")

# Check label distribution
print(df['dx'].value_counts())

# Optional: Rename labels
label_map = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Nevus',
    'vasc': 'Vascular lesion'
}
df['label'] = df['dx'].map(label_map)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode class labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['dx'])

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
print("✅ Saved train.csv and val.csv in /data")
