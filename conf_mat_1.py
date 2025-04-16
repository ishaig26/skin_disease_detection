import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Class labels
CLASS_NAMES = [
    'Melanocytic nevi (nv)',
    'Melanoma (mel)',
    'Benign keratosis-like lesions (bkl)',
    'Basal cell carcinoma (bcc)',
    'Actinic keratoses (akiec)',
    'Vascular lesions (vasc)',
    'Dermatofibroma (df)'
]

# Preprocess function matching your training
def preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image, label

# Create dataset from CSV
def create_dataset_from_csv(csv_path, batch_size=32):
    df = pd.read_csv(csv_path)
    image_paths = df['image_path'].values
    labels = df['label'].values

    # If the paths are relative, convert to absolute paths
    base_path = "E:/vs/Skin/data_total/raw/data/all_images/"
    image_paths = [os.path.join(base_path, os.path.basename(path)) for path in image_paths]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda path, label: preprocess_image(path, label))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, labels

def main():
    # Paths — customize yours!
    MODEL_PATH = "E:/vs/Skin/models/best_model_stage2.keras"
    VAL_CSV_PATH = "E:/vs/Skin/data_total/raw/data/val_clean.csv"

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")

    # Create validation dataset
    val_dataset, y_true = create_dataset_from_csv(VAL_CSV_PATH)

    # Predict
    predictions = model.predict(val_dataset)
    y_pred = np.argmax(predictions, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    plt.figure(figsize=(10,10))
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Skin Disease Detection - Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
