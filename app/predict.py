import tensorflow as tf
import numpy as np
import cv2

# Class index to label mapping (HAM10000)
CLASS_NAMES = [
    'Melanocytic nevi (nv)',
    'Melanoma (mel)',
    'Benign keratosis-like lesions (bkl)',
    'Basal cell carcinoma (bcc)',
    'Actinic keratoses (akiec)',
    'Vascular lesions (vasc)',
    'Dermatofibroma (df)'
]

def preprocess_image(image):
    # Convert from PIL Image to RGB numpy array
    image = image.convert("RGB")
    image = np.array(image)

    # Resize to 224x224
    image = cv2.resize(image, (224, 224))

    # Normalize pixel values to 0-1
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("[INFO] ✅ Loaded trained model.")
        return model
    except Exception as e:
        print(f"[ERROR] ❌ Failed to load model: {e}")
        return None

def predict_skin_disease(model, image):
    if model is None:
        return "Model not loaded."

    img = preprocess_image(image)
    preds = model.predict(img)

    print("🔍 Prediction probabilities:", preds)

    class_index = np.argmax(preds, axis=1)[0]
    predicted_class = CLASS_NAMES[class_index]

    print("🎯 Predicted class:", predicted_class)
    return predicted_class

if __name__ == "__main__":
    from PIL import Image

    # Path to a test image (change to your actual image path)
    test_image_path = "E:/vs/Skin/data_total/raw/data/all_images/ISIC_0024320.jpg"

    try:
        image = Image.open(test_image_path)
        model = load_model("E:/vs/Skin/models/best_model_stage1.keras")
        result = predict_skin_disease(model, image)
        print("✅ Final Prediction:", result)
    except Exception as e:
        print(f"[ERROR] Could not load test image: {e}")
