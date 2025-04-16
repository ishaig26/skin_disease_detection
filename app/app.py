import os
import gdown
import tensorflow as tf
import gradio as gr
from predict import predict_skin_disease

# Define the Google Drive model URL and local save path
MODEL_URL = "https://drive.google.com/uc?id=1zAeQ108XABvkO6ZbUb4HKcD9pTSso99c"
MODEL_PATH = "models/best_model_stage1.keras"

# Function to download the model if not already present
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("📥 Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded and saved to:", MODEL_PATH)
    else:
        print("✅ Model already exists.")

# Load the model
def load_model():
    download_model()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

# Initialize model once
model = load_model()

# Define Gradio interface function
def gradio_predict(image):
    if model is None:
        return "Model not loaded."
    return predict_skin_disease(model, image)

# Create and launch Gradio app
if __name__ == "__main__":
    gr.Interface(
        fn=gradio_predict,
        inputs=gr.Image(type="pil"),
        outputs="label",
        title="AI-Based Skin Disease Detector",
        description="Upload a skin lesion image and get the predicted disease"
    ).launch(server_name="0.0.0.0", server_port=8080)
