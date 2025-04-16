import os
import gdown
import gradio as gr
from predict import load_model, predict_skin_disease

# Google Drive model file link
MODEL_URL = 'https://drive.google.com/uc?id=1zAeQ108XABvkO6ZbUb4HKcD9pTSso99c'

# Path where the model will be stored
MODEL_PATH = "models/best_model_stage1.keras"

# Function to create the directory if it doesn't exist
def create_model_directory():
    if not os.path.exists('models'):
        os.makedirs('models')

# Function to download the model if not already downloaded
def download_model():
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully.")

# Create models directory if it doesn't exist
create_model_directory()

# In Render, it's good practice to pre-upload the model, but if you want to download dynamically:
if not os.path.exists(MODEL_PATH):
    download_model()

# Load the model once at the start
model = load_model(MODEL_PATH)

# Define the Gradio interface function
def gradio_predict(image):
    if model is None:
        return "Model not loaded."

    return predict_skin_disease(model, image)

# Create and launch the Gradio interface
gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="AI-Based Skin Disease Detector",
    description="Upload a skin lesion image and get the predicted disease"
).launch(server_name="0.0.0.0", server_port=8080)
