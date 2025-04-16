import gradio as gr
from predict import load_model, predict_skin_disease

# Load the model once at the start
MODEL_PATH = "E:/vs/Skin/models/best_model_stage1.keras"
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
