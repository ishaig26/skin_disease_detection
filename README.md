# 🧠 AI-Based Skin Disease Detector

An end-to-end deep learning pipeline for detecting skin diseases from dermatoscopic images, powered by Convolutional Neural Networks (CNN) and built for medical image classification tasks.
This project uses the HAM10000 dataset, applying modern deep learning practices including data augmentation, class balancing, transfer learning (ResNet), and an interactive Gradio app for predictions.

## 🗂️ Project Structure
```bash
skin_disease_detector/
├── app/
│   ├── app.py               # Gradio-based interactive prediction UI
│   └── predict.py           # Model prediction logic for single images
├── data_preprocessing/
│   ├── clean.py             # Cleans and filters the original dataset
│   └── merge.py             # Merges CSVs or datasets if needed
├── models/                  # Trained and saved model files (.keras, .h5, etc.)
├── src/
│   ├── preprocessing.py     # Image loading, preprocessing & augmentation
│   ├── model.py             # Model architecture definition (ResNet or custom CNN)
│   ├── train.py             # Training script including callbacks and metrics
│   └── dataset.py           # TensorFlow Dataset creation pipeline
├── check.py                 # Utility to check if image paths are valid and readable
├── reclean.py               # Script for re-cleaning / fixing the dataset if needed
├── trial.py                 # Quick script to print and inspect DataFrame columns
└── unzip.py                 # Script to handle zipped datasets locally
```

---

## 💡 Features

- ✅ Data Cleaning & Validation
- ✅ Class Balancing & Augmentation
- ✅ ResNet-based Transfer Learning
- ✅ Early Stopping & Checkpointing
- ✅ Confusion Matrix & Accuracy Reporting
- ✅ Interactive Gradio Web App
- ✅ Easy-to-use Python scripts

---

## 🖼️ Dataset

**Source:** [HAM10000 - Human Against Machine Skin Lesion Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

**Classes:**

- Melanocytic nevi (`nv`)
- Melanoma (`mel`)
- Benign keratosis-like lesions (`bkl`)
- Basal cell carcinoma (`bcc`)
- Actinic keratoses (`akiec`)
- Vascular lesions (`vasc`)
- Dermatofibroma (`df`)

---

## 🚀 How to Use

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/skin_disease_detector.git
cd skin_disease_detector
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
Customize src/train.py if needed, then run:
```bash
python src/train.py
```
Trained models will be saved in /models/.

### 4️⃣ Predict with Gradio Web App

```bash
python app/app.py
```
This will launch a web interface where you can upload images and see predictions.

### 4️⃣ Predict with Gradio Web App
You can validate your model predictions by using the provided predict.py or your own evaluation script.


