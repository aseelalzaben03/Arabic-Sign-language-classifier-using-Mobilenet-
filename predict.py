import os
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Load model from Hugging Face Hub using token from environment
REPO_ID = "Aseelalzaben03/arabic-sign-language"
MODEL_FILENAME = "final_model.keras"
HF_TOKEN = os.getenv("HF_TOKEN")  # Token from Streamlit secrets

# Download model file
MODEL_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename=MODEL_FILENAME,
    token=HF_TOKEN
)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (must match the training order)
classes = [
    'Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah', 'Heh',
    'Jeem', 'Kaf', 'Khah', 'Laa', 'Lam', 'Meem', 'Noon', 'Qaf', 'Reh', 'Sad',
    'Seen', 'Sheen', 'Tah', 'Teh', 'Teh_Marbuta', 'Thal', 'Theh', 'Waw', 'Yeh', 'Zah', 'Zain'
]

# Prediction function
def predict_image(image: Image.Image) -> str:
    image = image.resize((224, 224))  # Resize image to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)  # Run prediction
    predicted_index = np.argmax(prediction)
    predicted_label = classes[predicted_index]

    return predicted_label
