import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

import os
import gdown
import tensorflow as tf

MODEL_URL = "https://drive.google.com/uc?id=1Hn5BId7M-8B8FPwRAYO6mSwk-2xdwOlc"
MODEL_PATH = "final_finetuned_model_updated.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive using gdown...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Download complete.")

model = tf.keras.models.load_model(MODEL_PATH)


# Class labels (must match the training order)
classes = [
    'Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah', 'Heh',
    'Jeem', 'Kaf', 'Khah', 'Laa', 'Lam', 'Meem', 'Noon', 'Qaf', 'Reh', 'Sad',
    'Seen', 'Sheen', 'Tah', 'Teh', 'Teh_Marbuta', 'Thal', 'Theh', 'Waw', 'Yeh', 'Zah', 'Zain'
]

# Prediction function
def predict_image(image: Image.Image) -> str:
    # Resize the image to match the input shape
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    predicted_label = classes[predicted_index]

    return predicted_label
