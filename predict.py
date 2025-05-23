import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

MODEL_URL = "https://huggingface.co/Aseelalzaben03/arabic-sign-language/resolve/main/final_model.keras"
MODEL_PATH = "final_model.keras"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)
    if os.path.exists(MODEL_PATH):
        print("✅ Model downloaded successfully.")
        print(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")
    else:
        print("❌ Model download failed!")

print("🔄 Loading the model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    raise e

classes = [
    'Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah', 'Heh',
    'Jeem', 'Kaf', 'Khah', 'Laa', 'Lam', 'Meem', 'Noon', 'Qaf', 'Reh', 'Sad',
    'Seen', 'Sheen', 'Tah', 'Teh', 'Teh_Marbuta', 'Thal', 'Theh', 'Waw', 'Yeh', 'Zah', 'Zain'
]

def predict_image(image: Image.Image) -> str:
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    predicted_label = classes[predicted_index]

    return predicted_label



