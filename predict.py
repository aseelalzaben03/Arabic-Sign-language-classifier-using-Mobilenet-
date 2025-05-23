import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import zipfile

import os
import zipfile
import requests
import tensorflow as tf

MODEL_URL = "https://huggingface.co/Aseelalzaben03/arabic-sign-language/resolve/main/aseel_saved_model.zip"

ZIP_PATH = "aseel_saved_model.zip"
MODEL_DIR = "saved_model"

if not os.path.exists(MODEL_DIR):
    print("⬇️ Downloading zipped model...")
    r = requests.get(MODEL_URL)
    with open(ZIP_PATH, 'wb') as f:
        f.write(r.content)
    print("✅ Download complete.")
    
    print("📂 Extracting model...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    print("✅ Extraction complete.")

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_DIR)
print("✅ Model loaded successfully.")

# الأصناف (Labels)
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

