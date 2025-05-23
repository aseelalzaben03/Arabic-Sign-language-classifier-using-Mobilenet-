import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import zipfile

# رابط ملف zip لموديل saved_model على Hugging Face
MODEL_URL = "https://huggingface.co/Aseelalzaben03/arabic-sign-language/resolve/main/saved_model.zip"

# اسم ملف zip اللي بننزله
ZIP_PATH = "saved_model.zip"

# اسم مجلد الموديل بعد فك الضغط
MODEL_PATH = "saved_model"

# لو المجلد مش موجود، ننزل الملف ونفك الضغط
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(ZIP_PATH, 'wb') as f:
        f.write(r.content)
    print("✅ Download complete, extracting now...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_PATH)
    print("✅ Extraction done.")
else:
    print("Model folder already exists, skipping download.")

# تحميل الموديل
print("🔄 Loading the model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    raise e

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

