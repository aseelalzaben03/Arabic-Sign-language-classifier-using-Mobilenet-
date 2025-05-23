import os
import requests
import tensorflow as tf
from PIL import Image
import numpy as np

MODEL_URL = "https://huggingface.co/Aseelalzaben03/arabic-sign-language/resolve/main/final_finetuned_mode_updated.keras"
MODEL_PATH = "final_finetuned_mode_updated.keras"

# تحميل الموديل إذا مش موجود محلياً
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Model downloaded.")

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded.")

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

