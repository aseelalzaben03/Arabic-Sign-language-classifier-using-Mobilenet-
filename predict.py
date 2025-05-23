import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

# رابط تحميل الموديل من Hugging Face (✅ مباشر وصحيح)
MODEL_URL = "https://huggingface.co/Aseelalzaben03/arabic-sign-language/resolve/main/final_model.keras"
MODEL_PATH = "final_model.keras"

# تحميل الموديل إذا لم يكن موجودًا محليًا
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)
    print("✅ Model downloaded successfully.")

# تحميل الموديل
print("🔄 Loading the model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# الأصناف (Labels)
classes = [
    'Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah', 'Heh',
    'Jeem', 'Kaf', 'Khah', 'Laa', 'Lam', 'Meem', 'Noon', 'Qaf', 'Reh', 'Sad',
    'Seen', 'Sheen', 'Tah', 'Teh', 'Teh_Marbuta', 'Thal', 'Theh', 'Waw', 'Yeh', 'Zah', 'Zain'
]

# دالة التنبؤ بالحرف
def predict_image(image: Image.Image) -> str:
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    predicted_label = classes[predicted_index]
    return predicted_label

