import os
import zipfile
import requests
import tensorflow as tf
import numpy as np
from PIL import Image

# روابط ملفات الموديل من Hugging Face
PB_URL = "https://huggingface.co/Aseelalzaben03/arabic-sign-language/resolve/main/saved_model.pb"
VARIABLES_ZIP_URL = "https://huggingface.co/Aseelalzaben03/arabic-sign-language/resolve/main/variables.zip"

# مسارات الحفظ المحلي
MODEL_DIR = "saved_model"
VARIABLES_DIR = os.path.join(MODEL_DIR, "variables")
PB_PATH = os.path.join(MODEL_DIR, "saved_model.pb")
ZIP_PATH = "variables.zip"

# تحميل saved_model.pb إن لم يكن موجودًا
if not os.path.exists(PB_PATH):
    print("⬇️ Downloading saved_model.pb...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(PB_PATH, 'wb') as f:
        f.write(requests.get(PB_URL).content)
    print("✅ saved_model.pb downloaded.")

# تحميل variables.zip وفك الضغط إن لم تكن موجودة
if not os.path.exists(VARIABLES_DIR):
    print("⬇️ Downloading variables.zip...")
    with open(ZIP_PATH, 'wb') as f:
        f.write(requests.get(VARIABLES_ZIP_URL).content)

    print("📂 Extracting variables.zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(VARIABLES_DIR)
    print("✅ variables extracted.")

# تحميل الموديل
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_DIR)
print("✅ Model loaded successfully.")

# قائمة الحروف العربية (حسب ترتيب التدريب)
classes = [
    'Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah', 'Heh',
    'Jeem', 'Kaf', 'Khah', 'Laa', 'Lam', 'Meem', 'Noon', 'Qaf', 'Reh', 'Sad',
    'Seen', 'Sheen', 'Tah', 'Teh', 'Teh_Marbuta', 'Thal', 'Theh', 'Waw', 'Yeh', 'Zah', 'Zain'
]

# دالة التنبؤ بالصورة
def predict_image(image: Image.Image) -> str:
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    return classes[predicted_index]
