import os
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Print current working directory for debugging
print(f"📁 Current working directory: {os.getcwd()}")

# Model file name inside the Hugging Face repo
MODEL_FILENAME = "final_finetuned_mode_updated.keras"  # Correct filename from Hugging Face repo


# Local path where the model will be saved/downloaded
MODEL_PATH = MODEL_FILENAME

# Hugging Face repository ID (username/repo-name)
REPO_ID = "Aseelalzaben03/arabic-sign-language"

# Download the model from Hugging Face if not already downloaded
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Hugging Face Hub...")
    MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    print("✅ Model downloaded successfully.")

# Load the Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (must match the training order)
classes = [
    'Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah', 'Heh',
    'Jeem', 'Kaf', 'Khah', 'Laa', 'Lam', 'Meem', 'Noon', 'Qaf', 'Reh', 'Sad',
    'Seen', 'Sheen', 'Tah', 'Teh', 'Teh_Marbuta', 'Thal', 'Theh', 'Waw', 'Yeh', 'Zah', 'Zain'
]

def predict_image(image: Image.Image) -> str:
    """
    Preprocess the input image and predict the Arabic sign language letter.
    
    Args:
        image (PIL.Image.Image): Input image to classify.
        
    Returns:
        str: Predicted class label.
    """
    # Resize the image to 224x224 pixels (model input size)
    image = image.resize((224, 224))
    
    # Convert image to numpy array and normalize pixel values
    image = np.array(image) / 255.0
    
    # Add batch dimension for model input shape (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    
    # Make prediction using the loaded model
    prediction = model.predict(image)
    
    # Get the index of the class with highest predicted probability
    predicted_index = np.argmax(prediction)
    
    # Map index to class label
    predicted_label = classes[predicted_index]
    
    return predicted_label
