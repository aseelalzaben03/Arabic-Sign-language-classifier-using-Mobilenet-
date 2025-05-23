import streamlit as st
from PIL import Image
from predict import predict_image

st.set_page_config(
    page_title="AI-ISHARAH",
    page_icon="🤟",
    layout="centered"
)

st.title("📸 AI-ISHARAH .... Arabic Sign Language Letter Recognition")
st.markdown("""
👋 Welcome to **AI-ISHARAH**!  
Upload an image of a hand sign and we will predict the corresponding Arabic letter.  
---
""")

uploaded_file = st.file_uploader("📤 Upload an image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 This is the uploaded image", use_column_width=True)

    if st.button("🔍 Recognize Letter"):
        with st.spinner("⏳ Predicting..."):
            try:
                prediction = predict_image(image)
                st.success(f"✅ Predicted letter: **{prediction}** 🎉")
            except Exception as e:
                st.error("❌ An error occurred during prediction. Please check the image or the model.")
                st.exception(e)
else:
    st.info("👈 Please upload an image to get started")

