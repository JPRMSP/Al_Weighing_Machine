import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Load your trained model
model = joblib.load("ingredient_model.pkl")

# Apply Custom Styling
st.markdown(
    """
    <style>
    body {
        background-color: #a7c957; /* Pista Green */
        color: white;
    }
    .stApp {
        background-color: #a7c957; /* Pista Green */
    }
    .stButton>button {
        background-color: white;
        color: #a7c957;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input {
        background-color: white;
        color: #333;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("AI-Powered Weighing Machine")

# Upload Image
uploaded_file = st.file_uploader("Upload an Image of the Object", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to OpenCV format
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (128, 128))  # Resize for model
    img_resized = img_resized / 255.0  # Normalize

    # Predict Weight
    img_reshaped = img_resized.reshape(1, 128, 128, 3)
    estimated_weight = model.predict(img_reshaped)[0]

    st.success(f"Estimated Weight: {estimated_weight:.2f} grams")

# Footer
st.markdown("<p style='text-align:center; font-size:14px;'>Powered by AI & Computer Vision</p>", unsafe_allow_html=True)
