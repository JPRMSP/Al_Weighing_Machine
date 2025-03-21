import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 🌿 Custom CSS for Pista Green & White Theme
st.markdown(
    """
    <style>
    body {
        background-color: #f5ffe8;  /* Light pista green background */
        color: #2d6a4f;  /* Dark green text */
    }
    .stApp {
        background-color: #f5ffe8;
    }
    .stTitle {
        color: #1b4332;
        text-align: center;
    }
    .stSidebar {
        background-color: #d8f3dc;
        border-radius: 10px;
        padding: 20px;
    }
    .stButton > button {
        background-color: #40916c;
        color: white;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🎯 App Title
st.markdown("<h1 class='stTitle'>📏 AI Weighing Machine (Computer Vision-Based)</h1>", unsafe_allow_html=True)

# 📤 Upload Image
uploaded_file = st.file_uploader("📤 Upload an object image...", type=["jpg", "png", "jpeg"])

# 🎨 Sidebar Styling
st.sidebar.markdown("<h2 style='color:#1b4332;'>🔍 Reference Object (for Scaling)</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#2d6a4f;'>Use a known object (e.g., a coin) for better accuracy.</p>", unsafe_allow_html=True)

# 📏 Reference Object Inputs
ref_weight = st.sidebar.number_input("Weight of Reference Object (grams)", min_value=1, value=5)
ref_size = st.sidebar.number_input("Size of Reference Object (pixels)", min_value=1, value=2000)

# 🔢 Function to estimate weight
def estimate_weight(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_size = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        object_size = max(object_size, w * h)  # Take the largest detected object
    
    # 🧮 Calculate estimated weight
    estimated_weight = (object_size / ref_size) * ref_weight if object_size > 0 else 0
    return estimated_weight, edges

# 📌 Process Image
if uploaded_file is not None:
    # Convert the uploaded image to an OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # 🏋️ Estimate weight
    weight, processed_image = estimate_weight(image)

    # 📸 Display images
    st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)
    st.image(processed_image, caption="🔍 Edge Detection", use_column_width=True)

    # ⚖️ Show estimated weight
    if weight > 0:
        st.success(f"⚖️ Estimated Weight: **{weight:.2f} grams**")
    else:
        st.error("🚨 Could not detect an object properly. Try another image.")

# ℹ️ Sidebar Info
st.sidebar.info("⚠️ Ensure the object is placed clearly in the image with a reference object for better results.")
