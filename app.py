import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- 🎨 Custom Styles (Pista Green & White Theme) ---
st.markdown(
    """
    <style>
        body {
            background-color: #f0f8e6;  /* Pista Green */
            color: green;  /* Text Color */
        }
        .stApp {
            background-color: #f0f8e6; /* Pista Green */
        }
        .stTitle, .stHeader {
            color: green !important;
        }
        .stSidebar {
            background-color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 🏷 Title ---
st.markdown("<h1 style='color: green;'>📏 AI Weighing Machine (Computer Vision-Based)</h1>", unsafe_allow_html=True)

# --- 📤 Upload Image ---
uploaded_file = st.file_uploader("📤 Upload an object image...", type=["jpg", "png", "jpeg"])

# --- 🔍 Reference Object Details ---
st.sidebar.markdown("<h2 style='color: green;'>🔍 Reference Object (for Scaling)</h2>", unsafe_allow_html=True)
ref_weight = st.sidebar.number_input("Weight of Reference Object (grams)", min_value=1, value=5)
ref_size = st.sidebar.number_input("Size of Reference Object (pixels)", min_value=1, value=2000)

# --- ⚙️ Function to Estimate Weight ---
def estimate_weight(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_size = 0  # Default value

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        object_size = w * h  # Approximate area
    
    # Calculate estimated weight
    estimated_weight = (object_size / ref_size) * ref_weight
    return estimated_weight, edges

# --- 📸 Process Image ---
if uploaded_file is not None:
    # Convert the uploaded image to an OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Estimate weight
    weight, processed_image = estimate_weight(image)

    # Display results
    st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)
    st.image(processed_image, caption="🔍 Edge Detection", use_column_width=True)
    
    st.markdown(f"<h2 style='color: green;'>⚖️ Estimated Weight: {weight:.2f} grams</h2>", unsafe_allow_html=True)

st.sidebar.info("⚠️ Use a known object (e.g., a coin) as a reference for accuracy.")
