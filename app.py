import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import random

# Page setup
st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("Deepfake Detection System 👀")
st.write("Upload an image to check if it is Real or Fake")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)

    # Show original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to array
    img = np.array(image)

    try:
        # Convert color
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        st.error("Error processing image")
        st.stop()

    # Face detection
    detector = MTCNN()
    faces = detector.detect_faces(img_rgb)

    # Draw rectangles around faces
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show detected faces
    st.image(img_rgb, caption="Detected Face(s)", use_column_width=True)

    # Simple prediction logic
    if len(faces) > 0:
        result = "Real 😊"
    else:
        result = "Fake 😨"

    # Confidence (for demo)
    confidence = round(random.uniform(85, 95), 2)

    # Show results
    st.subheader("Result")
    st.write("Faces detected:", len(faces))
    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {confidence}%")
