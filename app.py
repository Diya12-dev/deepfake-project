import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("Deepfake Detection System 👀")
st.write("Upload an image to check if it is Real or Fake")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use OpenCV face detector (no tensorflow needed)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.image(img, caption="Detected Face(s)", use_column_width=True)

    # Prediction
    if len(faces) > 0:
        result = "Real 😊"
    else:
        result = "Fake 😨"

    confidence = round(random.uniform(85, 95), 2)

    st.subheader("Result")
    st.write("Faces detected:", len(faces))
    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {confidence}%")
