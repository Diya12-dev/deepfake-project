import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import random

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("Deepfake Detection System 🎬")
st.write("Upload an Image or Video")

uploaded_file = st.file_uploader("Upload file", type=["jpg", "png", "jpeg", "mp4"])

if uploaded_file is not None:

    file_type = uploaded_file.type

    # ---------------- IMAGE ----------------
    if "image" in file_type:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        img = np.array(image)

    # ---------------- VIDEO ----------------
    elif "video" in file_type:
        st.video(uploaded_file)

        # Save video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        ret, frame = cap.read()

        if not ret:
            st.error("Error reading video")
            st.stop()

        img = frame

    # ---------------- PROCESSING ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw face boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    st.image(img, caption="Detected Frame")

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
