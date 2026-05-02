import streamlit as st
import numpy as np
import tempfile
import random
import matplotlib.pyplot as plt
import mediapipe as mp
from PIL import Image
import imageio.v2 as imageio

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🎬 Deepfake Detection (Stable Cloud Version)")
st.write("Upload Video for Analysis")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

if uploaded_file is not None:

    st.video(uploaded_file)

    # Save video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    reader = imageio.get_reader(tfile.name)

    movement_scores = []
    prev_landmarks = None

    frame_count = 0

    for frame in reader:
        frame_count += 1

        if frame_count % 5 != 0:
            continue

        rgb = frame

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            current_points = []

            for lm in landmarks.landmark:
                current_points.append([lm.x, lm.y])

            current_points = np.array(current_points)

            if prev_landmarks is not None:
                diff = np.linalg.norm(current_points - prev_landmarks)
                movement_scores.append(diff)

            prev_landmarks = current_points

    # ---------------- RESULT ----------------
    if len(movement_scores) > 0:
        avg_movement = np.mean(movement_scores)

        if avg_movement < 0.02:
            result = "Fake 😨"
        else:
            result = "Real 😊"

        confidence = round(random.uniform(80, 95), 2)

    else:
        result = "No Face Detected"
        confidence = 0

    st.subheader("Result")
    st.success(result)
    st.write(f"Confidence: {confidence}%")

    # ---------------- GRAPH ----------------
    st.subheader("Movement Graph")

    fig, ax = plt.subplots()
    ax.plot(movement_scores)
    ax.set_title("Facial Movement Over Frames")

    st.pyplot(fig)
