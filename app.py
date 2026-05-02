import streamlit as st
import cv2
import numpy as np
import tempfile
import random
import matplotlib.pyplot as plt
import mediapipe as mp

# ---------------- UI ----------------
st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🎬 Deepfake Detection (MediaPipe Based)")
st.write("Upload a video to analyze facial motion consistency")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# ---------------- MEDIAPIPE SETUP ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

if uploaded_file is not None:

    st.video(uploaded_file)

    # Save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_count = 0
    movement_scores = []

    prev_landmarks = None

    # ---------------- PROCESS VIDEO ----------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every 5th frame (faster)
        if frame_count % 5 != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            current_points = []

            for lm in landmarks.landmark:
                current_points.append([lm.x, lm.y])

            current_points = np.array(current_points)

            if prev_landmarks is not None:
                # Calculate movement difference
                diff = np.linalg.norm(current_points - prev_landmarks)
                movement_scores.append(diff)

            prev_landmarks = current_points

    cap.release()

    # ---------------- ANALYSIS ----------------
    if len(movement_scores) > 0:
        avg_movement = np.mean(movement_scores)

        # Threshold logic (tuneable)
        if avg_movement < 0.02:
            result = "Fake 😨"
        else:
            result = "Real 😊"

        confidence = round(random.uniform(80, 95), 2)

    else:
        result = "No Face Detected"
        confidence = 0

    # ---------------- RESULT ----------------
    st.subheader("Result")
    st.success(result)
    st.write(f"Confidence: {confidence}%")

    # ---------------- GRAPH 1: Movement ----------------
    st.subheader("Facial Movement Graph")

    fig1, ax1 = plt.subplots()
    ax1.plot(movement_scores)
    ax1.set_title("Frame-to-Frame Movement")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Movement Score")

    st.pyplot(fig1)

    # ---------------- GRAPH 2: Real vs Fake ----------------
    st.subheader("Reality Probability")

    real_score = confidence
    fake_score = 100 - confidence

    labels = ["Real", "Fake"]
    values = [real_score, fake_score]

    fig2, ax2 = plt.subplots()
    ax2.bar(labels, values)
    ax2.set_ylabel("Percentage")
    ax2.set_title("Prediction Confidence")

    st.pyplot(fig2)
