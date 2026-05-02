import streamlit as st
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import mediapipe as mp
import imageio.v2 as imageio

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🎬 Deepfake Detection (Stable Version)")
st.write("Upload short video (5–10 sec recommended)")

uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

if uploaded_file is not None:

    st.video(uploaded_file)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    reader = imageio.get_reader(tfile.name)

    movement_scores = []
    prev_landmarks = None

    frame_limit = 30   # 🚨 VERY IMPORTANT (limit frames)

    for i, frame in enumerate(reader):

        if i > frame_limit:
            break

        if i % 3 != 0:
            continue

        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            current_points = np.array([[lm.x, lm.y] for lm in landmarks.landmark])

            if prev_landmarks is not None:
                diff = np.linalg.norm(current_points - prev_landmarks)
                movement_scores.append(diff)

            prev_landmarks = current_points

    # ---------- RESULT ----------
    if len(movement_scores) > 0:
        avg = np.mean(movement_scores)

        if avg < 0.02:
            result = "Fake 😨"
        else:
            result = "Real 😊"
    else:
        result = "No Face Detected"

    st.subheader("Result")
    st.success(result)

    # ---------- GRAPH ----------
    st.subheader("Movement Graph")

    fig, ax = plt.subplots()
    ax.plot(movement_scores)
    ax.set_title("Frame Movement")

    st.pyplot(fig)
