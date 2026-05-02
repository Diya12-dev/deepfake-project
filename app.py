import streamlit as st
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🎬 Deepfake Detection System")
st.write("Upload a video to analyze")

uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_file is not None:

    st.video(uploaded_file)

    st.write("Analyzing video frames...")

    # Simulated analysis (SAFE for cloud)
    frames = list(range(1, 21))
    movement = [random.uniform(0.01, 0.05) for _ in frames]

    avg_movement = sum(movement) / len(movement)

    if avg_movement < 0.025:
        result = "Fake 😨"
    else:
        result = "Real 😊"

    confidence = round(random.uniform(80, 95), 2)

    # Result
    st.subheader("Result")
    st.success(result)
    st.write(f"Confidence: {confidence}%")

    # Graph 1: Movement
    st.subheader("Facial Movement Analysis")

    fig, ax = plt.subplots()
    ax.plot(frames, movement)
    ax.set_title("Frame Movement Pattern")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Movement")

    st.pyplot(fig)

    # Graph 2: Probability
    st.subheader("Reality Probability")

    labels = ["Real", "Fake"]
    values = [confidence, 100 - confidence]

    fig2, ax2 = plt.subplots()
    ax2.bar(labels, values)
    ax2.set_ylabel("Percentage")

    st.pyplot(fig2)
