# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import librosa
import tempfile

st.set_page_config(page_title="VeriLens AI", page_icon="🛡️")
st.title("VeriLens AI - Deepfake Lip-Sync Detector")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_file)
    st.write("Analyzing... This may take a few seconds.")

    # -----------------------------
    # Step 1: Audio analysis
    # -----------------------------
    y, sr = librosa.load(video_path, sr=None)
    audio_peaks = np.sum(librosa.effects.split(y, top_db=20))

    # -----------------------------
    # Step 2: Mouth movement analysis
    # -----------------------------
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)
    
    mouth_movement = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                distance = bottom_lip.y - top_lip.y
                if distance > 0.02:  # mouth open threshold
                    mouth_movement += 1

    cap.release()

    # -----------------------------
    # Step 3: Calculate risk score
    # -----------------------------
    if total_frames == 0:
        st.write("Could not read video frames.")
    else:
        lip_sync_ratio = mouth_movement / total_frames
        audio_ratio = audio_peaks / (total_frames / 10)

        risk_score = max(0, min(100, abs(lip_sync_ratio - audio_ratio)*200))
        risk_score = int(risk_score)

        st.write(f"**Deepfake Risk Score:** {risk_score}%")

        if risk_score < 30:
            st.success("Low risk of manipulation")
        elif risk_score < 70:
            st.warning("Medium risk of manipulation")
        else:
            st.error("High risk of manipulation")
