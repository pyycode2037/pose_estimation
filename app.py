import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
import mediapipe as mp
import os
import ffmpeg


st.set_page_config(page_title="pose estimation", layout="wide")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_image(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    with mp_pose.Pose(static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 轉成 RGB 供 Mediapipe 使用
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # 回轉為 BGR 以供 OpenCV 使用
            annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

            frames.append(annotated_image)
        cap.release()
    return frames

def main():

    # 初始化 session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    st.sidebar.title("人體姿態預測")
    input_type = st.sidebar.radio("選擇輸入類型", ("單張影像", "影片", "即時影像"))
    
    if input_type == "單張影像":
        st.title("人體姿態預測 - 單張影像")
        # 關閉攝影機
        st.session_state.camera_active = False

        # 影像上傳
        uploaded_file = st.sidebar.file_uploader("請上傳影像檔案", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert("RGB"))
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            result_img = process_image(image_bgr.copy())
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="原始影像")
            with col2:
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="姿態預測結果")
    
    elif input_type == "影片":
        st.title("人體姿態預測 - 影片處理")
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # 關閉攝影機
        st.session_state.camera_active = False

        # 影像上傳
        uploaded_video = st.file_uploader("請上傳影片", type=["mp4", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            with mp_pose.Pose() as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    stframe.image(frame, channels="BGR", use_column_width=True)
            cap.release()
            os.unlink(tfile.name)
        # 釋放 MediaPipe 資源
        pose.close()
    
    else:  # 即時影像
        st.title("人體姿態預測 - 即時影像")
        st.info("請開啟攝影機進行即時姿態預估")
        run = st.sidebar.button("開始")
        stop = st.sidebar.button("停止")
        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        if run:
            cap = cv2.VideoCapture(0)
            with mp_pose.Pose() as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or stop:
                        break
                    result = frame.copy()
                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(result, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    result_placeholder.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB", caption="姿態預測結果")
                    
            cap.release()

    

if __name__ == "__main__":
    main()
