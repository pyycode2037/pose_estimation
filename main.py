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
        uploaded_video = st.sidebar.file_uploader("上傳影片", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            # 儲存上傳的影片到臨時檔案
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()

            # 開啟影片
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("無法開啟影片檔案")
            else:
                # 獲取影片屬性
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # 影片長度
                duration = frame_count / fps if fps > 0 else 0

                st.write(f"影片資訊：長度：{duration:.2f} 秒，解析度 {width}x{height}，幀率 {fps}，總幀數 {frame_count}")
                
                status_placeholder = st.empty()
                status_placeholder.info("⏳ 影片處理中 …")

                # 創建用於儲存處理後影片的臨時檔案
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_output_path = temp_output.name
                temp_converted_path = temp_output_path + '_converted.mp4'

                # 使用 avc1 編碼器
                try:
                    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width * 2, height))
                    if not out.isOpened():
                        st.error("無法創建輸出影片檔案，請檢查 OpenCV 是否支援 avc1 編碼器。建議重新安裝 opencv-python。")
                        cap.release()
                        os.unlink(tfile.name)
                        os.unlink(temp_output_path)
                        pose.close()
                        st.stop()
                except Exception as e:
                    st.error(f"影片寫入初始化失敗：{str(e)}")
                    cap.release()
                    os.unlink(tfile.name)
                    os.unlink(temp_output_path)
                    pose.close()
                    st.stop()

                # 建立進度條
                progress_bar = st.progress(0)
                progress_text = st.empty()
                frame_idx = 0

                # 處理影片
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 轉換為 RGB 格式
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 確保幀尺寸一致
                    frame = cv2.resize(frame, (width, height))
                    annotated_frame = frame.copy()

                    # 進行姿態估計
                    results = pose.process(frame_rgb)

                    # 更新進度條
                    frame_idx += 1
                    percent_complete = int(frame_idx / frame_count * 100)
                    progress_bar.progress(percent_complete)
                    progress_text.text(f"處理進度：{percent_complete}%")


                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )

                    # 確保合併後的幀尺寸正確
                    combined_frame = np.hstack((frame, annotated_frame))
                    
                    out.write(combined_frame)

                # 釋放資源
                cap.release()
                out.release()
                progress_bar.empty()

                # 使用 FFmpeg 轉碼確保相容性
                try:
                    stream = ffmpeg.input(temp_output_path)
                    stream = ffmpeg.output(stream, temp_converted_path, vcodec='libx264', acodec='aac', format='mp4', pix_fmt='yuv420p')
                    ffmpeg.run(stream, overwrite_output=True)
                except ffmpeg.Error as e:
                    st.error(f"FFmpeg 轉碼失敗：{e.stderr.decode()}")
                    os.unlink(tfile.name)
                    os.unlink(temp_output_path)
                    pose.close()
                    st.stop()

                # 顯示處理後的影片
                status_placeholder.empty()
                st.subheader("原始影片與姿態估計結果")
                try:
                    with open(temp_converted_path, "rb") as f:
                        st.video(f, format="video/mp4")
                except Exception as e:
                    st.error(f"影片顯示失敗：{str(e)}")

                # 清理臨時檔案
                try:
                    os.unlink(tfile.name)
                    os.unlink(temp_output_path)
                    os.unlink(temp_converted_path)
                except:
                    st.warning("臨時檔案清理失敗")

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