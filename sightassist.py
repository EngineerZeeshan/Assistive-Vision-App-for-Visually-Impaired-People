import os
import streamlit as st
from ultralytics import YOLO
import cv2
import random
import time
from gtts import gTTS
from datetime import datetime, timedelta

# Load YOLOv8 model
yolo = YOLO("yolov8n.pt")

# Streamlit app layout
st.set_page_config(page_title="Assistive Vision App", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
        font-family: "Arial", sans-serif;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        justify-content: center;
        align-items: center;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    .stCheckbox {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display welcome image
welcome_image_path = "bismillah.png"  # Ensure this image exists in the script's directory
if os.path.exists(welcome_image_path):
    st.image(welcome_image_path, use_container_width=True, caption="Bismillah hir Rehman Ar Raheem")
else:
    st.warning("Welcome image not found! Please add 'bismillah.png' in the script directory.")

st.title("Object Detection & Assistive Vision App for Visually Impaired People")
st.write("This application provides real-time object recognition and optional audio alerts.")

# Directory to store temp audio files
audio_temp_dir = "audio_temp_files"
if not os.path.exists(audio_temp_dir):
    os.makedirs(audio_temp_dir)

# Placeholder for video frames
stframe = st.empty()

# User controls (checkbox and buttons)
col1, col2 = st.columns(2)
with col1:
    start_detection = st.button("Start Detection")
with col2:
    stop_detection = st.button("Stop Detection")
audio_activation = st.checkbox("Enable Audio Alerts", value=False)  # Ensure this is defined before usage

# Categories for audio alerts (hazardous objects or living things)
alert_categories = {"person", "cat", "dog", "knife", "fire", "gun"}

# Dictionary to store the last alert timestamp for each object
last_alert_time = {}
alert_cooldown = timedelta(seconds=10)  # 10-second cooldown for alerts


def generate_audio_alert(label, position):
    """Generate an audio alert file."""
    phrases = [
        f"Be careful, there's a {label} on your {position}.",
        f"Watch out! {label} detected on your {position}.",
        f"Alert! A {label} is on your {position}.",
    ]
    caution_note = random.choice(phrases)

    temp_file_path = os.path.join(audio_temp_dir, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp3")

    tts = gTTS(caution_note)
    tts.save(temp_file_path)
    return temp_file_path


def process_frame(frame, audio_mode):
    """Process a single video frame for object detection."""
    results = yolo(frame)
    result = results[0]

    detected_objects = {}
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]

        if audio_mode and label not in alert_categories:
            continue

        frame_center_x = frame.shape[1] // 2
        obj_center_x = (x1 + x2) // 2
        position = "left" if obj_center_x < frame_center_x else "right"

        detected_objects[label] = position

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return detected_objects, frame


# Main logic
if start_detection:
    st.success("Object detection started.")
    try:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            st.error("Could not access the webcam. Please check your camera settings.")
        else:
            while not stop_detection:
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Failed to capture video. Please check your camera.")
                    break

                # Ensure `audio_activation` is properly used after being defined
                detected_objects, processed_frame = process_frame(frame, audio_activation)

                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)

                if audio_activation:  # Safely use `audio_activation` here
                    current_time = datetime.now()
                    for label, position in detected_objects.items():
                        if (
                            label not in last_alert_time
                            or current_time - last_alert_time[label] > alert_cooldown
                        ):
                            temp_file_path = generate_audio_alert(label, position)
                            last_alert_time[label] = current_time

                            # Provide the audio file for download
                            with open(temp_file_path, "rb") as audio_file:
                                st.download_button(
                                    label=f"Download alert for {label} on your {position}",
                                    data=audio_file,
                                    file_name=f"{label}_alert.mp3",
                                    mime="audio/mpeg",
                                )
                time.sleep(0.1)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        if 'video_capture' in locals() and video_capture.isOpened():
            video_capture.release()
            cv2.destroyAllWindows()

elif stop_detection:
    st.warning("Object detection stopped.")
