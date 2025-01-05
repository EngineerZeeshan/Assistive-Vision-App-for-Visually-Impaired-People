import os
import streamlit as st
from ultralytics import YOLO
import cv2
from gtts import gTTS
import tempfile
from datetime import datetime, timedelta

# Load YOLOv8 model
yolo = YOLO("yolov8n.pt")

# Streamlit app layout
st.set_page_config(page_title="Assistive Vision App", layout="wide")

st.title("Object Detection & Assistive Vision App for Visually Impaired People")
st.write("This application provides real-time object recognition and optional audio alerts.")

# Placeholder for video frames
stframe = st.empty()

# User controls
ip_webcam_url = st.text_input("Enter your IP Webcam URL (e.g., http://<ip-address>:8080/video):")
start_detection = st.button("Start Detection")
audio_activation = st.checkbox("Enable Audio Alerts", value=False)

# Categories for audio alerts
alert_categories = {"person", "cat", "dog", "knife", "fire", "gun"}

# Dictionary to store the last alert timestamp for each object
last_alert_time = {}
alert_cooldown = timedelta(seconds=10)  # 10-second cooldown for alerts


def play_audio_alert(label):
    """Generate and play an audio alert."""
    alert_text = f"Alert! A {label} is detected."
    tts = gTTS(alert_text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    os.system(f"mpg123 {temp_file.name}")
    os.remove(temp_file.name)


def process_frame(frame):
    """Process a single video frame for object detection."""
    results = yolo(frame)
    result = results[0]

    detected_objects = []
    for box in result.boxes:
        label = result.names[int(box.cls[0])]
        detected_objects.append(label)

    return detected_objects, frame


# Main logic
if start_detection:
    st.success("Object detection started.")
    try:
        cap = cv2.VideoCapture(ip_webcam_url)
        if not cap.isOpened():
            st.error("Could not access the webcam. Please check your camera settings.")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video. Please check your camera.")
                    break

                detected_objects, processed_frame = process_frame(frame)

                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_column_width=True)

                if audio_activation:
                    current_time = datetime.now()
                    for label in detected_objects:
                        if label in alert_categories and (
                            label not in last_alert_time
                            or current_time - last_alert_time[label] > alert_cooldown
                        ):
                            play_audio_alert(label)
                            last_alert_time[label] = current_time

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
