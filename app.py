import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

import time

# Disable CORS and XSRF protection
st.set_option('server.enableCORS', False)
st.set_option('server.enableXsrfProtection', False)

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Detects 80 common objects

# Load robotic hand images (Ensure both have transparency)
def load_transparent_image(path):
    """ Loads an image with transparency, ensuring it has an alpha channel. """
    img = Image.open(path).convert("RGBA")  # Convert to RGBA for transparency
    return img

open_hand = load_transparent_image("robot_hand_open.png")
closed_hand = load_transparent_image("robot_hand_closed.png")

# COCO class labels (for object names)
CLASS_NAMES = model.names  

# Streamlit UI
st.title("🤖 Industrial Object Detection & Robotic Arm Simulation")
st.write("Capture an image and watch the robotic arm interact with the objects.")

# Initialize camera
cap = cv2.VideoCapture(0)
stframe = st.empty()  # Live camera feed
captured_frame = None  # Store captured image
camera_active = True  # Track camera state

# Buttons for controlling camera
col1, col2 = st.columns(2)
with col1:
    capture_btn = st.button("📸 Capture Image")
with col2:
    stop_camera_btn = st.button("🛑 Stop Camera")

# If "Stop Camera" is pressed, release the camera
if stop_camera_btn:
    cap.release()
    camera_active = False
    st.warning("Camera stopped. Refresh to restart.")

# Live camera feed
while camera_active and captured_frame is None:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access the camera.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    # Capture Image
    if capture_btn:
        captured_frame = frame.copy()
        break

# Function to overlay an image with transparency (No Stretching)
def overlay_image(background, overlay, x, y):
    """ Overlays a transparent image on top of another image without stretching. """
    overlay = overlay.resize((100, 100), Image.Resampling.LANCZOS)  # Resize hand image
    overlay_np = np.array(overlay)

    h, w = overlay_np.shape[:2]

    # Ensure overlay does not go out of bounds
    if x + w > background.shape[1]:
        x = background.shape[1] - w
    if y + h > background.shape[0]:
        y = background.shape[0] - h

    # Extract the alpha mask from the overlay
    if overlay_np.shape[2] == 4:  # Ensure it's RGBA
        alpha_mask = overlay_np[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask

        for c in range(3):  # Blend each channel
            background[y:y+h, x:x+w, c] = (
                alpha_mask * overlay_np[:, :, c] +
                alpha_inv * background[y:y+h, x:x+w, c]
            )

    return background

# Perform object detection and robotic arm simulation
if captured_frame is not None:
    frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)

    # Store detected objects
    objects = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_x, obj_y = (x1 + x2) // 2, (y1 + y2) // 2
            class_id = int(box.cls[0])  
            object_name = CLASS_NAMES[class_id]  

            objects.append((obj_x, obj_y, x1, y1, x2, y2, object_name))

            # Draw bounding boxes with labels
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_rgb, object_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Initial robotic hand position
    hand_x, hand_y = 50, 50

    # Show captured image
    stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    # Move hand to each object and grab it
    for obj_x, obj_y, x1, y1, x2, y2, object_name in objects:
        for _ in range(20):  # Smooth movement to object
            hand_x += (obj_x - hand_x) // 10
            hand_y += (obj_y - hand_y) // 10
            time.sleep(0.05)

            # Overlay open hand (without stretching)
            frame_copy = frame_rgb.copy()
            frame_copy = overlay_image(frame_copy, open_hand, hand_x, hand_y)
            stframe.image(frame_copy, channels="RGB", use_column_width=True)

        time.sleep(0.5)

        # Switch to closed hand
        frame_rgb = overlay_image(frame_rgb, closed_hand, hand_x, hand_y)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # ✅ Show grab message with object name
        st.success(f"✅ Robot grabbed a {object_name}!")
        time.sleep(1)

    st.success("✅ Robotic arm has interacted with all detected objects!")
