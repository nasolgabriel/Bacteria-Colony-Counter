import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

# Function to convert relative path to absolute path, ensure the drive letter is lowercase, and normalize the path
def clean_path(path):
    # Convert to absolute path if not already
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    # Ensure the drive letter is lowercase on Windows
    if os.name == 'nt':
        path = path[0].lower() + path[1:]
    
    # Normalize the path separators
    path = os.path.normpath(path)
    
    # Remove duplicate folder names
    parts = path.split(os.path.sep)
    seen = set()
    unique_parts = []
    
    for part in parts:
        if part not in seen:
            unique_parts.append(part)
            seen.add(part)
    
    # Reassemble the path
    cleaned_path = os.path.sep.join(unique_parts)
    
    return cleaned_path

# Load your YOLOv8 model with path adjustment
@st.cache_resource
def load_model():
    # Relative path to the model
    relative_path = r"BacteriaCounterV1/960px-60_epoch-yolom-augment/best.pt"
    
    # Clean the path
    model_path = clean_path(relative_path)
    
    print(f"Model path resolved to: {model_path}")  # Debugging output
    model = YOLO(model_path)
    return model

model = load_model()

# Function to draw bounding boxes with confidence scores and thinner lines
def draw_boxes(image, boxes, show_confidence, thickness=2, font_scale=1, font_thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
        confidence = box.conf[0]  # Get confidence score
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness)
        
        if show_confidence:
            # Adjust the font scale and thickness to make the text larger and more readable
            label = f'{confidence:.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
    
    return image


# Function to resize image to fit the screen
def resize_image(image, max_width=1200, max_height=800):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image

# Function to perform inference and count bacterial colonies
def count_bacterial_colonies(image, show_confidence):
    image_np = np.array(image)
    results = model(image_np, max_det=600, conf=0.40, iou=0.55)
    total_colonies = 0

    for result in results:
        total_colonies += len(result.boxes)
        img_with_boxes = draw_boxes(image_np.copy(), result.boxes, show_confidence, thickness=2)
        resized_img = resize_image(img_with_boxes, max_width=1200, max_height=800)

    return img_with_boxes, resized_img, total_colonies

# Streamlit UI
st.title("YOLOv8 Bacterial Colony Counter")

# Create layout: uploader on top and images below
uploaded_file = st.file_uploader("Drag image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    
    # Checkbox to toggle confidence level display
    show_confidence = st.checkbox("Show confidence level", value=True)
    
    # Button to count bacterial colonies
    if st.button("Count"):
        img_with_boxes, resized_img, total_colonies = count_bacterial_colonies(image, show_confidence)
        
        # Create a two-column layout to display images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            st.image(resized_img, caption=f'Processed Image', use_column_width=True)
            st.write(f"Total number of bacterial colonies detected: {total_colonies}")
    else:
        # Display the raw image in the middle if no predictions are made
        st.image(image, caption='Uploaded Image', use_column_width=True)

print("Image upload and display complete.")
