import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load your YOLOv8 model
@st.cache_resource
def load_model():
    model = YOLO("D:\\Downloads\\results (4)\\runs\\detect\\960px-60_epoch-yolom-augment\\weights\\best.pt")
    return model

model = load_model()

# Function to draw bounding boxes without labels and with thinner lines
def draw_boxes(image, boxes, thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness) 
    return image

# Function to resize image to fit the screen
def resize_image(image, max_width=1200, max_height=800):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image

# Function to perform inference and count bacterial colonies
def count_bacterial_colonies(image):
    image_np = np.array(image)
    results = model(image_np)
    total_colonies = 0

    for result in results:
        total_colonies += len(result.boxes)
        img_with_boxes = draw_boxes(image_np.copy(), result.boxes, thickness=2)
        resized_img = resize_image(img_with_boxes, max_width=1200, max_height=800)

    return img_with_boxes, resized_img, total_colonies

# Streamlit UI
st.title("YOLOv8 Bacterial Colony Counter")

# Create layout: uploader on top and images below
uploaded_file = st.file_uploader("Drag image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    
    # Button to count bacterial colonies
    if st.button("Count"):
        img_with_boxes, resized_img, total_colonies = count_bacterial_colonies(image)
        
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
