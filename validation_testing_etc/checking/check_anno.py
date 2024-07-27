import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load a model
model = YOLO("previous_training/detect-50_epoch-yolon-704px(with_augment)/weights/best.pt")  # your custom-trained YOLO model

def draw_boxes(image, boxes, class_names=None, display_labels=True, color=(0, 255, 0), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)  # Draw rectangle
        if display_labels and class_names:
            class_id = int(box.cls[0])
            label = class_names[class_id]
            confidence = box.conf[0]
            text = f"{label} {confidence:.2f}"
            cv2.putText(image, text, (x1, y1 - 10), font, font_scale, color, font_thickness)
    return image

def visualize_yolo_annotation(image_path, model, class_names=None, display_labels=True, color=(0, 255, 0), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1):
    # Run inference
    results = model(image_path)
    
    # Process results
    for result in results:
        img = cv2.imread(result.path)  # Read the original image
        img = draw_boxes(img, result.boxes, class_names, display_labels, color, thickness, font, font_scale, font_thickness)  # Draw boxes with custom properties

        # Resize the image to fit the screen
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # Make the window resizable
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.imshow('Result', img)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()
        
        # Save the image
        output_path = "result.jpg"
        cv2.imwrite(output_path, img)  # Save the image to disk

    print("Processing complete.")

# Define class names if needed
class_names = ['class1', 'class2', 'class3']  # Replace with your actual class names

# Customize parameters
image_path = 'D:\\Downloads\\Petri_plates\\Petri_plates\\IMG_7812paureginosa_T8_10^-7_188.JPG'
display_labels = True
color = (255, 0, 0)  # Red bounding boxes
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

# Visualize the annotations
visualize_yolo_annotation(image_path, model, class_names, display_labels, color, thickness, font, font_scale, font_thickness)
