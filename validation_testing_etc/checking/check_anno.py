import cv2
import matplotlib.pyplot as plt

def visualize_yolo_annotation(image_path, annotation_path):
    # Load the image
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Read the annotation file
    with open(annotation_path, 'r') as file:
        annotations = file.readlines()

    # Draw each bounding box
    for annotation in annotations:
        # Parse the annotation
        class_id, x_center, y_center, width, height = map(float, annotation.split())

        # Convert normalized coordinates to absolute values
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        # Calculate the top-left and bottom-right coordinates
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw the rectangle on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Paths to the image and annotation file
image_path = 'd:\\GitHub_repositories\\pythonProject\\Bacteria_counter\\validation_testing_etc\\checking\\3402.jpg'
annotation_path = 'd:\\GitHub_repositories\\pythonProject\\Bacteria_counter\\validation_testing_etc\\checking\\3402.txt'

# Visualize the annotations
visualize_yolo_annotation(image_path, annotation_path)
