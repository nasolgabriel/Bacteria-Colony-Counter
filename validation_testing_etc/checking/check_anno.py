import cv2
import matplotlib.pyplot as plt

def visualize_yolo_annotation(image_path, annotation_path):
    # Load the image
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Read the annotation file
    with open(annotation_path, 'r') as file:
        annotations = file.readlines()

    # Initialize a counter for the bounding boxes
    num_bounding_boxes = 0

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

        # Increment the bounding box counter
        num_bounding_boxes += 1

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Print the number of bounding boxes
    print(f'Number of bounding boxes: {num_bounding_boxes}')

# Paths to the image and annotation file
image_path = 'D:\Downloads\labels_cof-colony-annotation_2024-07-12-01-50-42\images\IMG_1793.jpg'
annotation_path = 'D:\Downloads\labels_cof-colony-annotation_2024-07-12-01-50-42\labels\IMG_1793.txt'

# Visualize the annotations and print the number of bounding boxes
visualize_yolo_annotation(image_path, annotation_path)
