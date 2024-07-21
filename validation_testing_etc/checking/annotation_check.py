import os
import cv2
import matplotlib.pyplot as plt

def visualize_annotations(image_dir, annotation_dir, output_dir, class_names):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all images in the image directory
    for image_filename in os.listdir(image_dir):
        # Check if file is an image (optional: based on file extension)
        if image_filename.endswith(('.jpg', '.png', '.jpeg')):
            # Corresponding annotation filename
            annotation_filename = os.path.splitext(image_filename)[0] + '.txt'
            
            image_path = os.path.join(image_dir, image_filename)
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            output_path = os.path.join(output_dir, image_filename)
            
            # Load image
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # Load annotations
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    annotations = f.readlines()

                # Parse annotations
                for annotation in annotations:
                    class_id, center_x, center_y, bbox_width, bbox_height = map(float, annotation.split())

                    # Convert normalized coordinates to pixel values
                    center_x *= width
                    center_y *= height
                    bbox_width *= width
                    bbox_height *= height

                    # Calculate top-left and bottom-right coordinates
                    top_left_x = int(center_x - bbox_width / 2)
                    top_left_y = int(center_y - bbox_height / 2)
                    bottom_right_x = int(center_x + bbox_width / 2)
                    bottom_right_y = int(center_y + bbox_height / 2)

                    # Draw the bounding box
                    cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
                    cv2.putText(image, class_names[int(class_id)], (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Save the annotated image
                cv2.imwrite(output_path, image)
            else:
                print(f"Annotation file {annotation_filename} not found for image {image_filename}")

# Example usage
image_dir = 'd:\\GitHub_repositories\\pythonProject\\Bacteria_counter\\validation_testing_etc\\Train_Test_Split\\dataset_train-test\\train\\images\\5034.jpg'
annotation_dir = 'd:\\GitHub_repositories\\pythonProject\\Bacteria_counter\\validation_testing_etc\\Train_Test_Split\\dataset_train-test\\train\\labels\\5034.txt'
output_dir = 'd:\\GitHub_repositories\\pythonProject\\Bacteria_counter\\validation_testing_etc\\checking\\check_out'
class_names = ['0']  # Replace with your class names

visualize_annotations(image_dir, annotation_dir, output_dir, class_names)
