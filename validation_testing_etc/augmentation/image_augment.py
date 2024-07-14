import os
import cv2
import numpy as np

def load_image_and_annotation(image_path, annotation_path):
    image = cv2.imread(image_path)
    with open(annotation_path, 'r') as file:
        annotations = [line.strip().split() for line in file]
    return image, annotations

def save_image_and_annotation(image, annotations, image_path, annotation_path):
    cv2.imwrite(image_path, image)
    with open(annotation_path, 'w') as file:
        for ann in annotations:
            file.write(' '.join(map(str, ann)) + '\n')

def rotate_image_and_annotation(image, annotations, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # Rotate annotations
    rotated_annotations = []
    for ann in annotations:
        cls, x_center, y_center, width, height = map(float, ann)
        # Convert normalized coordinates to pixel coordinates
        x_center_pixel = x_center * w
        y_center_pixel = y_center * h
        width_pixel = width * w
        height_pixel = height * h

        # Rotate the center point
        new_center = np.dot(M, np.array([x_center_pixel, y_center_pixel, 1]))
        new_x_center_pixel, new_y_center_pixel = new_center[:2]

        # Convert back to normalized coordinates
        new_x_center = new_x_center_pixel / w
        new_y_center = new_y_center_pixel / h

        rotated_annotations.append([cls, new_x_center, new_y_center, width, height])

    return rotated_image, rotated_annotations

def flip_image_and_annotation(image, annotations, flip_code):
    flipped_image = cv2.flip(image, flip_code)
    (h, w) = image.shape[:2]

    # Flip annotations
    flipped_annotations = []
    for ann in annotations:
        cls, x_center, y_center, width, height = map(float, ann)
        if flip_code == 1:  # Horizontal flip
            x_center = 1 - x_center
        elif flip_code == 0:  # Vertical flip
            y_center = 1 - y_center
        elif flip_code == -1:  # Both horizontal and vertical flip
            x_center = 1 - x_center
            y_center = 1 - y_center
        flipped_annotations.append([cls, x_center, y_center, width, height])
    
    return flipped_image, flipped_annotations

def augment_dataset(dataset_dir, output_dir):
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    augmented_images_dir = os.path.join(output_dir, 'augmented_images')
    augmented_labels_dir = os.path.join(output_dir, 'augmented_labels')

    if not os.path.exists(augmented_images_dir):
        os.makedirs(augmented_images_dir)
    if not os.path.exists(augmented_labels_dir):
        os.makedirs(augmented_labels_dir)

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_dir, filename)
            annotation_path = os.path.join(labels_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

            image, annotations = load_image_and_annotation(image_path, annotation_path)

            # Rotate images and annotations
            for angle in [90, 180, 270]:
                rotated_image, rotated_annotations = rotate_image_and_annotation(image, annotations, angle)
                rotated_image_path = os.path.join(augmented_images_dir, f"{filename.split('.')[0]}_rot{angle}.jpg")
                rotated_annotation_path = rotated_image_path.replace(augmented_images_dir, augmented_labels_dir).replace('.jpg', '.txt')
                save_image_and_annotation(rotated_image, rotated_annotations, rotated_image_path, rotated_annotation_path)

            # Flip images and annotations
            for flip_code, flip_name in [(1, 'hflip'), (0, 'vflip'), (-1, 'hvflip')]:
                flipped_image, flipped_annotations = flip_image_and_annotation(image, annotations, flip_code)
                flipped_image_path = os.path.join(augmented_images_dir, f"{filename.split('.')[0]}_{flip_name}.jpg")
                flipped_annotation_path = flipped_image_path.replace(augmented_images_dir, augmented_labels_dir).replace('.jpg', '.txt')
                save_image_and_annotation(flipped_image, flipped_annotations, flipped_image_path, flipped_annotation_path)

# Usage
dataset_dir = 'D:\\Downloads\\labels_cof-colony-annotation_2024-07-12-01-50-42'
output_dir = 'validation_testing_etc/augmentation'
augment_dataset(dataset_dir, output_dir)
