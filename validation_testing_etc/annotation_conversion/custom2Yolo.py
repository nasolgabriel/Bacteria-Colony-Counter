import json
import os
import shutil
from PIL import Image

class BacteriaAnnotationConverter:
    def __init__(self, json_folder, image_folder, json_output_folder, image_output_folder, ranges):
        self.json_folder = json_folder
        self.image_folder = image_folder
        self.json_output_folder = json_output_folder
        self.image_output_folder = image_output_folder
        self.ranges = ranges

    def get_dimensions(self, image_path):
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height

    def convert_to_yolov8(self, json_data, image_dimensions):
        image_width, image_height = image_dimensions
        annotations = []

        for label in json_data['labels']:
            class_id = 0  # Assuming 'S.aureus' corresponds to class 0 in YOLO format
            x_center = (label['x'] + label['width'] / 2) / image_width
            y_center = (label['y'] + label['height'] / 2) / image_height
            width = label['width'] / image_width
            height = label['height'] / image_height

            annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

        return annotations

    def save_annotations(self, annotations, output_path):
        with open(output_path, 'w') as file:
            file.write("\n".join(annotations))

    def is_within_ranges(self, filename):
        try:
            file_number = int(filename.split('.')[0])
            for start_range, end_range in self.ranges:
                if start_range <= file_number <= end_range:
                    return True
            return False
        except ValueError:
            return False

    def copy_image(self, src_path, dst_path):
        """
        Copies an image from the source path to the destination path.
        
        :param src_path: The source path of the image.
        :param dst_path: The destination path where the image will be copied.
        """
        try:
            # Ensure the source file exists
            if not os.path.isfile(src_path):
                print(f"Source file {src_path} does not exist.")
                return
            
            # Create the destination directory if it doesn't exist
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Copy the image
            shutil.copy2(src_path, dst_path)
            print(f"Image successfully copied from {src_path} to {dst_path}.")
        
        except Exception as e:
            print(f"Error occurred while copying the image: {e}")

    def process_files(self):
        converted_count = 0  # Counter for the number of JSON files converted

        for filename in os.listdir(self.json_folder):
            if filename.endswith('.json') and self.is_within_ranges(filename):
                json_path = os.path.join(self.json_folder, filename)
                image_path = os.path.join(self.image_folder, filename.replace('.json', '.jpg'))
                output_txt_path = os.path.join(self.json_output_folder, filename.replace('.json', '.txt'))
                output_image_path = os.path.join(self.image_output_folder, filename.replace('.json', '.jpg'))

                # Load JSON data
                with open(json_path, 'r') as file:
                    json_data = json.load(file)

                # Get image dimensions
                image_dimensions = self.get_dimensions(image_path)

                # Convert JSON to YOLOv8 format
                annotations = self.convert_to_yolov8(json_data, image_dimensions)

                # Save the YOLOv8 annotations to a file
                self.save_annotations(annotations, output_txt_path)

                # Copy the corresponding image to the output folder
                self.copy_image(image_path, output_image_path)

                print(f"YOLOv8 annotations saved to {output_txt_path} and image saved to {output_image_path}")
                converted_count += 1  # Increment the counter

        print(f"Total number of JSON files converted: {converted_count}")

# Inputs
json_folder = 'd:\\GitHub_repositories\\pythonProject\\Bacteria_counter\\AGAR_dataset\\AGAR_dataset\\dataset'
image_folder = 'd:\\GitHub_repositories\\pythonProject\\Bacteria_counter\\AGAR_dataset\\AGAR_dataset\\dataset'

# Output directories
json_output_folder = 'd:\\GitHub_repositories\\pythonProject\\Bacteria_counter\\validation_testing_etc\\annotation_conversion\\testing data\\directory\\annotation'
image_output_folder = 'd:\\GitHub_repositories\\pythonProject\\Bacteria_counter\\validation_testing_etc\\annotation_conversion\\testing data\\directory\\images'

ranges = [(1, 308),
    (2089, 2711),
    (11738, 11760),
    (309, 1302),
    (2712, 8709),
    (11761, 12617),
    (12994, 17417)
]

converter = BacteriaAnnotationConverter(json_folder, image_folder, json_output_folder, image_output_folder, ranges)
converter.process_files()
