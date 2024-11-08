import os
import shutil
import random

def count_bacteria_in_annotation(annotation_path):
    """Counts the number of lines in the annotation file."""
    try:
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
            return len(lines)
    except FileNotFoundError:
        return 0

def copy_random_images(
    source_images_folder, 
    source_annotations_folder, 
    destination_folder, 
    start_range, 
    end_range, 
    num_images_to_pick,
    bacteria_threshold
):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Get all the image files within the specified range
    all_images = [
        f for f in os.listdir(source_images_folder) 
        if f.endswith(('.jpg', '.png'))
    ]
    
    # Filter images based on the given range
    filtered_images = [
        img for img in all_images 
        if start_range <= int(os.path.splitext(img)[0]) <= end_range
    ]
    
    if len(filtered_images) == 0:
        print("No images found in the specified range.")
        return
    
    selected_images = []
    
    while len(selected_images) < num_images_to_pick:
        # Randomly select an image from the filtered list
        img = random.choice(filtered_images)
        
        # Get the paths for the image and its annotation
        image_source_path = os.path.join(source_images_folder, img)
        annotation_name = os.path.splitext(img)[0] + ".txt"
        annotation_source_path = os.path.join(source_annotations_folder, annotation_name)
        
        # Check if the annotation file exists and validate the bacteria count
        if os.path.exists(annotation_source_path):
            bacteria_count = count_bacteria_in_annotation(annotation_source_path)
            
            if bacteria_count >= bacteria_threshold:
                shutil.copy(image_source_path, destination_folder)
                
                shutil.copy(annotation_source_path, destination_folder)
                
                selected_images.append(img)
                print(f"Copied: {img} with {bacteria_count} colonies.")
                
        # Remove the selected image from the filtered list to avoid re-selection
        filtered_images.remove(img)
        
        # Stop if there are not enough images left to select from
        if len(filtered_images) == 0:
            print("Ran out of images to select. Try a larger range or lower threshold.")
            break
    
    print(f"Successfully copied {len(selected_images)} images and their annotations to '{destination_folder}'.")

# Example usage
source_images_folder = r"validation_testing_etc\Train_Test_Split\dataset_train-test\val\images"
source_annotations_folder = r"validation_testing_etc\Train_Test_Split\dataset_train-test\val\labels"
destination_folder = r"VALIDATION-panel_request/selected"
start_range = 1
end_range = 17000
num_images_to_pick = 300
bacteria_threshold = 30  # Minimum number of colonies required

copy_random_images(
    source_images_folder, 
    source_annotations_folder, 
    destination_folder, 
    start_range, 
    end_range, 
    num_images_to_pick,
    bacteria_threshold
)
