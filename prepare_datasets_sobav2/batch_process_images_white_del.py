import os
import numpy as np
from PIL import Image

def process_images(folder_path, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img_array = np.array(img)

            # Change white pixels to black
            if img_array.shape[2] == 4:  # Check if image has an alpha channel
                white_mask = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 255) & (img_array[:, :, 2] == 255)
                img_array[white_mask] = [0, 0, 0, 255]  # Preserve the alpha channel
            else:
                white_mask = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 255) & (img_array[:, :, 2] == 255)
                img_array[white_mask] = [0, 0, 0]

            # Convert the array back to an image and save
            img_modified = Image.fromarray(img_array)
            img_modified.save(os.path.join(output_folder, filename))
    
    print("Processed Finished")

# Specify the path to your image folder and output folder
folder_path = "./data_only_full_mask/train/labels"
output_folder = "./modified_images"

# Process all images
process_images(folder_path, output_folder)
