import os
import shutil
import json


def batch_process_images(json_path, base_folder, image_folder, mask_folder):

    with open(json_path, "r") as file:
        data = json.load(file)

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    for image in data["images"]:
        try:
            file_name = os.path.join(base_folder, image["file_name"])
            if os.path.exists(file_name):
                shutil.copy(file_name, image_folder)
            else:
                print(f"File not found: {file_name}")

            full_mask_path = os.path.join(base_folder, image["full_mask_path"])
            # object_mask_path = os.path.join(base_folder, image["object_mask_path"])
            # shadow_mask_path = os.path.join(base_folder, image["shadow_mask_path"])

            # for mask_path in [full_mask_path, object_mask_path, shadow_mask_path]:
            for mask_path in [full_mask_path]:
                if os.path.exists(mask_path):
                    shutil.copy(mask_path, mask_folder)
                else:
                    print(f"File not found: {mask_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

    print("Processed Finished")


# # Train
# json_path = "./SOBA_train_relation_v2.json"
# base_folder = "./SOBAv2"
# image_folder = "./data/train/images"
# mask_folder = "./data/train/labels"

# # Test
# json_path = "./SOBA_val_v2.json"
# base_folder = "./SOBAv2"
# image_folder = "./data/test/images"
# mask_folder = "./data/test/labels"

# challenge
json_path = "./SOBA_challenge_v2.json"
base_folder = "./SOBAv2"
image_folder = "./data/challenge/images"
mask_folder = "./data/challenge/labels"

batch_process_images(json_path, base_folder, image_folder, mask_folder)
