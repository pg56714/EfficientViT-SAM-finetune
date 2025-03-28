import os

folder_path = "./data_only_full_mask/challenge/labels"
for filename in os.listdir(folder_path):
    if filename.endswith("-1.png"):
        new_name = filename.replace("-1.png", ".png")
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_name}'")
