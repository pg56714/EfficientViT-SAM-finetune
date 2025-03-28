import json

with open("./SOBA_train_relation_v2.json", "r") as file:
    data = json.load(file)

results = []

for image in data["images"]:
    image_info = {
        "image_name": image["image_name"],
        "file_name": image["file_name"],
        "full_mask_path": image["full_mask_path"],
        "object_mask_path": image["object_mask_path"],
        "shadow_mask_path": image["shadow_mask_path"],
    }
    results.append(image_info)

# for result in results:
# print(result)

print(f"Total number of image_name: {len(results)}")
