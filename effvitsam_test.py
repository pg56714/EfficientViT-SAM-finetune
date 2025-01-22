import os
import cv2
import json
from tqdm import tqdm

import torch
import numpy as np

from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.samcore.data_provider.utils import (
    ResizeLongestSide,
    Normalize_and_Pad,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_training_files(path):
    image_dir = os.path.join(path, "images")
    img_all = [os.path.join(image_dir, img) for img in sorted(os.listdir(image_dir))]
    return img_all


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # test_path = "./datasets/train"
    test_path = "./datasets/test"

    sam_model = (
        create_sam_model(name="xl1", weight_url="./checkpoints/sam/99.pt")
        .to(device)
        .eval()
    )

    print(
        "Params: {}M".format(
            sum(p.numel() for p in sam_model.mask_decoder.parameters()) / 1e6
        )
    )

    # with open("./datasets/sam_train.json", "r") as f:
    with open("./datasets/sam_test.json", "r") as f:
        meta = json.load(f)

    img_all = get_training_files(test_path)

    img_all_pbar = tqdm(img_all)

    for i, img_path in enumerate(img_all_pbar):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # transform
        sam_trans = ResizeLongestSide(sam_model.image_size[0])
        original_image_size = image.shape[:2]

        image_tensor = (
            torch.from_numpy(image).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
        )
        resize_image = sam_trans.apply_image(image_tensor, original_image_size).squeeze(
            0
        )
        input_image_torch = resize_image.contiguous()
        input_size = tuple(input_image_torch.shape[-2:])

        file_name = os.path.basename(img_path).replace("jpg", "png")

        bboxes = meta[file_name]["bbox"]
        bboxes = np.array(bboxes)

        with torch.no_grad():
            normalize_and_pad = Normalize_and_Pad(sam_model.image_size[0])

            box = torch.from_numpy(bboxes).to(torch.float32)
            box = sam_trans.apply_boxes(box, original_image_size).to(device)

            if len(box.shape) == 2:
                box = box[:, None, :]

            processed_image = normalize_and_pad(
                {
                    "image": input_image_torch,
                    "masks": torch.empty(
                        (0, *input_image_torch.shape[-2:]), dtype=torch.float32
                    ),
                    "points": torch.tensor([]),
                    "bboxs": box,
                    "shape": input_image_torch.shape[-2:],
                }
            )["image"]
            processed_image = processed_image.to(device)

            processed_image = processed_image.unsqueeze(0)

            image_embedding = sam_model.image_encoder(processed_image)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )

        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)
        upscaled_masks = sam_model.postprocess_masks(
            low_res_masks, input_size, original_image_size
        ).to(device)

        mask_save = (upscaled_masks > 0.5)[0].detach().squeeze(0).cpu().numpy()
        mask_save = np.array(mask_save * 255).astype(np.uint8)

        vi = os.path.basename(os.path.dirname(img_path))
        fi = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        os.makedirs(
            os.path.join("results", "sam", "labels", vi), mode=0o777, exist_ok=True
        )
        cv2.imwrite(os.path.join("results", "sam", "labels", vi, fi), mask_save)


if __name__ == "__main__":
    main()
