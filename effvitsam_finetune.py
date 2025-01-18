import os
import cv2
import json
import random
from tqdm import tqdm
from statistics import mean

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
    training_path = "./datasets/train"

    sam_model = (
        create_sam_model(name="xl1", weight_url="./checkpoints/sam/xl1.pt")
        .to(device)
        .train()
    )

    # pretrained_weights_path = "./checkpoints/sam/xl1.pt"
    # sam_model = create_sam_model(name="xl1", pretrained=False).to(device).train()

    # # 加載預訓練的權重
    # checkpoint = torch.load(
    #     pretrained_weights_path, map_location=device, weights_only=True
    # )
    # model_dict = sam_model.state_dict()

    # # 過濾掉不匹配的權重（不符合形狀的權重會被忽略）
    # pretrained_dict = {
    #     k: v
    #     for k, v in checkpoint.items()
    #     if k in model_dict and v.shape == model_dict[k].shape
    # }

    # # 更新現有模型的 state_dict
    # model_dict.update(pretrained_dict)
    # sam_model.load_state_dict(model_dict, strict=False)

    # print(f"成功加載了 {len(pretrained_dict)} 層的權重，其他層將在訓練過程中進行學習。")

    # print(
    #     "Params: {}M".format(
    #         sum(p.numel() for p in sam_model.mask_decoder.parameters()) / 1e6
    #     )
    # )

    num_epochs = 100

    optimizer = torch.optim.Adam(
        sam_model.mask_decoder.parameters(), lr=1e-4, weight_decay=0
    )

    # optimizer = torch.optim.AdamW(
    #     sam_model.mask_decoder.parameters(), lr=5e-5, weight_decay=1e-4
    # )

    # # 使用 CosineAnnealingLR 和 ReduceLROnPlateau 結合
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=num_epochs, eta_min=1e-8
    # )

    # # 加入 ReduceLROnPlateau 作為輔助策略，根據 validation loss 動態調整學習率
    # scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=0.2,
    #     patience=10,
    #     min_lr=1e-9,
    # )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    with open("./datasets/sam_train.json", "r") as f:
        meta = json.load(f)

    img_all = get_training_files(training_path)

    last_model_path = None  # 用來追蹤之前的模型檔案

    for epoch in range(num_epochs):
        epoch_losses = []
        random.shuffle(img_all)
        lab_all = [
            p.replace("images", "labels").replace(".jpg", ".png") for p in img_all
        ]

        img_all_pbar = tqdm(img_all)
        for i, img_path in enumerate(img_all_pbar):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(lab_all[i])
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

            sam_trans = ResizeLongestSide(sam_model.image_size[0])
            original_image_size = image.shape[:2]

            image_tensor = (
                torch.from_numpy(image).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
            )
            resize_image = sam_trans.apply_image(
                image_tensor, original_image_size
            ).squeeze(0)
            input_image_torch = resize_image.contiguous()
            input_size = tuple(input_image_torch.shape[-2:])

            if label.shape != input_size:
                label = cv2.resize(label, input_size[::-1])

            file_name = os.path.basename(img_path).replace("jpg", "png")

            bboxes = meta[file_name]["bbox"]
            bboxes = np.array(bboxes, dtype=np.float32)

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

            gt_mask_resized = torch.from_numpy(
                np.resize(label, (1, 1, label.shape[0], label.shape[1]))
            ).to(device)

            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            upscaled_masks = torch.nn.functional.interpolate(
                upscaled_masks,
                size=gt_binary_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = loss_fn(upscaled_masks, gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            if i % 100 == 0:
                img_all_pbar.set_postfix(loss=mean(epoch_losses))
                image_save = cv2.imread(img_path)
                image_save = cv2.cvtColor(image_save, cv2.COLOR_BGR2RGB)
                mask_save = (upscaled_masks > 0.5)[0].detach().squeeze(0).cpu().numpy()
                mask_save = np.array(mask_save * 255).astype(np.uint8)
                mask_save = np.tile(mask_save[:, :, np.newaxis], 3)
                image_save_resized = cv2.resize(
                    image_save, (mask_save.shape[1], mask_save.shape[0])
                )
                _save = np.concatenate((image_save_resized, mask_save), axis=1)
                cv2.imwrite("./img_logs_sam/{}_{}.jpg".format(epoch, i), _save)

        # # 更新 CosineAnnealingLR 調度器
        # scheduler_cosine.step()

        # # 使用 ReduceLROnPlateau 動態調整學習率
        # scheduler_plateau.step(mean(epoch_losses))

        print(f"EPOCH: {epoch}  Mean loss: {mean(epoch_losses)}")

        # # 刪除上一次儲存的模型
        # if last_model_path and os.path.exists(last_model_path):
        #     os.remove(last_model_path)

        # # 儲存這次的模型
        # last_model_path = f"./checkpoints/{epoch}.pt"
        # torch.save(sam_model.state_dict(), last_model_path)

        # 每100個epoch儲存模型
        if (epoch + 1) % 100 == 0:
            last_model_path = f"./checkpoints/{epoch}.pt"
            torch.save(sam_model.state_dict(), last_model_path)


if __name__ == "__main__":
    main()
