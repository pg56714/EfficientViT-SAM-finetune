import os
import cv2
import json
import random
from tqdm import tqdm
from statistics import mean

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold

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


def split_train_val(img_all, val_split=0.1):
    random.shuffle(img_all)
    val_size = int(len(img_all) * val_split)
    val_imgs = img_all[:val_size]
    train_imgs = img_all[val_size:]
    return train_imgs, val_imgs


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union != 0 else 0
    return iou


def evaluate_masks(true_mask, pred_mask):
    true_flat = (true_mask.flatten() / 255).astype(np.uint8)
    pred_flat = (pred_mask.flatten() / 255).astype(np.uint8)

    accuracy = accuracy_score(true_flat, pred_flat)
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)

    iou = calculate_iou(true_mask, pred_mask)

    return accuracy, precision, recall, f1, iou


def train(
    device,
    sam_model,
    optimizer,
    loss_fn,
    meta,
    train_imgs,
    val_imgs,
    num_epochs,
    writer,
    early_stopping_patience,
    scheduler=None,
    fold=0,
):
    global_step = 0
    best_val_loss = float("inf")  # 初始化 best_val_loss
    patience_counter = 0  # 初始化 patience_counter

    for epoch in range(num_epochs):
        sam_model.train()
        epoch_losses = []
        random.shuffle(train_imgs)
        lab_all = [
            p.replace("images", "labels").replace(".jpg", ".png") for p in train_imgs
        ]

        img_all_pbar = tqdm(train_imgs)
        for i, img_path in enumerate(img_all_pbar):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(lab_all[i])
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

            # Resize & Process images
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

            global_step = epoch * len(train_imgs) + i

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
                cv2.imwrite(
                    "./img_logs_sam/fold_{}_{}_{}.jpg".format(fold, epoch, i), _save
                )

                writer.add_scalar("Loss/train", mean(epoch_losses), global_step)
        # Validation
        val_results = validate(
            device, sam_model, meta, val_imgs, writer, epoch, global_step
        )
        avg_val_loss = val_results["loss"]

        # 學習率調整
        if scheduler:
            scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(sam_model.state_dict(), f"./checkpoints/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(
            f"EPOCH: {epoch}  Mean loss: {mean(epoch_losses)}  Val loss: {avg_val_loss}"
        )

        torch.save(
            sam_model.state_dict(),
            f"D:/checkpoints/efficientvit_sam_fold_{fold}_{epoch}.pt",
        )
        # torch.save(
        #     sam_model.state_dict(),
        #     f"./checkpoints/efficientvit_sam_fold_{fold}_{epoch}.pt",
        # )


def cross_validation(
    device,
    sam_model,
    loss_fn,
    meta,
    img_all,
    k_folds=5,
    num_epochs=100,
    early_stopping_patience=10,
):
    kf = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(img_all)):
        print(f"Fold {fold+1}/{k_folds}")
        train_imgs = [img_all[i] for i in train_idx]
        val_imgs = [img_all[i] for i in val_idx]

        # Reinitialize the model and optimizer before each fold
        # sam_model = create_sam_model(name="xl1", pretrained=False).to(device)
        optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )

        writer = SummaryWriter(log_dir=f"./logs/fold_{fold+1}")

        # Train and Validate for the current fold
        train(
            device,
            sam_model,
            optimizer,
            loss_fn,
            meta,
            train_imgs,
            val_imgs,
            num_epochs,
            writer,
            early_stopping_patience,
            scheduler,
            fold,
        )

        global_step = num_epochs * len(train_imgs)
        val_results = validate(
            device, sam_model, meta, val_imgs, writer, num_epochs, global_step
        )
        fold_results.append(val_results)

        print(f"Fold {fold+1} Results: {val_results}")

        writer.close()

    # Calculate and log average results over all folds
    avg_accuracy = mean([res["accuracy"] for res in fold_results])
    avg_precision = mean([res["precision"] for res in fold_results])
    avg_recall = mean([res["recall"] for res in fold_results])
    avg_f1 = mean([res["f1"] for res in fold_results])
    avg_iou = mean([res["miou"] for res in fold_results])

    print(
        f"Cross-Validation Results: Average Accuracy={avg_accuracy:.4f}, Precision={avg_precision:.4f}, "
        f"Recall={avg_recall:.4f}, F1 Score={avg_f1:.4f}, mIoU={avg_iou:.4f}"
    )


def validate(device, sam_model, meta, val_imgs, writer, epoch, global_step):
    return evaluate_model(
        device, sam_model, meta, val_imgs, writer, global_step, mode="Validation"
    )


def test(device, sam_model, meta, test_imgs, writer):
    return evaluate_model(device, sam_model, meta, test_imgs, writer, mode="Test")


def evaluate_model(
    device, sam_model, meta, imgs, writer, global_step=None, mode="Validation"
):
    sam_model.eval()
    val_losses = []
    accuracies, precisions, recalls, f1_scores, ious = [], [], [], [], []

    img_all_pbar = tqdm(imgs)

    for i, img_path in enumerate(img_all_pbar):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 讀取 label
        label_path = img_path.replace("images", "labels").replace(".jpg", ".png")
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

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

        # 計算 loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            upscaled_masks, gt_binary_mask
        )
        val_losses.append(loss.item())

        pred_mask = (upscaled_masks > 0.5)[0].detach().cpu().numpy().astype(np.uint8)

        true_mask = convert_to_white_mask(gt_binary_mask.cpu().numpy().squeeze())
        pred_mask = convert_to_white_mask(pred_mask.squeeze())

        accuracy, precision, recall, f1, iou_value = evaluate_masks(
            true_mask, pred_mask
        )
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        ious.append(iou_value)

    avg_loss = mean(val_losses)
    avg_accuracy = mean(accuracies)
    avg_precision = mean(precisions)
    avg_recall = mean(recalls)
    avg_f1 = mean(f1_scores)
    avg_iou = mean(ious)

    print(
        f"{mode} Results: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}, "
        f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1 Score={avg_f1:.4f}, mIoU={avg_iou:.4f}"
    )

    # 記錄到 TensorBoard
    if writer is not None:
        writer.add_scalar(f"{mode}/Loss", avg_loss, global_step)
        writer.add_scalar(f"{mode}/Accuracy", avg_accuracy, global_step)
        writer.add_scalar(f"{mode}/Precision", avg_precision, global_step)
        writer.add_scalar(f"{mode}/Recall", avg_recall, global_step)
        writer.add_scalar(f"{mode}/F1_Score", avg_f1, global_step)
        writer.add_scalar(f"{mode}/mIoU", avg_iou, global_step)

    return {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "miou": avg_iou,
    }


def convert_to_white_mask(mask):
    if mask.dtype != np.uint8 or np.max(mask) > 1:
        _, white_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    else:
        white_mask = mask * 255
    return white_mask


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_path = "./datasets/train"
    test_path = "./datasets/test"

    writer = SummaryWriter(log_dir="./logs")

    # sam_model = create_sam_model(name="xl1", weight_url="./checkpoints/sam/xl1.pt").to(
    #     device
    # )
    sam_model = create_sam_model(name="xl1", pretrained=False).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    with open("./datasets/sam_train.json", "r") as f:
        meta_train = json.load(f)

    img_all_train = get_training_files(training_path)

    early_stopping_patience = 10

    # Perform K-Fold Cross Validation on the entire training set
    cross_validation(
        device,
        sam_model,
        loss_fn,
        meta_train,
        img_all_train,
        k_folds=5,
        num_epochs=100,
        early_stopping_patience=early_stopping_patience,
    )

    # Test
    with open("./datasets/sam_test.json", "r") as f:
        meta_test = json.load(f)

    img_all_test = get_training_files(test_path)
    test(device, sam_model, meta_test, img_all_test, writer)

    writer.close()


if __name__ == "__main__":
    main()
