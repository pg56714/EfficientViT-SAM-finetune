import cv2
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm


def convert_to_white_mask(color_mask):
    gray_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    return white_mask


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


def main():
    pred_mask_dir = "./results/sam/labels/images/"
    true_mask_dir = "./datasets/test/labels/"
    # true_mask_dir = "./datasets/train/labels/"

    (
        accuracy_scores,
        precision_scores,
        recall_scores,
        f1_scores,
        iou_scores,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )

    pred_files = sorted(os.listdir(pred_mask_dir))
    true_files = sorted(os.listdir(true_mask_dir))

    # Wrap the loop with tqdm for a progress bar
    for pred_file, true_file in tqdm(
        zip(pred_files, true_files), total=len(pred_files), desc="Processing images"
    ):
        pred_path = os.path.join(pred_mask_dir, pred_file)
        true_path = os.path.join(true_mask_dir, true_file)
        pred_mask = cv2.imread(pred_path)
        true_mask = cv2.imread(true_path)

        if pred_mask is None or true_mask is None:
            print(f"Error loading image {pred_file} or {true_file}")
            continue

        pred_mask = convert_to_white_mask(pred_mask)
        true_mask = convert_to_white_mask(true_mask)

        accuracy, precision, recall, f1, iou = evaluate_masks(true_mask, pred_mask)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        iou_scores.append(iou)

    # Calculate and print average metrics after processing all images
    print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"Average Precision: {np.mean(precision_scores):.4f}")
    print(f"Average Recall: {np.mean(recall_scores):.4f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
    print(f"Average mIoU: {np.mean(iou_scores):.4f}")


if __name__ == "__main__":
    main()
