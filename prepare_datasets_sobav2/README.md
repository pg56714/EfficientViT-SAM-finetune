# Prepare Datasets

## Getting Started

1. Move Photos

```
python batch_process_images.py
```

2. Rename

```
python batch_process_names.py
```

3. Remove Unnecessary White Mask

```
python batch_process_images_white_del.py
```

4. Confirm Image Count

```
python count_image_names.py
```

## Prerequisites

Please download and extract the [SOBAv2 Dataset](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP). Extract the contents of `SOBA_v2/SOBA/SOBA` to the `prepare_datasets_sobav2/SOBAv2` directory, and extract the contents of `SOBA_v2/SOBA/annotations` to the `prepare_datasets_sobav2` directory.

I only let the sam model train the full_mask photos of SOBA_v2.

## Finetune Repository

[Segment-Anything-finetune](https://github.com/pg56714/Segment-Anything-finetune)

[EfficientViT-SAM-finetune](https://github.com/pg56714/EfficientViT-SAM-finetune)
