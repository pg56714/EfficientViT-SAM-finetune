# EfficientViT-SAM-finetune

## Getting Started

[uv](https://docs.astral.sh/uv/)

```bash
uv venv

.venv\Scripts\activate

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

uv pip install -r pyproject.toml
```

## Datasets Steps

1. Download from the [SOBA_v2-Datasets](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP) and place the files inside the `datasets` folder.

2. Run the `save_json.py` script in the `tool` folder to create `sam_train.json` and `sam_test.json` and place them inside the `datasets` folder.

3. Finish.

You can use the `save_labels.py` script in the `tool` folder to verify the labels.

Organized [SOBA_v2-Datasets](https://drive.google.com/drive/folders/1561wGAf0oik7C7__3byLHBNJOIadFuMw?usp=sharing) for use with the SAM model.

### Weights

Download the weights from the following links and save them in the `weights` directory.

[EfficientViT-SAM-XL1](https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt)

## Fine-tune

```bash
uv run effvitsam_finetune.py
```

## Test

```bash
uv run effvitsam_test.py
```

## Eval

```bash
uv run effvitsam_eval.py
```

## Demo

```bash
uv run demo_app.py
```

## Source

[efficientvit](https://github.com/mit-han-lab/efficientvit)

[Segment-Anything-finetune](https://github.com/pg56714/Segment-Anything-finetune)

[SOBA_v2-Datasets](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP)

[SSIS](https://github.com/stevewongv/SSIS)
