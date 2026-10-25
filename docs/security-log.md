# Security Log

English | [繁體中文](security-log.zh-TW.md)

- Date: 2026-10-25
- Branch: `fix/security-upgrades`
- Scope: `pyproject.toml`, `uv.lock`

## Summary

PyTorch moved to the CUDA 12.6 index, and gradio, onnx and onnxruntime-gpu were upgraded. The unused torchaudio was removed. pillow still has open advisories because gradio caps its version.

## Changes

| Item | Before | After | Reason |
|---|---|---|---|
| torch / torchvision | 2.5.1+cu121 / 0.20.1 | 2.14.1+cu126 / 0.29.1 | CVE-2025-32434 (`torch.load(weights_only=True)` remote code execution) and later memory-corruption advisories |
| torchaudio | 2.5.1 (from PyPI) | removed | Not imported anywhere; it came from PyPI instead of the CUDA index, and cu126 has no build matching torch 2.14 |
| gradio | 5.12.0 | 6.30.0 | File read, SSRF and path traversal advisories (fixed in 6.16) |
| onnx | 1.17.0 | 1.23.2 | Arbitrary file read through external data (symlink / hardlink traversal) and memory-safety advisories |
| onnxruntime-gpu | 1.20.1 | 1.31.0 | Refreshed with the lock |

## What you need to know

- PyTorch now comes from the CUDA 12.6 wheel index (`pytorch-cu126`) instead of cu121, because the cu121 index stops at torch 2.5.1. You need an NVIDIA driver that supports CUDA 12.6 or newer. Recreate the virtual environment so the new wheels are installed.

## Remaining issues

### Dependency versions held back upstream

| Package | Version | Advisories | Why it is not fixed | Impact and mitigation |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 (fixed in 13.0.0) | gradio >= 6.16 requires `pillow<13`; no gradio release allows Pillow 13 yet. | Crafted image files (for example TIFF or PDF) can crash the process or read out-of-bounds memory when decoded. Run the Gradio app locally only (no `share=True`, no public port) and open only images you trust. Upgrade once gradio allows Pillow 13. |

## How it was verified

- osv-scanner on `uv.lock`; only pillow remains.
- A checkpoint saved with `state_dict()` loads through `create_sam_model` (`torch.load(weights_only=True)`).
- `demo_app.py` builds its UI and runs `generate_mask_sam()` on CPU.

## Not verified

- Inference on a GPU with the real model weights (the test environment has no GPU and no weights).
- Fine-tuning, test and eval scripts on real datasets.
