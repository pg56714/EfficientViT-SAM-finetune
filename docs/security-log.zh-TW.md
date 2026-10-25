# 安全性紀錄

[English](security-log.md) | 繁體中文

- 日期: 2026-10-25
- 分支: `fix/security-upgrades`
- 範圍: `pyproject.toml`、`uv.lock`

## 摘要

PyTorch 改用 CUDA 12.6 的套件來源，gradio、onnx、onnxruntime-gpu 升級，並移除沒有使用的 torchaudio。pillow 因為 gradio 限制版本，仍有未修正的漏洞。

## 修改內容

| 項目 | 修改前 | 修改後 | 原因 |
|---|---|---|---|
| torch / torchvision | 2.5.1+cu121 / 0.20.1 | 2.14.1+cu126 / 0.29.1 | CVE-2025-32434（`torch.load(weights_only=True)` 可遠端執行程式碼）與之後的記憶體破壞漏洞 |
| torchaudio | 2.5.1 (from PyPI) | removed | 沒有任何程式使用；原本從 PyPI 而非 CUDA 來源安裝，且 cu126 沒有對應 torch 2.14 的版本 |
| gradio | 5.12.0 | 6.30.0 | 任意檔案讀取、SSRF、路徑穿越漏洞（6.16 修正） |
| onnx | 1.17.0 | 1.23.2 | 可透過 external data（符號連結、硬連結）任意讀檔，以及記憶體安全漏洞 |
| onnxruntime-gpu | 1.20.1 | 1.31.0 | 隨 lock 更新 |

## 使用前須知

- PyTorch 改由 CUDA 12.6 的套件來源（`pytorch-cu126`）安裝，不再使用 cu121，因為 cu121 最高只到 torch 2.5.1。需要支援 CUDA 12.6 以上的 NVIDIA 驅動；請重建虛擬環境以安裝新版套件。

## 尚未解決的問題

### 被上游限制的套件版本

| 套件 | 版本 | 漏洞 | 無法修正的原因 | 影響與因應 |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 筆（13.0.0 修正） | gradio 6.16 以上要求 `pillow<13`，目前沒有允許 Pillow 13 的 gradio 版本。 | 解碼惡意製作的圖片（例如 TIFF、PDF）可能讓程式當掉或越界讀取記憶體。請只在本機執行 Gradio 介面（不要開 `share=True`、不要對外開放連接埠），只開啟可信任的圖片；等 gradio 允許 Pillow 13 後再升級。 |

## 驗證方式

- 以 osv-scanner 掃描 `uv.lock`，只剩 pillow。
- 以 `state_dict()` 儲存的權重檔能透過 `create_sam_model`（`torch.load(weights_only=True)`）載入。
- `demo_app.py` 能建立介面，並在 CPU 上執行 `generate_mask_sam()`。

## 未驗證項目

- 用 GPU 與實際權重執行推論（測試環境沒有 GPU 與權重檔）。
- 以實際資料集執行微調、測試與評估腳本。
