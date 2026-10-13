#!/usr/bin/env python3
"""下載預訓練模型：YOLO11n（COCO，車輛偵測用，約 11MB）。"""
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "yolo11n.onnx": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
}
out = Path(__file__).resolve().parent.parent / "models"
out.mkdir(exist_ok=True)
for name, url in MODELS.items():
    dst = out / name
    if dst.exists():
        print(f"已存在 {dst}")
        continue
    print(f"下載 {name} …")
    try:
        urllib.request.urlretrieve(url, dst)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"下載失敗：{e}\n可手動下載 {url} 放到 {dst}")
    print(f"完成 {dst}（{dst.stat().st_size / 1e6:.1f} MB）")
