"""通用 ONNX 影像分類器（由 training/train_classifier.py 匯出）。

每個模型檔旁邊要有同名 .json：{"labels": [...], "input_size": 224, "mean": [...], "std": [...], "temperature": 1.0}
temperature 是訓練後用驗證集做 temperature scaling 得到的校準參數。"""
import json
from pathlib import Path

import cv2
import numpy as np


class OnnxClassifier:
    def __init__(self, path: Path):
        import onnxruntime as ort
        self.meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.labels = self.meta["labels"]
        self.size = int(self.meta.get("input_size", 224))
        self.mean = np.array(self.meta.get("mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.array(self.meta.get("std", [0.229, 0.224, 0.225]), np.float32)
        self.T = float(self.meta.get("temperature", 1.0))
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def predict(self, img_bgr: np.ndarray) -> dict:
        h, w = img_bgr.shape[:2]
        s = self.size / min(h, w)  # 短邊縮放後中心裁切（與訓練時一致）
        r = cv2.resize(img_bgr, (max(self.size, round(w * s)), max(self.size, round(h * s))), interpolation=cv2.INTER_AREA)
        y0, x0 = (r.shape[0] - self.size) // 2, (r.shape[1] - self.size) // 2
        r = r[y0:y0 + self.size, x0:x0 + self.size]
        x = (cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.mean) / self.std
        logits = self.sess.run(None, {self.input: x.transpose(2, 0, 1)[None]})[0][0] / self.T
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return {lab: round(float(v), 4) for lab, v in zip(self.labels, p)}
