"""車輛偵測（YOLO11n，COCO 預訓練，ONNX 推論）：判斷車有沒有入鏡、太遠或太近。

模型檔：models/yolo11n.onnx（scripts/fetch_models.py 會下載）。"""
import cv2
import numpy as np

from ..config import QUALITY, YOLO_PATH

VEHICLE = {2: "car", 5: "bus", 7: "truck"}
_session = None


def available() -> bool:
    return YOLO_PATH.exists()


def _sess():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(str(YOLO_PATH), providers=["CPUExecutionProvider"])
    return _session


def detect_vehicle(img: np.ndarray, conf_th: float = 0.30):
    """回傳畫面中最大的一台車：{box:[x1,y1,x2,y2] 0~1, score, area_ratio, edges}；沒有車回傳 None。"""
    h, w = img.shape[:2]
    size = 640
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    out = _sess().run(None, {"images": x})[0][0].T  # (anchors, 84)
    cls_scores = out[:, 4:]
    cls = cls_scores.argmax(1)
    score = cls_scores.max(1)
    keep = np.isin(cls, list(VEHICLE)) & (score >= conf_th)
    if not keep.any():
        return None
    b, sc = out[keep, :4], score[keep]
    x1, y1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    rects = np.stack([x1, y1, b[:, 2], b[:, 3]], 1)
    idx = cv2.dnn.NMSBoxes(rects.tolist(), sc.tolist(), conf_th, 0.5)
    idx = np.array(idx).reshape(-1)
    best = max(idx, key=lambda i: rects[i, 2] * rects[i, 3])
    bx, by, bw, bh = rects[best] / s
    box = [max(0, bx) / w, max(0, by) / h, min(w, bx + bw) / w, min(h, by + bh) / h]
    area = (box[2] - box[0]) * (box[3] - box[1])
    m = 0.01
    edges = sum([box[0] <= m, box[1] <= m, box[2] >= 1 - m, box[3] >= 1 - m])
    return {"box": [round(float(v), 4) for v in box], "score": round(float(sc[best]), 3),
            "area_ratio": round(float(area), 3), "edges": int(edges)}


def distance_check(det) -> dict:
    """依車輛框判斷遠近（外觀照用）。"""
    q = QUALITY
    if det is None:
        return {"key": "distance", "label": "車輛入鏡", "ok": False, "value": 0.0,
                "message": "畫面中找不到車輛，請對準車身拍攝"}
    a = det["area_ratio"]
    if a < q["car_area_min"]:
        return {"key": "distance", "label": "拍攝距離", "ok": False, "value": a,
                "message": "距離太遠，請靠近一點，讓車身佔滿大部分畫面"}
    if a > q["car_area_max"] or det["edges"] >= 3:
        return {"key": "distance", "label": "拍攝距離", "ok": False, "value": a,
                "message": "距離太近，車身被裁切，請退後一步"}
    return {"key": "distance", "label": "拍攝距離", "ok": True, "value": a, "message": "距離適中"}
