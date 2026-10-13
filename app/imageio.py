"""影像讀取：修正手機 EXIF 方向，統一成 OpenCV BGR。"""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def decode(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"無法讀取影像：{e}") from e
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def to_jpeg(img: np.ndarray, long_side: int = 1280, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", resize_long(img, long_side), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG 編碼失敗")
    return buf.tobytes()
