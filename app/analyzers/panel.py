"""拍攝面分類：這張照片拍的是左前／右前／左後／右後／前車內／後車內／其他。

後端優先順序：
  1. models/panel_classifier.onnx（用 iRent ImageType 清洗後 fine-tune，最終方案）
  2. Gemini（零樣本，同時讀車牌）
  3. DEMO_MOCK=1 時的模擬結果（介面會標示「模擬」）
"""
import hashlib

import numpy as np

from ..config import DEMO_MOCK, PANEL_CODES, PANEL_ONNX
from ..imageio import to_jpeg
from . import gemini

_onnx = None

PROMPT = """You are checking photos uploaded by customers of a car-sharing service in Taiwan when they pick up or return a rental car.
Classify which view this photo shows. Left/right always refer to the CAR's own left/right, as seen by a driver sitting in the car facing forward (Taiwan is left-hand drive: the driver's door is on the car's LEFT side).

- LF: exterior, front-left corner (front of car + the car's left side visible)
- RF: exterior, front-right corner
- LR: exterior, rear-left corner (rear of car + the car's left side)
- RR: exterior, rear-right corner
- INT_F: interior, front cabin (dashboard, steering wheel, front seats)
- INT_R: interior, rear cabin (rear seats, rear footwell)
- OTHER: anything else (not a car, odometer close-up only, trunk only, ground, blurred beyond recognition, etc.)

Tip: for an exterior corner shot, decide front vs rear first (headlights/grille vs taillights/trunk), then decide which side of the car is visible.
Also report whether the license plate is visible and readable, and its text if readable (Taiwan format, e.g. ABC-1234)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "panel": {"type": "string", "enum": PANEL_CODES},
        "plate_visible": {"type": "boolean"},
        "plate_text": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["panel", "plate_visible", "plate_text", "reason"],
}


def backend() -> str:
    if PANEL_ONNX.exists():
        return "onnx"
    if gemini.available():
        return "gemini"
    if DEMO_MOCK:
        return "mock"
    return "none"


def classify(img: np.ndarray, hint: str | None = None) -> dict:
    """回傳 {panel, probs, confidence, method, plate}；plate = None 代表未檢查。"""
    b = backend()
    if b == "onnx":
        global _onnx
        if _onnx is None:
            from .onnx_cls import OnnxClassifier
            _onnx = OnnxClassifier(PANEL_ONNX)
        probs = _onnx.predict(img)
        probs = {k: probs.get(k, 0.0) for k in PANEL_CODES}
        plate = None
    elif b == "gemini":
        answers = gemini.ask_json(PROMPT, [to_jpeg(img, 1024)], SCHEMA)
        probs = gemini.vote([a.get("panel") for a in answers], PANEL_CODES)
        texts = [a.get("plate_text", "").strip().upper() for a in answers if a.get("plate_visible")]
        plate = {"visible": len(texts) * 2 > len(answers),
                 "text": max(set(texts), key=texts.count) if texts else ""}
    elif b == "mock":
        code = hint if hint in PANEL_CODES else PANEL_CODES[int(hashlib.md5(img[::17, ::17].tobytes()).hexdigest(), 16) % 6]
        rng = np.random.default_rng(int(img[::23, ::23].sum()) % 9999)
        top = float(rng.uniform(0.55, 0.97))
        rest = (1 - top) / (len(PANEL_CODES) - 1)
        probs = {k: round(top if k == code else rest, 3) for k in PANEL_CODES}
        plate = {"visible": True, "text": "MOCK-0000"}
    else:
        return {"panel": None, "probs": {}, "confidence": 0.0, "method": "none", "plate": None}

    panel = max(probs, key=probs.get)
    return {"panel": panel, "probs": probs, "confidence": round(probs[panel], 3), "method": b, "plate": plate}
