"""環節二-1：新車損判斷——同一拍攝面的「取車照 vs 還車照」成對比較。

後端：
  - Gemini 成對比較（主要方案），問 N 次，新車損機率 = 回答「有新車損」的比例
  - 影像差異基準線（不用 AI）：ORB 特徵對齊 + 差異區域，只當示意與後備，
    機率被限制在 0.05~0.75，所以永遠不會自動預警，有差異就送人工審核
"""
import cv2
import numpy as np

from ..imageio import resize_long, to_jpeg
from . import detector, gemini

TYPE_ZH = {"scratch": "刮痕", "dent": "凹陷", "crack": "裂痕", "paint_chip": "掉漆",
           "broken_part": "零件破損", "other": "其他"}
SEV_ZH = {"minor": "輕微", "moderate": "中等", "severe": "嚴重"}

PROMPT = """You are a damage inspector for a car-sharing service.
Image 1 = PICKUP photo (taken when the customer picked up the car).
Image 2 = RETURN photo (same car, same side, taken when the customer returned it).
Find body damage that is visible in the RETURN photo but NOT in the PICKUP photo (i.e. new damage caused during this rental):
scratches, dents, cracks, paint chips, broken or missing parts (mirror, bumper, lights).

Rules:
- Damage already visible in the pickup photo is NOT new.
- Ignore differences caused by lighting, reflections, shadows, camera angle, distance, dirt, dust, water drops, or the background.
- If the two photos do not show the same side of the car with enough overlap to compare, set comparable=false.
- Boxes are on the RETURN photo, format [ymin, xmin, ymax, xmax] normalised to 0-1000.
- Write location in Traditional Chinese (e.g. 右後保桿下緣)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "comparable": {"type": "boolean"},
        "new_damage": {"type": "boolean"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(TYPE_ZH)},
                    "severity": {"type": "string", "enum": list(SEV_ZH)},
                    "location": {"type": "string"},
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["type", "severity", "location", "box_2d"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["comparable", "new_damage", "items", "reason"],
}


def backend() -> str:
    return "gemini" if gemini.available() else "baseline"


def compare(pickup: np.ndarray, ret: np.ndarray) -> dict:
    if backend() == "gemini":
        return _gemini(pickup, ret)
    return _baseline(pickup, ret)


def _gemini(pickup, ret):
    answers = gemini.ask_json(PROMPT, [to_jpeg(pickup, 1280), to_jpeg(ret, 1280)], SCHEMA)
    n = len(answers)
    comparable = sum(bool(a.get("comparable")) for a in answers) * 2 > n
    p_new = round(sum(bool(a.get("new_damage")) for a in answers) / n, 3)
    pos = [a for a in answers if a.get("new_damage")]
    items = []
    if pos:
        for it in max(pos, key=lambda a: len(a.get("items", []))).get("items", []):
            b = it.get("box_2d") or []
            box = [b[1] / 1000, b[0] / 1000, b[3] / 1000, b[2] / 1000] if len(b) == 4 else None
            items.append({"type": it.get("type"), "type_zh": TYPE_ZH.get(it.get("type"), "其他"),
                          "severity": it.get("severity"), "severity_zh": SEV_ZH.get(it.get("severity"), ""),
                          "location": it.get("location", ""), "box": box})
    reason = (pos[0] if pos else answers[0]).get("reason", "")
    return {"comparable": comparable, "p_new": p_new, "confidence": round(max(p_new, 1 - p_new), 3),
            "items": items, "reason": reason, "method": "gemini", "samples": n}


def _baseline(pickup, ret):
    a = resize_long(pickup, 1024)
    b = resize_long(ret, 1024)
    ga, gb = (cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)) for x in (a, b))

    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    fail = {"comparable": False, "p_new": 0.5, "confidence": 0.5, "items": [], "method": "baseline",
            "reason": "兩張照片的角度差異太大，無法對齊比較"}
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return fail
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, *n in matches if n and m.distance < 0.75 * n[0].distance]
    if len(good) < 25:
        return fail
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or inl.sum() < 15:
        return fail

    h, w = gb.shape
    warped = cv2.warpPerspective(ga, H, (w, h))
    valid = cv2.erode(cv2.warpPerspective(np.full_like(ga, 255), H, (w, h)), np.ones((15, 15), np.uint8)) > 0
    if valid.mean() < 0.35:
        return fail

    # 亮度校正後取差異
    wa, wb = warped[valid].astype(np.float32), gb[valid].astype(np.float32)
    adj = ((warped.astype(np.float32) - wa.mean()) / (wa.std() + 1e-6)) * wb.std() + wb.mean()
    diff = cv2.absdiff(cv2.GaussianBlur(adj, (7, 7), 0), cv2.GaussianBlur(gb.astype(np.float32), (7, 7), 0))
    mask = ((diff > 40) & valid).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    roi = np.ones_like(valid)
    if detector.available():
        det = detector.detect_vehicle(b)
        if det:
            x1, y1, x2, y2 = det["box"]
            roi = np.zeros_like(valid)
            roi[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = True
    mask[~roi] = 0

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.0015 * h * w
    cnts = sorted([c for c in cnts if cv2.contourArea(c) >= min_area], key=cv2.contourArea, reverse=True)[:5]
    items = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        items.append({"type": "other", "type_zh": "差異區域", "severity": None, "severity_zh": "",
                      "location": "", "box": [x / w, y / h, (x + bw) / w, (y + bh) / h]})
    frac = sum(cv2.contourArea(c) for c in cnts) / max(1, roi.sum())
    p = float(np.clip(1 / (1 + np.exp(-(frac - 0.01) / 0.004)), 0.05, 0.75))
    return {"comparable": True, "p_new": round(p, 3), "confidence": round(max(p, 1 - p), 3), "items": items,
            "method": "baseline", "inliers": int(inl.sum()),
            "reason": f"影像差異基準線：找到 {len(items)} 個差異區域（非 AI 判斷，僅供示意）"}
