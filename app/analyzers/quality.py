"""環節一：照片品質檢查（純規則，不需要模型，可移植到 App 端）。

每一項回傳 {key, label, ok, value, message}。"""
import cv2
import numpy as np

from ..config import QUALITY
from ..imageio import resize_long


def _check(key, label, ok, value, message):
    return {"key": key, "label": label, "ok": bool(ok), "value": round(float(value), 3), "message": message}


def check_quality(img: np.ndarray) -> list[dict]:
    q = QUALITY
    h, w = img.shape[:2]
    small = resize_long(img, 1024)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    out = []
    short = min(h, w)
    out.append(_check("resolution", "解析度", short >= q["min_short_side"], short,
                      "解析度足夠" if short >= q["min_short_side"] else f"解析度過低（短邊 {short}px），請用原相機拍攝"))

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    out.append(_check("blur", "清晰度", blur >= q["blur_min"], blur,
                      "影像清晰" if blur >= q["blur_min"] else "照片模糊，請拿穩手機、對焦後再拍"))

    mean = gray.mean()
    if mean < q["bright_min"]:
        msg, ok = "光線太暗，請開啟閃光燈或換到明亮處", False
    elif mean > q["bright_max"]:
        msg, ok = "畫面過亮，請避開直射光", False
    else:
        msg, ok = "亮度正常", True
    out.append(_check("brightness", "亮度", ok, mean, msg))

    glare = float((gray >= 250).mean())
    out.append(_check("glare", "反光", glare <= q["glare_max"], glare,
                      "無明顯反光" if glare <= q["glare_max"] else "反光面積過大，車身細節看不清楚，請換個角度"))

    # 鏡頭髒污 / 霧化：整體對比過低，且高頻細節少
    contrast = gray.std()
    ok = contrast >= q["contrast_min"]
    out.append(_check("haze", "鏡頭髒污", ok, contrast,
                      "畫面清透" if ok else "畫面霧化，可能是鏡頭髒污，請擦拭鏡頭後重拍"))
    return out
