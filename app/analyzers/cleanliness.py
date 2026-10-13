"""環節二-2：內裝整潔度三分類（髒汙／普通／乾淨），附髒污類型給清潔人員參考。

後端優先順序：models/cleanliness_classifier.onnx → Gemini → DEMO_MOCK。"""
import numpy as np

from ..config import CLEAN_LEVELS, CLEAN_ONNX, DEMO_MOCK
from ..imageio import to_jpeg
from . import gemini

_onnx = None

ISSUES = ["trash", "liquid_stain", "hair", "mud_sand", "food", "other"]
ISSUE_ZH = {"trash": "垃圾", "liquid_stain": "液體污漬", "hair": "毛髮", "mud_sand": "泥沙",
            "food": "食物殘渣", "other": "其他"}

PROMPT = """You are inspecting the interior of a shared rental car right after a customer returned it.
Rate the cleanliness for the next customer:
- clean: ready to rent, no visible trash or stains
- normal: minor issues (a little dust, small crumbs) that do not need immediate cleaning
- dirty: needs cleaning before the next rental (trash left behind, visible stains, mud, sand, pet hair, food waste, strong mess)
List the issue types you see and briefly describe where they are (in Traditional Chinese).
Ignore normal wear (faded seats, worn floor mats) and lighting conditions."""

SCHEMA = {
    "type": "object",
    "properties": {
        "level": {"type": "string", "enum": CLEAN_LEVELS},
        "issues": {"type": "array", "items": {"type": "string", "enum": ISSUES}},
        "description": {"type": "string"},
    },
    "required": ["level", "issues", "description"],
}


def backend() -> str:
    if CLEAN_ONNX.exists():
        return "onnx"
    if gemini.available():
        return "gemini"
    if DEMO_MOCK:
        return "mock"
    return "none"


def assess(img: np.ndarray) -> dict:
    b = backend()
    issues, desc = [], ""
    if b == "onnx":
        global _onnx
        if _onnx is None:
            from .onnx_cls import OnnxClassifier
            _onnx = OnnxClassifier(CLEAN_ONNX)
        probs = _onnx.predict(img)
    elif b == "gemini":
        answers = gemini.ask_json(PROMPT, [to_jpeg(img, 1024)], SCHEMA)
        probs = gemini.vote([a.get("level") for a in answers], CLEAN_LEVELS)
        counts = {}
        for a in answers:
            for i in set(a.get("issues", [])):
                counts[i] = counts.get(i, 0) + 1
        issues = [i for i, c in counts.items() if c * 2 > len(answers)]
        desc = answers[0].get("description", "")
    elif b == "mock":
        rng = np.random.default_rng(int(img[::19, ::19].sum()) % 9999)
        p = rng.dirichlet([2, 2, 1.5])
        probs = dict(zip(CLEAN_LEVELS, [round(float(v), 3) for v in p]))
        if probs["dirty"] > 0.4:
            issues, desc = ["trash"], "（模擬）後座地墊有垃圾"
    else:
        return {"level": None, "probs": {}, "p_dirty": None, "confidence": 0.0, "issues": [],
                "description": "", "method": "none"}

    level = max(probs, key=probs.get)
    return {"level": level, "probs": probs, "p_dirty": probs.get("dirty", 0.0),
            "confidence": round(probs[level], 3), "issues": issues, "description": desc, "method": b}
