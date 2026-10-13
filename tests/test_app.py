"""離線測試：不需要 Gemini、不需要網路（只用合成影像）。執行：pytest -q"""
import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp()
os.environ["DEMO_MOCK"] = "1"
for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
    os.environ.pop(k, None)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.analyzers import damage, quality  # noqa: E402
from app.main import app, original_label, route_of  # noqa: E402

client = TestClient(app)


def scene(seed=0, w=1200, h=900):
    """有紋理的合成場景（足夠讓 ORB 對齊）。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 150, np.uint8)
    for _ in range(250):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 90)), y + int(rng.integers(10, 90))), c, -1)
    return img


def jpg(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_quality_blur_and_dark():
    img = scene()
    assert all(c["ok"] for c in quality.check_quality(img))
    keys = {c["key"]: c["ok"] for c in quality.check_quality(cv2.GaussianBlur(img, (51, 51), 0))}
    assert keys["blur"] is False
    keys = {c["key"]: c["ok"] for c in quality.check_quality((img * 0.15).astype(np.uint8))}
    assert keys["brightness"] is False


def test_damage_baseline_detects_change():
    a = scene(1)
    h, w = a.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                    np.float32([[12, 8], [w - 6, 15], [w - 15, h - 6], [6, h - 12]]))
    b = cv2.warpPerspective(a, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    same = damage._baseline(a, b)
    assert same["comparable"] and same["p_new"] < 0.2
    c = b.copy()
    cv2.circle(c, (600, 450), 60, (10, 10, 10), -1)
    hit = damage._baseline(a, c)
    assert hit["p_new"] > same["p_new"] and len(hit["items"]) >= 1
    assert hit["p_new"] <= 0.75  # 基準線永遠不會自動預警


def test_routing():
    th = {"alert_min": 0.8, "pass_max": 0.2}
    assert route_of(0.1, th) == "pass"
    assert route_of(0.5, th) == "review"
    assert route_of(0.9, th) == "alert"
    assert route_of(0.9, th, uncertain=True) == "review"
    assert route_of(None, th) == "review"


def test_original_label_from_filename():
    assert original_label("20260501-100000_LF_1_0.jpg") == "LF"
    assert original_label("abc_11_0.jpg") == "INT_R"
    assert original_label("IMG_1234.jpg") is None


def test_check_and_inspect_flow():
    r = client.post("/api/check", files={"file": ("a.jpg", jpg(scene()), "image/jpeg")}, data={"target": "LF"})
    assert r.status_code == 200 and r.json()["verdict"] in ("pass", "retake")

    a = scene(3)
    files = [("pickup_LF", ("p.jpg", jpg(a), "image/jpeg")), ("return_LF", ("r.jpg", jpg(a), "image/jpeg")),
             ("return_INT_F", ("i.jpg", jpg(scene(4)), "image/jpeg"))]
    case = client.post("/api/inspect", data={"car_no": "ABC-1234"}, files=files).json()
    lf = next(i for i in case["items"] if i["key"] == "LF")
    assert lf["route"] == "pass"  # 同一張照片 → 沒有新車損

    r = client.post(f"/api/cases/{case['id']}/review", json={"item_key": "LF", "human_abnormal": False})
    assert r.status_code == 200
    m = client.get("/api/metrics").json()
    assert m["tasks"]["damage"]["n"] == 1


def test_monitor_escalation():
    client.post("/api/reset")
    m = client.post("/api/simulate-feedback", json={"task": "damage", "n": 40, "disagree_rate": 0.5}).json()
    assert m["level"] == 2
    assert m["thresholds_effective"]["alert_min"] > m["thresholds_base"]["alert_min"]


def test_gemini_aggregation(monkeypatch):
    """不連網：用假回答檢查 Gemini 多次回答的彙整邏輯。"""
    from app.analyzers import cleanliness, damage as dmg, gemini, panel

    monkeypatch.setattr(gemini, "available", lambda: True)
    monkeypatch.setattr(panel, "PANEL_ONNX", panel.PANEL_ONNX.with_name("missing.onnx"))
    monkeypatch.setattr(cleanliness, "CLEAN_ONNX", cleanliness.CLEAN_ONNX.with_name("missing.onnx"))
    img = scene(5)

    monkeypatch.setattr(gemini, "ask_json", lambda *a, **k: [
        {"panel": "LF", "plate_visible": True, "plate_text": "abc-1234", "reason": ""},
        {"panel": "LF", "plate_visible": True, "plate_text": "ABC-1234", "reason": ""},
        {"panel": "RF", "plate_visible": False, "plate_text": "", "reason": ""}])
    r = panel.classify(img)
    assert r["panel"] == "LF" and abs(r["confidence"] - 0.667) < 0.01 and r["plate"]["visible"]

    monkeypatch.setattr(gemini, "ask_json", lambda *a, **k: [
        {"level": "dirty", "issues": ["trash"], "description": "後座有垃圾"},
        {"level": "dirty", "issues": ["trash", "hair"], "description": ""},
        {"level": "normal", "issues": [], "description": ""}])
    r = cleanliness.assess(img)
    assert r["level"] == "dirty" and r["issues"] == ["trash"] and abs(r["p_dirty"] - 0.667) < 0.01

    monkeypatch.setattr(gemini, "ask_json", lambda *a, **k: [
        {"comparable": True, "new_damage": True, "reason": "右後保桿刮痕",
         "items": [{"type": "scratch", "severity": "minor", "location": "右後保桿", "box_2d": [500, 100, 600, 300]}]},
        {"comparable": True, "new_damage": True, "reason": "", "items": []},
        {"comparable": True, "new_damage": False, "reason": "", "items": []}])
    r = dmg.compare(img, img)
    assert r["method"] == "gemini" and abs(r["p_new"] - 0.667) < 0.01
    assert r["items"][0]["box"] == [0.1, 0.5, 0.3, 0.6] and r["items"][0]["type_zh"] == "刮痕"
