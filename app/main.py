"""車況管家 Demo：FastAPI 後端 + 靜態前端。

啟動：uvicorn app.main:app --reload --port 8000
"""
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store
from .analyzers import cleanliness, damage, detector, gemini, panel, quality
from .config import (CLEAN_ZH, DEMO_MOCK, EXTERIOR, IMAGETYPE_TO_PANEL, INTERIOR, PANEL_CODES, PANEL_ZH,
                     UPLOAD_DIR)
from .imageio import decode, to_jpeg

app = FastAPI(title="車況管家 Demo")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
store.init()
POOL = ThreadPoolExecutor(6)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {
        "panel": panel.backend(),
        "cleanliness": cleanliness.backend(),
        "damage": damage.backend(),
        "vehicle_detector": detector.available(),
        "gemini": gemini.available(),
        "mock": DEMO_MOCK,
        "panels": [{"code": c, "zh": PANEL_ZH[c]} for c in PANEL_CODES],
    }


async def _read(f: UploadFile):
    data = await f.read()
    try:
        return decode(data)
    except ValueError as e:
        raise HTTPException(400, f"{f.filename}: {e}") from e


def _plate_norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


# ======================= 環節一：拍照當下的即時防呆 =======================
def run_check(img, target: str | None, car_no: str = "") -> dict:
    checks = quality.check_quality(img)
    vehicle = None
    if target in EXTERIOR or target is None:
        if detector.available():
            vehicle = detector.detect_vehicle(img)
            if target in EXTERIOR:
                checks.append(detector.distance_check(vehicle))
        elif target in EXTERIOR:
            checks.append({"key": "distance", "label": "拍攝距離", "ok": None, "value": 0,
                           "message": "未檢查（缺少車輛偵測模型，執行 scripts/fetch_models.py）"})

    try:
        pr = panel.classify(img)
    except Exception as e:  # noqa: BLE001
        pr = {"panel": None, "probs": {}, "confidence": 0, "method": "error", "plate": None, "error": str(e)}

    if target:
        if pr["panel"] is None:
            checks.append({"key": "panel", "label": "拍攝位置", "ok": None, "value": 0,
                           "message": "未檢查（尚未啟用拍攝面分類，見 README）"})
        elif pr["panel"] == target:
            low = pr["confidence"] < 0.5
            checks.append({"key": "panel", "label": "拍攝位置", "ok": True, "value": pr["confidence"],
                           "message": f"已確認為{PANEL_ZH[target]}" + ("（信心偏低，已標記）" if low else "")})
        else:
            checks.append({"key": "panel", "label": "拍攝位置", "ok": False, "value": pr["confidence"],
                           "message": f"這張看起來是「{PANEL_ZH[pr['panel']]}」，請改拍「{PANEL_ZH[target]}」"})

    if target in EXTERIOR:
        plate = pr.get("plate")
        if plate is None:
            checks.append({"key": "plate", "label": "車牌", "ok": None, "value": 0,
                           "message": "未檢查（車牌辨識需要 Gemini）"})
        elif not plate["visible"]:
            checks.append({"key": "plate", "label": "車牌", "ok": False, "value": 0,
                           "message": "看不到車牌，請讓車牌入鏡"})
        elif car_no and _plate_norm(plate["text"]) and _plate_norm(plate["text"]) != _plate_norm(car_no):
            checks.append({"key": "plate", "label": "車牌", "ok": False, "value": 0,
                           "message": f"車牌 {plate['text']} 與訂單車號 {car_no} 不符，請確認是否拍錯車"})
        else:
            checks.append({"key": "plate", "label": "車牌", "ok": True, "value": 1,
                           "message": f"車牌入鏡 {plate['text']}".strip()})

    failed = [c for c in checks if c["ok"] is False]
    return {"verdict": "retake" if failed else "pass", "checks": checks, "panel": pr, "vehicle": vehicle}


@app.post("/api/check")
async def api_check(file: UploadFile = File(...), target: str = Form(""), car_no: str = Form("")):
    if target and target not in PANEL_CODES:
        raise HTTPException(400, "target 必須是 " + ",".join(PANEL_CODES))
    img = await _read(file)
    return run_check(img, target or None, car_no)


# ======================= 照片自動分類（批次） =======================
FNAME_CODE = re.compile(r"(?:^|[_\-\s])(LF|RF|LR|RR|INT_F|INT_R|OTHER)(?:[_\-\s.]|$)")
FNAME_TYPE = re.compile(r"_(\d{1,2})_\d+\.\w+$")


def original_label(name: str):
    """從檔名猜原本的上傳分類（支援 tools/download_irent.py 的命名）。"""
    m = FNAME_CODE.search(name)
    if m:
        return m.group(1)
    m = FNAME_TYPE.search(name)
    if m:
        return IMAGETYPE_TO_PANEL.get(int(m.group(1)), "OTHER")
    return None


@app.post("/api/classify")
async def api_classify(files: list[UploadFile] = File(...)):
    imgs = [(f.filename, await _read(f)) for f in files]

    def work(item):
        name, img = item
        try:
            pr = panel.classify(img)
        except Exception as e:  # noqa: BLE001
            pr = {"panel": None, "probs": {}, "confidence": 0, "method": "error", "error": str(e)}
        q = quality.check_quality(img)
        orig = original_label(name)
        return {"filename": name, "panel": pr, "original": orig,
                "mismatch": bool(orig and pr["panel"] and orig != pr["panel"]),
                "quality_issues": [c["label"] for c in q if not c["ok"]]}

    return {"results": list(POOL.map(work, imgs)), "method": panel.backend()}


class PanelFeedback(BaseModel):
    ai_panel: str
    human_panel: str
    confidence: float = 0


@app.post("/api/feedback/panel")
def api_panel_feedback(fb: PanelFeedback):
    store.add_feedback("panel", "ops", None, None, fb.ai_panel, fb.human_panel, fb.confidence)
    return {"ok": True}


# ======================= 環節二＋三：還車判讀與分流 =======================
def route_of(p_abn, thresholds, uncertain=False):
    if uncertain or p_abn is None:
        return "review"
    if p_abn >= thresholds["alert_min"]:
        return "alert"
    if p_abn <= thresholds["pass_max"]:
        return "pass"
    return "review"


def save_img(cid, key, img):
    d = UPLOAD_DIR / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.jpg").write_bytes(to_jpeg(img, 1600))
    return f"/uploads/{cid}/{key}.jpg"


@app.post("/api/inspect")
async def api_inspect(request: Request):
    form = await request.form()
    order_no = str(form.get("order_no") or f"DEMO-{uuid.uuid4().hex[:6].upper()}")
    car_no = str(form.get("car_no") or "")
    cid = uuid.uuid4().hex[:10]
    th = store.effective_thresholds()

    imgs = {}
    for key, val in form.multi_items():
        if hasattr(val, "read") and getattr(val, "filename", ""):
            imgs[key] = await _read(val)
    if not imgs:
        raise HTTPException(400, "請至少上傳一張還車照")
    urls = {k: save_img(cid, k, v) for k, v in imgs.items()}

    def exterior(code):
        p, r = imgs.get(f"pickup_{code}"), imgs.get(f"return_{code}")
        base = {"key": code, "kind": "damage", "panel": code, "panel_zh": PANEL_ZH[code],
                "pickup_url": urls.get(f"pickup_{code}"), "return_url": urls.get(f"return_{code}")}
        if r is None:
            return {**base, "route": "review", "p_abnormal": None, "confidence": 0, "result": None,
                    "note": "缺少還車照，無法判定"}
        q_fail = [c["label"] for c in quality.check_quality(r) if not c["ok"]]
        if p is None:
            return {**base, "route": "review", "p_abnormal": None, "confidence": 0, "result": None,
                    "quality_issues": q_fail, "note": "缺少取車照，無法比對新舊車損"}
        try:
            res = damage.compare(p, r)
        except Exception as e:  # noqa: BLE001
            return {**base, "route": "review", "p_abnormal": None, "confidence": 0, "result": None,
                    "note": f"判讀失敗：{e}"}
        uncertain = not res["comparable"]
        note = "兩張照片無法對應比較" if uncertain else ""
        if q_fail:
            note = (note + "；" if note else "") + "還車照品質不佳：" + "、".join(q_fail)
        return {**base, "route": route_of(res["p_new"], th, uncertain), "p_abnormal": res["p_new"],
                "confidence": res["confidence"], "result": res, "quality_issues": q_fail, "note": note}

    def interior(code):
        r = imgs.get(f"return_{code}")
        base = {"key": code, "kind": "cleanliness", "panel": code, "panel_zh": PANEL_ZH[code],
                "pickup_url": urls.get(f"pickup_{code}"), "return_url": urls.get(f"return_{code}")}
        if r is None:
            return {**base, "route": "review", "p_abnormal": None, "confidence": 0, "result": None,
                    "note": "缺少還車內裝照"}
        try:
            res = cleanliness.assess(r)
        except Exception as e:  # noqa: BLE001
            return {**base, "route": "review", "p_abnormal": None, "confidence": 0, "result": None,
                    "note": f"判讀失敗：{e}"}
        if res["level"] is None:
            return {**base, "route": "review", "p_abnormal": None, "confidence": 0, "result": res,
                    "note": "未啟用整潔度判讀（需要 Gemini 或 fine-tune 模型）"}
        return {**base, "route": route_of(res["p_dirty"], th), "p_abnormal": res["p_dirty"],
                "confidence": res["confidence"], "result": res, "note": ""}

    jobs = [POOL.submit(exterior, c) for c in ("LF", "RF", "LR", "RR")
            if f"pickup_{c}" in imgs or f"return_{c}" in imgs]
    jobs += [POOL.submit(interior, c) for c in ("INT_F", "INT_R") if f"return_{c}" in imgs]
    items = [j.result() for j in jobs]

    order = {"pass": 0, "review": 1, "alert": 2}
    overall = max((it["route"] for it in items), key=order.get)
    actions = []
    if any(it["route"] == "alert" and it["kind"] == "damage" for it in items):
        actions += ["產生維修評估工單", "暫停此車下一筆預約", "通知下一位預約用戶並推薦換車"]
    if any(it["route"] == "alert" and it["kind"] == "cleanliness" for it in items):
        actions += ["產生清潔工單"]
        if "暫停此車下一筆預約" not in actions:
            actions += ["清潔完成前暫停出租"]
    if overall == "review":
        actions += ["送入人工審核佇列"]
    if overall != "pass":
        actions += ["客服可調閱前後對照與 AI 判讀"]

    case = {"id": cid, "created": store.now(), "order_no": order_no, "car_no": car_no, "route": overall,
            "items": items, "actions": actions, "thresholds": th,
            "methods": {"damage": damage.backend(), "cleanliness": cleanliness.backend()}}
    store.save_case(case)
    return case


@app.get("/api/cases")
def api_cases(route: str | None = None):
    return store.list_cases(route)


@app.get("/api/cases/{cid}")
def api_case(cid: str):
    c = store.get_case(cid)
    if not c:
        raise HTTPException(404, "找不到案件")
    return c


class Review(BaseModel):
    item_key: str
    human_abnormal: bool | None = None   # 審核人員：是否真的異常（新車損／髒汙）
    human_level: str | None = None       # 下一位用戶評價：clean / normal / dirty
    source: str = "ops"                  # ops = 審核人員；next_renter = 下一位用戶


@app.post("/api/cases/{cid}/review")
def api_review(cid: str, rv: Review):
    case = store.get_case(cid)
    if not case:
        raise HTTPException(404, "找不到案件")
    it = next((i for i in case["items"] if i["key"] == rv.item_key), None)
    if not it:
        raise HTTPException(404, "找不到項目")

    if rv.human_level:  # 整潔度三分類回饋
        ai = (it.get("result") or {}).get("level")
        store.add_feedback("cleanliness", rv.source, cid, rv.item_key, ai, rv.human_level, it.get("confidence"))
        it.setdefault("feedback", []).append({"source": rv.source, "label": CLEAN_ZH.get(rv.human_level),
                                              "agree": ai == rv.human_level, "at": store.now()})
    elif rv.human_abnormal is not None:
        ai = None if it["p_abnormal"] is None else it["p_abnormal"] >= 0.5
        task = "damage" if it["kind"] == "damage" else "cleanliness"
        if ai is not None:
            if task == "cleanliness":  # 轉成三分類的二元版本比較
                ai_l, hu_l = ("dirty" if ai else "not_dirty"), ("dirty" if rv.human_abnormal else "not_dirty")
            else:
                ai_l, hu_l = ai, rv.human_abnormal
            store.add_feedback(task, rv.source, cid, rv.item_key, ai_l, hu_l, it.get("confidence"))
        it.setdefault("feedback", []).append({"source": rv.source, "label": "異常" if rv.human_abnormal else "正常",
                                              "agree": None if ai is None else ai == rv.human_abnormal,
                                              "at": store.now()})
        it["resolved"] = True
    else:
        raise HTTPException(400, "需提供 human_abnormal 或 human_level")
    store.save_case(case)
    return case


@app.get("/api/metrics")
def api_metrics():
    return store.metrics()


class Thresholds(BaseModel):
    alert_min: float
    pass_max: float


@app.put("/api/thresholds")
def api_thresholds(t: Thresholds):
    try:
        store.set_thresholds(t.alert_min, t.pass_max)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return store.metrics()


class Simulate(BaseModel):
    task: str = "damage"
    n: int = 20
    disagree_rate: float = 0.25


@app.post("/api/simulate-feedback")
def api_simulate(s: Simulate):
    """Demo 用：一次灌入 n 筆回饋，展示監控指標如何觸發分級升級。"""
    import random
    for _ in range(max(1, min(s.n, 200))):
        ai = random.random() < 0.5
        human = (not ai) if random.random() < s.disagree_rate else ai
        store.add_feedback(s.task, "simulated", None, None, ai, human, round(random.uniform(0.5, 1), 2))
    return store.metrics()


@app.post("/api/reset")
def api_reset():
    store.reset_all()
    return {"ok": True}


# 讓 cv2 在多執行緒下不要再自己開一堆執行緒
cv2.setNumThreads(1)
