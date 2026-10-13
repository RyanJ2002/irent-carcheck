"""全域設定：拍攝面定義、品質門檻、分流門檻、模型路徑。
所有門檻都可以用環境變數覆寫，方便現場調整。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 讀取 .env（不覆蓋已存在的環境變數）
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _v.strip():
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "carcheck.db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _f(name, default):
    return float(os.getenv(name, default))


# 拍攝面：代碼、中文、iRent ImageType
PANELS = [
    ("LF", "左前", 1),
    ("RF", "右前", 2),
    ("LR", "左後", 3),
    ("RR", "右後", 4),
    ("INT_F", "前車內", 10),
    ("INT_R", "後車內", 11),
    ("OTHER", "其他", None),
]
PANEL_CODES = [p[0] for p in PANELS]
PANEL_ZH = {p[0]: p[1] for p in PANELS}
IMAGETYPE_TO_PANEL = {p[2]: p[0] for p in PANELS if p[2] is not None}
EXTERIOR = {"LF", "RF", "LR", "RR"}
INTERIOR = {"INT_F", "INT_R"}

CLEAN_LEVELS = ["clean", "normal", "dirty"]
CLEAN_ZH = {"clean": "乾淨", "normal": "普通", "dirty": "髒汙"}

# ---- 環節一：照片品質（規則，不用模型）----
QUALITY = {
    "min_short_side": _f("Q_MIN_SHORT_SIDE", 480),   # 解析度下限（短邊像素）
    "blur_min": _f("Q_BLUR_MIN", 60),                # 拉普拉斯變異數下限（長邊縮到 1024 後計算）
    "bright_min": _f("Q_BRIGHT_MIN", 45),            # 平均亮度下限
    "bright_max": _f("Q_BRIGHT_MAX", 220),           # 平均亮度上限
    "contrast_min": _f("Q_CONTRAST_MIN", 28),        # 灰階標準差下限：過低代表霧化／鏡頭髒污
    "glare_max": _f("Q_GLARE_MAX", 0.18),            # 過曝像素比例上限
    "car_area_min": _f("Q_CAR_AREA_MIN", 0.18),      # 車輛框佔畫面比例下限：低於 = 太遠
    "car_area_max": _f("Q_CAR_AREA_MAX", 0.93),      # 高於 = 太近
}

# ---- 信心分流（營運可在儀表板調整）----
ROUTING = {
    "alert_min": _f("R_ALERT_MIN", 0.80),   # 異常機率 >= 此值 → 預警
    "pass_max": _f("R_PASS_MAX", 0.20),     # 異常機率 <= 此值 → 自動放行；中間 → 人工審核
}

# ---- 自我監控門檻 ----
MONITOR = {
    "min_feedback": int(os.getenv("M_MIN_FEEDBACK", 10)),   # 回饋筆數不足時不判斷
    "level1_disagree": _f("M_L1_DISAGREE", 0.10),           # 不一致率超過 → 第一級：自動擴大人工審核區間
    "level2_disagree": _f("M_L2_DISAGREE", 0.20),           # 超過 → 第二級：通知工程師
    "level1_review_ratio": _f("M_L1_REVIEW", 0.45),         # 送人工比例過高也觸發第一級
    "widen_by": _f("M_WIDEN_BY", 0.07),                     # 第一級時審核區間往兩側各擴大多少
}

# ---- 模型 ----
YOLO_PATH = MODELS_DIR / "yolo11n.onnx"
PANEL_ONNX = MODELS_DIR / "panel_classifier.onnx"
CLEAN_ONNX = MODELS_DIR / "cleanliness_classifier.onnx"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_SAMPLES = int(os.getenv("GEMINI_SAMPLES", 3))     # 自我一致性：同一題問幾次
DEMO_MOCK = os.getenv("DEMO_MOCK", "0") == "1"           # 沒有模型時產生「模擬結果」（介面會明確標示）
