"""SQLite 儲存：案件、回饋、設定；並計算自我監控指標。"""
import json
import sqlite3
import threading
from datetime import datetime

from .config import DB_PATH, MONITOR, ROUTING

_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY, created TEXT, order_no TEXT, car_no TEXT,
                                         route TEXT, payload TEXT);
        CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, task TEXT,
                                            source TEXT, case_id TEXT, item_key TEXT, ai_label TEXT,
                                            human_label TEXT, agree INTEGER, ai_conf REAL);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
        """)


def now():
    return datetime.now().isoformat(timespec="seconds")


# ---------- 案件 ----------
def save_case(case: dict):
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO cases VALUES(?,?,?,?,?,?)",
                  (case["id"], case["created"], case["order_no"], case["car_no"], case["route"],
                   json.dumps(case, ensure_ascii=False)))


def get_case(cid: str):
    with _conn() as c:
        r = c.execute("SELECT payload FROM cases WHERE id=?", (cid,)).fetchone()
    return json.loads(r["payload"]) if r else None


def list_cases(route: str | None = None, limit: int = 200):
    q, args = "SELECT payload FROM cases", []
    if route:
        q, args = q + " WHERE route=?", [route]
    with _conn() as c:
        rows = c.execute(q + " ORDER BY created DESC LIMIT ?", (*args, limit)).fetchall()
    return [json.loads(r["payload"]) for r in rows]


# ---------- 回饋 ----------
def add_feedback(task, source, case_id, item_key, ai_label, human_label, ai_conf):
    with _lock, _conn() as c:
        c.execute("INSERT INTO feedback(created,task,source,case_id,item_key,ai_label,human_label,agree,ai_conf)"
                  " VALUES(?,?,?,?,?,?,?,?,?)",
                  (now(), task, source, case_id, item_key, str(ai_label), str(human_label),
                   int(str(ai_label) == str(human_label)), ai_conf))


def reset_all():
    with _lock, _conn() as c:
        c.executescript("DELETE FROM cases; DELETE FROM feedback; DELETE FROM settings;")


# ---------- 門檻 ----------
def base_thresholds() -> dict:
    with _conn() as c:
        rows = {r["key"]: float(r["value"]) for r in c.execute("SELECT * FROM settings")}
    return {"alert_min": rows.get("alert_min", ROUTING["alert_min"]),
            "pass_max": rows.get("pass_max", ROUTING["pass_max"])}


def set_thresholds(alert_min: float, pass_max: float):
    if not 0 <= pass_max < alert_min <= 1:
        raise ValueError("需滿足 0 ≤ 放行上限 < 預警下限 ≤ 1")
    with _lock, _conn() as c:
        c.executemany("INSERT OR REPLACE INTO settings VALUES(?,?)",
                      [("alert_min", str(alert_min)), ("pass_max", str(pass_max))])


# ---------- 監控 ----------
def metrics(window: int = 50) -> dict:
    m = MONITOR
    with _conn() as c:
        fb = c.execute("SELECT task, agree FROM feedback ORDER BY id DESC LIMIT ?", (window * 3,)).fetchall()
        cases = [json.loads(r["payload"]) for r in
                 c.execute("SELECT payload FROM cases ORDER BY created DESC LIMIT 100").fetchall()]

    tasks = {}
    for task in ("damage", "cleanliness", "panel"):
        rows = [r["agree"] for r in fb if r["task"] == task][:window]
        n = len(rows)
        tasks[task] = {"n": n, "disagree": round(1 - sum(rows) / n, 3) if n else None}

    items = [it for cs in cases for it in cs.get("items", [])]
    counts = {k: sum(1 for it in items if it["route"] == k) for k in ("pass", "review", "alert")}
    total = sum(counts.values())
    review_ratio = round(counts["review"] / total, 3) if total else None

    level, reasons = 0, []
    for task, t in tasks.items():
        if t["n"] >= m["min_feedback"] and t["disagree"] is not None:
            if t["disagree"] >= m["level2_disagree"]:
                level = max(level, 2)
                reasons.append(f"{task} 不一致率 {t['disagree']:.0%} ≥ {m['level2_disagree']:.0%}")
            elif t["disagree"] >= m["level1_disagree"]:
                level = max(level, 1)
                reasons.append(f"{task} 不一致率 {t['disagree']:.0%} ≥ {m['level1_disagree']:.0%}")
    if total >= m["min_feedback"] and review_ratio is not None and review_ratio >= m["level1_review_ratio"]:
        level = max(level, 1)
        reasons.append(f"送人工審核比例 {review_ratio:.0%} ≥ {m['level1_review_ratio']:.0%}")

    base = base_thresholds()
    eff = dict(base)
    if level >= 1:
        eff = {"alert_min": min(0.99, base["alert_min"] + m["widen_by"]),
               "pass_max": max(0.01, base["pass_max"] - m["widen_by"])}
    return {"tasks": tasks, "route_counts": counts, "review_ratio": review_ratio, "level": level,
            "reasons": reasons, "thresholds_base": base, "thresholds_effective": eff, "monitor": m}


def effective_thresholds() -> dict:
    return metrics()["thresholds_effective"]
