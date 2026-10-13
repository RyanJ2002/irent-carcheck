#!/usr/bin/env python3
"""
iRent 解題資料下載器
讀取 Excel（order_number / CarNo / ImageType / Image / 下載網址 / 上傳分類 / 時間），
依「車號 / 訂單 / 取車或還車」分資料夾下載照片，可中斷續傳，並輸出 manifest.csv。

用法：
    pip install pandas openpyxl requests
    python download_irent.py 資料.xlsx --out irent_images --workers 16

取車／還車判斷：同一訂單內依時間排序，找最大的時間間隔切成兩段，
前段 = pickup（取車），後段 = return（還車）。
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import pandas as pd
import requests

TYPE_NAME = {1: "LF", 2: "RF", 3: "LR", 4: "RR", 10: "INT_F", 11: "INT_R"}  # 其餘 = OTHER

# Excel 欄名可能有空白或些微差異，用關鍵字對應
COL_KEYS = {
    "order": ["order_number", "order"],
    "car": ["CarNo", "car"],
    "itype": ["ImageType"],
    "image": ["Image"],
    "url": ["下載網址", "url", "URL"],
    "upload_cat": ["上傳分類"],
    "time": ["時間", "time"],
}


def find_cols(df):
    cols = {}
    for key, cands in COL_KEYS.items():
        for c in df.columns:
            name = str(c).strip()
            if any(name == k or name.startswith(k) for k in cands):
                # Image 與 ImageType 前綴相同，避免誤配
                if key == "image" and name.startswith("ImageType"):
                    continue
                cols[key] = c
                break
    missing = [k for k in ("order", "car", "itype", "url", "time") if k not in cols]
    if missing:
        sys.exit(f"找不到欄位：{missing}\n目前欄位：{list(df.columns)}")
    return cols


def assign_phase(df):
    """每張訂單內找最大時間間隔，切成 pickup / return。"""
    df = df.sort_values(["order", "time"]).copy()
    df["phase"] = "pickup"
    for _, g in df.groupby("order", sort=False):
        if len(g) < 2:
            continue
        gaps = g["time"].diff()
        split_at = gaps.idxmax()
        if pd.isna(gaps.max()) or gaps.max() <= pd.Timedelta(0):
            continue
        df.loc[g.index[g.index.get_loc(split_at):], "phase"] = "return"
    return df


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))


def build_jobs(df, out):
    jobs = []
    df["seq"] = df.groupby(["order", "phase", "itype"]).cumcount()
    for r in df.itertuples():
        ext = os.path.splitext(urlparse(str(r.url)).path)[1].lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".heic"):
            ext = ".jpg"
        tname = TYPE_NAME.get(int(r.itype) if pd.notna(r.itype) else -1, "OTHER")
        fname = f"{r.time:%Y%m%d-%H%M%S}_{tname}_{int(r.itype) if pd.notna(r.itype) else 'NA'}_{r.seq}{ext}"
        path = os.path.join(out, safe(r.car), safe(r.order), r.phase, fname)
        jobs.append((r.Index, str(r.url).strip(), path))
    return jobs


def fetch(session, url, path, retries=3):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skip", ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    for attempt in range(retries):
        try:
            with session.get(url, timeout=30, stream=True) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(1 << 16):
                        f.write(chunk)
            os.replace(tmp, path)
            return "ok", ""
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    if os.path.exists(tmp):
        os.remove(tmp)
    return "fail", err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--out", default="irent_images")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--limit", type=int, default=0, help="只下載前 N 張（測試用）")
    args = ap.parse_args()

    raw = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    c = find_cols(raw)
    df = pd.DataFrame({
        "order": raw[c["order"]].astype(str).str.strip(),
        "car": raw[c["car"]].astype(str).str.strip(),
        "itype": pd.to_numeric(raw[c["itype"]], errors="coerce"),
        "image": raw[c["image"]] if "image" in c else "",
        "url": raw[c["url"]],
        "upload_cat": raw[c["upload_cat"]] if "upload_cat" in c else "",
        "time": pd.to_datetime(raw[c["time"]], errors="coerce"),
    })
    bad = df["url"].isna() | df["time"].isna()
    if bad.any():
        print(f"略過 {bad.sum()} 列（缺網址或時間）")
    df = df[~bad]
    df = assign_phase(df)
    if args.limit:
        df = df.head(args.limit)

    jobs = build_jobs(df, args.out)
    print(f"共 {len(jobs)} 張，{df['order'].nunique()} 筆訂單，{df['car'].nunique()} 台車")

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=args.workers)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    status = {}
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, session, u, p): (i, p) for i, u, p in jobs}
        for f in as_completed(futs):
            i, p = futs[f]
            status[i] = (*f.result(), p)
            done += 1
            if done % 200 == 0 or done == len(jobs):
                n_fail = sum(1 for s in status.values() if s[0] == "fail")
                print(f"  {done}/{len(jobs)}  失敗 {n_fail}", flush=True)

    df["local_path"] = [os.path.relpath(status[i][2], args.out) for i in df.index]
    df["status"] = [status[i][0] for i in df.index]
    df["error"] = [status[i][1] for i in df.index]
    os.makedirs(args.out, exist_ok=True)
    man = os.path.join(args.out, "manifest.csv")
    df.drop(columns=["seq"]).to_csv(man, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    n_fail = (df["status"] == "fail").sum()
    print(f"完成。manifest：{man}；失敗 {n_fail} 張（重跑同指令即可只補抓失敗的）")


if __name__ == "__main__":
    main()
