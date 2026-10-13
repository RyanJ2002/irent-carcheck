#!/usr/bin/env python3
"""
Fine-tune 影像分類器並匯出 ONNX（給 app 直接載入）。

兩種任務：
  --task panel        拍攝面 7 類（LF RF LR RR INT_F INT_R OTHER）
  --task cleanliness  內裝整潔 3 類（clean normal dirty）

兩種資料來源：
  1) tools/download_irent.py 產生的 manifest.csv（拍攝面用 ImageType 當標籤）
       python training/train_classifier.py --task panel --manifest irent_images/manifest.csv
  2) 資料夾結構（子資料夾名稱 = 標籤），適合自己標的整潔度資料
       python training/train_classifier.py --task cleanliness --folder labeled/cleanliness
       labeled/cleanliness/clean/*.jpg  labeled/cleanliness/normal/*.jpg  labeled/cleanliness/dirty/*.jpg

重點設計：
  - 拍攝面任務「不能」做水平翻轉增強：左右翻轉會把左前變成右前，等於製造錯誤標籤。
  - 依訂單切分訓練／驗證集，避免同一台車的照片同時出現在兩邊而高估準確率。
  - 訓練後用驗證集做 temperature scaling，讓輸出機率可以當信心分數（報告 ECE 前後）。
  - --find-mislabels：2-fold 交叉預測，列出「模型很有把握、但和原標籤不同」的照片，
    輸出 suspects.csv 給人工確認（ImageType 常有誤，這一步就是清洗標籤）。
    加上 --drop-suspects 會排除這些照片後再訓練最終模型。

需要：pip install -r requirements-train.txt（torch、torchvision）
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PANEL_LABELS = ["LF", "RF", "LR", "RR", "INT_F", "INT_R", "OTHER"]
CLEAN_LABELS = ["clean", "normal", "dirty"]
ITYPE = {1: "LF", 2: "RF", 3: "LR", 4: "RR", 10: "INT_F", 11: "INT_R"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------------- 資料 ----------------
def load_manifest(path: Path):
    root = path.parent
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("status") not in (None, "", "ok", "skip"):
                continue
            try:
                lab = ITYPE.get(int(float(r["itype"])), "OTHER")
            except (KeyError, ValueError):
                continue
            p = root / r["local_path"]
            if p.exists():
                out.append({"path": p, "label": lab, "group": r.get("order") or r.get("car") or str(p)})
    return out


def load_folder(path: Path, labels):
    out = []
    for lab in labels:
        d = path / lab
        for p in sorted(d.rglob("*")) if d.exists() else []:
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                out.append({"path": p, "label": lab, "group": p.stem})
    return out


def split_by_group(samples, val_frac, seed):
    groups = sorted({s["group"] for s in samples})
    random.Random(seed).shuffle(groups)
    val_g = set(groups[: max(1, int(len(groups) * val_frac))])
    return [s for s in samples if s["group"] not in val_g], [s for s in samples if s["group"] in val_g]


class ImgDS(Dataset):
    def __init__(self, samples, labels, tf):
        self.s, self.idx, self.tf = samples, {l: i for i, l in enumerate(labels)}, tf

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        img = ImageOps.exif_transpose(Image.open(s["path"])).convert("RGB")
        return self.tf(img), self.idx[s["label"]]


def transforms_for(task, size):
    aug = [transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
           transforms.ColorJitter(0.3, 0.3, 0.2, 0.03),
           transforms.RandomApply([transforms.GaussianBlur(5)], p=0.15)]
    if task == "cleanliness":
        aug.append(transforms.RandomHorizontalFlip())   # 整潔度與左右無關，可以翻轉
    # panel：刻意不翻轉（翻轉會讓左前變右前）
    train_tf = transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                                  transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train_tf, eval_tf


# ---------------- 模型 ----------------
def build_model(n_cls, pretrained=True):
    w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    try:
        m = models.mobilenet_v3_large(weights=w)
    except Exception as e:  # noqa: BLE001  離線時退回隨機初始化
        print(f"[警告] 無法下載 ImageNet 預訓練權重（{e}），改用隨機初始化，準確率會明顯較差")
        m = models.mobilenet_v3_large(weights=None)
        w = None
    if w is None:  # 從頭訓練時，torchvision 預設的 BN momentum=0.01 太慢，評估時統計量會失準
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.momentum = 0.1
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_cls)
    return m


def run_epoch(model, dl, opt, device, train):
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05 if train else 0.0)
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = lossf(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        tot += len(y)
        correct += (out.argmax(1) == y).sum().item()
        loss_sum += loss.item() * len(y)
    return loss_sum / max(1, tot), correct / max(1, tot)


@torch.no_grad()
def logits_of(model, dl, device):
    model.eval()
    L, Y = [], []
    for x, y in dl:
        L.append(model(x.to(device)).cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def fit_temperature(logits, y):
    t = torch.zeros(1, requires_grad=True)  # log T
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.detach().exp().clamp(0.05, 20))


def ece(probs, y, bins=10):
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    e = torch.zeros(1)
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return float(e)


def train(samples_tr, samples_va, labels, args, device, verbose=True):
    train_tf, eval_tf = transforms_for(args.task, args.size)
    # 類別不平衡：依頻率加權抽樣
    cnt = Counter(s["label"] for s in samples_tr)
    weights = [1.0 / cnt[s["label"]] for s in samples_tr]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(samples_tr), replacement=True)
    dl_tr = DataLoader(ImgDS(samples_tr, labels, train_tf), batch_size=args.batch, sampler=sampler,
                       num_workers=args.workers)
    dl_va = DataLoader(ImgDS(samples_va, labels, eval_tf), batch_size=args.batch, num_workers=args.workers)

    model = build_model(len(labels), not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state = -1, None
    for ep in range(args.epochs):
        tl, ta = run_epoch(model, dl_tr, opt, device, True)
        vl, va = run_epoch(model, dl_va, None, device, False)
        sched.step()
        if verbose:
            print(f"  epoch {ep + 1:>2}/{args.epochs}  train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")
        if va > best:
            best, best_state = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, dl_va


@torch.no_grad()
def predict_samples(model, samples, labels, args, device):
    _, eval_tf = transforms_for(args.task, args.size)
    dl = DataLoader(ImgDS(samples, labels, eval_tf), batch_size=args.batch, num_workers=args.workers)
    L, _ = logits_of(model, dl, device)
    return L.softmax(1)


def find_mislabels(samples, labels, args, device):
    """2-fold 交叉預測：每張照片都由「沒看過它」的模型預測。"""
    groups = sorted({s["group"] for s in samples})
    random.Random(args.seed + 1).shuffle(groups)
    half = set(groups[: len(groups) // 2])
    folds = [[s for s in samples if s["group"] in half], [s for s in samples if s["group"] not in half]]
    suspects = []
    for k in range(2):
        tr, te = folds[1 - k], folds[k]
        print(f"[清洗] fold {k + 1}/2：用 {len(tr)} 張訓練，預測 {len(te)} 張")
        sub_tr, sub_va = split_by_group(tr, 0.1, args.seed)
        model, _ = train(sub_tr, sub_va, labels, args, device, verbose=False)
        P = predict_samples(model, te, labels, args, device)
        for s, p in zip(te, P):
            given = labels.index(s["label"])
            top = int(p.argmax())
            # 模型認為另一類明顯比原標籤更可能（差距 >= suspect_margin）就列為疑似錯標
            if top != given and float(p[top] - p[given]) >= args.suspect_margin:
                suspects.append({"path": str(s["path"]), "given": s["label"], "predicted": labels[top],
                                 "p_predicted": round(float(p[top]), 3), "p_given": round(float(p[given]), 3)})
    return suspects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["panel", "cleanliness"], required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path)
    src.add_argument("--folder", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 張（快速測試）")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--find-mislabels", action="store_true")
    ap.add_argument("--drop-suspects", action="store_true")
    ap.add_argument("--suspect-margin", type=float, default=0.5,
                    help="預測類別機率 - 原標籤機率 >= 此值才列為疑似錯標；越高越保守")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    labels = PANEL_LABELS if args.task == "panel" else CLEAN_LABELS
    samples = load_manifest(args.manifest) if args.manifest else load_folder(args.folder, labels)
    if args.limit:
        random.Random(args.seed).shuffle(samples)
        samples = samples[: args.limit]
    present = sorted({s["label"] for s in samples}, key=labels.index)
    if len(samples) < 20 or len(present) < 2:
        sys.exit(f"資料不足：{len(samples)} 張、{len(present)} 個類別")
    print(f"共 {len(samples)} 張，類別分布：{dict(Counter(s['label'] for s in samples))}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"裝置：{device}")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.find_mislabels:
        sus = find_mislabels(samples, labels, args, device)
        sus_path = args.out / f"{args.task}_suspects.csv"
        with open(sus_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["path", "given", "predicted", "p_predicted", "p_given"])
            w.writeheader()
            w.writerows(sus)
        rate = len(sus) / len(samples)
        print(f"[清洗] 疑似錯標 {len(sus)} 張（{rate:.1%}），清單：{sus_path}")
        pairs = Counter((s["given"], s["predicted"]) for s in sus).most_common(8)
        for (g, p), c in pairs:
            print(f"        原標 {g:>5} → 模型認為 {p:>5}：{c} 張")
        if args.drop_suspects:
            bad = {s["path"] for s in sus}
            samples = [s for s in samples if str(s["path"]) not in bad]
            print(f"[清洗] 排除後剩 {len(samples)} 張")

    tr, va = split_by_group(samples, args.val, args.seed)
    print(f"訓練 {len(tr)} 張／驗證 {len(va)} 張（依訂單切分）")
    model, dl_va = train(tr, va, labels, args, device)

    L, Y = logits_of(model, dl_va, device)
    T = fit_temperature(L, Y)
    p0, p1 = L.softmax(1), (L / T).softmax(1)
    acc = float((p1.argmax(1) == Y).float().mean())
    print(f"驗證準確率 {acc:.3f}｜temperature {T:.3f}｜ECE 校準前 {ece(p0, Y):.3f} → 校準後 {ece(p1, Y):.3f}")
    cm = defaultdict(Counter)
    for t, p in zip(Y.tolist(), p1.argmax(1).tolist()):
        cm[labels[t]][labels[p]] += 1
    print("混淆矩陣（列 = 真實，欄 = 預測）")
    print("        " + " ".join(f"{l:>6}" for l in labels))
    for t in labels:
        print(f"{t:>7} " + " ".join(f"{cm[t][p]:>6}" for p in labels))

    model = model.cpu().eval()
    onnx_path = args.out / f"{args.task}_classifier.onnx"
    torch.onnx.export(model, torch.randn(1, 3, args.size, args.size), str(onnx_path),
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    meta = {"labels": labels, "input_size": args.size, "mean": MEAN, "std": STD, "temperature": T,
            "val_accuracy": acc, "ece_before": ece(p0, Y), "ece_after": ece(p1, Y),
            "n_train": len(tr), "n_val": len(va), "arch": "mobilenet_v3_large"}
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已匯出 {onnx_path}（重新啟動 app 即會自動改用這個模型）")


if __name__ == "__main__":
    main()
