# 車況管家 Demo：AI 取還車照片判讀

2026 和泰 AI 黑客松 iRent 題目的原型。上傳照片就能跑完整流程：

| 頁面 | 對應題目 | 做什麼 |
|---|---|---|
| 拍照防呆 | 環節一 | 模擬 App 拍照流程：每拍一張就即時檢查模糊、亮度、反光、鏡頭髒污、距離、拍攝位置、車牌，不合格當場請用戶重拍 |
| 照片自動分類 | 環節一延伸 | 一次丟一批照片，自動分成左前、右前、左後、右後、前車內、後車內、其他，並標出與原上傳分類不一致的照片 |
| 還車判讀 | 環節二 | 取車照與還車照成對比較，找出新車損；內裝分成髒汙、普通、乾淨；依信心分數分流成自動放行、人工審核、預警 |
| 營運儀表板 | 環節三 + 自我監控 | 案件佇列、前後對照、審核回饋、下一位用戶評價、門檻調整，以及監控等級（第 1 級自動擴大人工審核、第 2 級通知工程師） |

## 快速開始

```bash
python -m venv .venv && source .venv/bin/activate   # Windows：.venv\Scripts\activate
pip install -r requirements.txt
python scripts/fetch_models.py        # 下載車輛偵測模型 YOLO11n（約 11MB）
cp .env.example .env                  # 填入 GEMINI_API_KEY（沒有的話見下方「沒有金鑰也能跑」）
uvicorn app.main:app --reload --port 8000
```

打開 http://localhost:8000 。手機和電腦在同一個網路時，用 `--host 0.0.0.0` 啟動，手機開 `http://電腦IP:8000` 就能直接用手機相機拍照測試。

### 沒有金鑰也能跑

```bash
DEMO_MOCK=1 uvicorn app.main:app --port 8000
```

模擬模式下，拍攝面和整潔度是隨機的「模擬結果」，畫面頂端會有黃色橫幅標示，只用來展示介面。品質檢查、車輛偵測、新車損的影像差異基準線都是真的計算。

## 每個判斷用什麼做

右上角的狀態列會顯示目前實際啟用的後端。

| 判斷 | 實作 | 類型 |
|---|---|---|
| 解析度、模糊、亮度、反光、鏡頭髒污 | OpenCV 規則（拉普拉斯變異數、亮度、對比、過曝比例） | 規則 |
| 車輛入鏡、太遠、太近 | YOLO11n（COCO 預訓練）車輛框佔畫面比例 | 現成模型 |
| 拍攝面分類 | ① `models/panel_classifier.onnx`（fine-tune）→ ② Gemini → ③ 模擬 | fine-tune／現成 |
| 車牌入鏡與比對 | Gemini 讀車牌，與訂單車號比對 | 現成模型 |
| 新車損 | Gemini 成對比較，問 N 次取一致比例當機率；沒有 Gemini 時用影像差異基準線 | 現成模型 |
| 內裝整潔 | ① `models/cleanliness_classifier.onnx`（fine-tune）→ ② Gemini → ③ 模擬 | fine-tune／現成 |

幾個設計重點：

- **信心分數都經過處理才拿來分流。** fine-tune 模型用 temperature scaling 校準；Gemini 本身不給可靠機率，所以同一題問 `GEMINI_SAMPLES` 次，用答案一致的比例當機率。
- **影像差異基準線不是 AI。** 它用 ORB 特徵把兩張照片對齊再找差異區域，只當示意和後備。它的機率被限制在 0.05～0.75，所以永遠不會自動預警，有差異就送人工審核。
- **拍不到、缺照片、照片品質差、兩張無法對應**，一律分到「人工審核」並寫明原因，不會被當成「沒有異常」。
- **分流門檻由營運決定**，在儀表板調整。

## 用 iRent 資料 fine-tune

```bash
pip install -r requirements-train.txt

# 1. 下載 Excel 裡的照片（依 車號/訂單/pickup|return 分資料夾，輸出 manifest.csv）
python tools/download_irent.py 資料.xlsx --out irent_images --workers 16

# 2. 先找出疑似錯標（ImageType 常有誤），輸出 models/panel_suspects.csv 給人工確認
python training/train_classifier.py --task panel --manifest irent_images/manifest.csv --find-mislabels

# 3. 排除錯標後訓練最終模型，匯出 models/panel_classifier.onnx
python training/train_classifier.py --task panel --manifest irent_images/manifest.csv --find-mislabels --drop-suspects

# 內裝整潔：把自己標好的照片放成 labeled/cleanliness/{clean,normal,dirty}/*.jpg
python training/train_classifier.py --task cleanliness --folder labeled/cleanliness
```

訓練完重新啟動 app，就會自動改用 fine-tune 的模型。訓練腳本的幾個重點：

- **拍攝面任務不做水平翻轉增強**，因為翻轉會把左前變成右前，等於製造錯誤標籤。
- **依訂單切分訓練集和驗證集**，同一台車的照片不會同時出現在兩邊，避免高估準確率。
- **`--find-mislabels` 用 2-fold 交叉預測**：每張照片都由沒看過它的模型預測，列出「模型很有把握但和原標籤不同」的照片。疑似錯標的比例本身就是「上傳分類常有誤」的量化證據。
- 輸出驗證準確率、混淆矩陣，以及校準前後的 ECE。

## API

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/api/status` | 目前啟用的後端 |
| POST | `/api/check` | 單張照片即時防呆。`file`、`target`（LF/RF/LR/RR/INT_F/INT_R）、`car_no` |
| POST | `/api/classify` | 批次拍攝面分類。`files` 多檔 |
| POST | `/api/inspect` | 還車判讀。`pickup_LF`、`return_LF`、…、`return_INT_F`、`return_INT_R`、`order_no`、`car_no` |
| GET | `/api/cases`、`/api/cases/{id}` | 案件 |
| POST | `/api/cases/{id}/review` | 審核回饋 `{item_key, human_abnormal}` 或下一位用戶評價 `{item_key, human_level, source:"next_renter"}` |
| GET | `/api/metrics` | 監控指標與分級 |
| PUT | `/api/thresholds` | 調整分流門檻 `{pass_max, alert_min}` |
| POST | `/api/simulate-feedback` | Demo 用：灌入模擬回饋，展示監控升級 |

## 和 iRent 系統的介面假設

這個原型只假設 iRent 能提供三件事，其他元件都由本系統自己提供：

1. **讀取**：照片網址，加上 `order_number`、`CarNo`、`ImageType`、時間。解題資料的 Excel 已經證明這些資料存在。
2. **觸發**：照片上傳完成的通知。若沒有，改用定時輪詢新照片。
3. **寫回**：判讀結果送到哪個系統。若沒有，使用本系統的儀表板加上 LINE 推播。

App 端的即時防呆是獨立模組：輸入一張相機畫面和要拍的那一面，輸出通過或不通過加上原因。正式導入時，拍攝面分類器匯出成 TFLite／CoreML，品質規則直接移植到 App 端。

## 測試

```bash
pip install pytest && pytest -q
```

測試只用合成影像，不需要網路和金鑰。

## 專案結構

```
app/
  main.py              API 與分流邏輯
  config.py            拍攝面定義、所有門檻（可用環境變數覆寫）
  store.py             SQLite：案件、回饋、監控指標
  analyzers/
    quality.py         品質規則
    detector.py        YOLO11n 車輛偵測
    panel.py           拍攝面分類（ONNX / Gemini / 模擬）
    cleanliness.py     內裝整潔（ONNX / Gemini / 模擬）
    damage.py          新車損成對比較（Gemini / 影像差異基準線）
    gemini.py          Gemini 呼叫與自我一致性
    onnx_cls.py        fine-tune 模型推論
  static/              前端（原生 HTML/CSS/JS，不需要建置）
training/train_classifier.py   fine-tune、清洗標籤、校準、匯出 ONNX
tools/download_irent.py        依 Excel 批次下載照片
scripts/fetch_models.py        下載 YOLO11n
tests/                         離線測試
```
