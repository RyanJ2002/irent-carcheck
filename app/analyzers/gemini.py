"""Gemini 呼叫封裝 + 自我一致性（同一題問 N 次，用答案一致程度當信心）。

啟用方式（擇一）：
  - 設定 GEMINI_API_KEY（或 GOOGLE_API_KEY）
  - Vertex AI：GOOGLE_GENAI_USE_VERTEXAI=true、GOOGLE_CLOUD_PROJECT、GOOGLE_CLOUD_LOCATION
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

from ..config import GEMINI_MODEL, GEMINI_SAMPLES

_client = None


def available() -> bool:
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return True
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true") and bool(os.getenv("GOOGLE_CLOUD_PROJECT"))


def _get():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client()
    return _client


def ask_json(prompt: str, images: list[bytes], schema: dict, samples: int | None = None,
             temperature: float = 1.0) -> list[dict]:
    """回傳 samples 個獨立回答（JSON）。失敗的樣本會被略過；全部失敗則丟出例外。"""
    from google.genai import types

    n = samples or GEMINI_SAMPLES
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images] + [prompt]
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_json_schema=schema,
    )

    def one(_):
        resp = _get().models.generate_content(model=GEMINI_MODEL, contents=parts, config=cfg)
        return json.loads(resp.text)

    results, errors = [], []
    with ThreadPoolExecutor(n) as ex:
        for fut in [ex.submit(one, i) for i in range(n)]:
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")
    if not results:
        raise RuntimeError("Gemini 呼叫全部失敗：" + "; ".join(errors[:2]))
    return results


def vote(values: list, labels: list) -> dict:
    """把 N 次回答轉成各類別比例（作為機率估計）。"""
    probs = {k: 0.0 for k in labels}
    for v in values:
        if v in probs:
            probs[v] += 1
    total = max(1, len(values))
    return {k: round(c / total, 3) for k, c in probs.items()}
