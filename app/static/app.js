"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = v => v == null ? "—" : `${Math.round(v * 100)}%`;
const ZH = { LF: "左前", RF: "右前", LR: "左後", RR: "右後", INT_F: "前車內", INT_R: "後車內", OTHER: "其他" };
const ORDER = ["LF", "RF", "LR", "RR", "INT_F", "INT_R"];
const ROUTE = { pass: "自動放行", review: "人工審核", alert: "預警" };
const CLEAN = { clean: "乾淨", normal: "普通", dirty: "髒汙" };
const ISSUE = { trash: "垃圾", liquid_stain: "液體污漬", hair: "毛髮", mud_sand: "泥沙", food: "食物殘渣", other: "其他" };
const METHOD = { onnx: "fine-tune 模型", gemini: "Gemini", mock: "模擬", none: "未啟用", baseline: "影像差異基準線", error: "錯誤" };
let STATUS = {};

function toast(msg, ms = 2600) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), ms);
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body; try { body = await r.json(); } catch { body = {}; }
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  return body;
}
const jpost = (path, data, method = "POST") =>
  api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

function boxed(url, boxes = []) {
  const w = document.createElement("div"); w.className = "boxwrap";
  w.innerHTML = `<img src="${esc(url)}" alt="">` + boxes.filter(b => b.box).map(b => {
    const [x1, y1, x2, y2] = b.box;
    return `<div class="bbox ${b.cls || ""}" style="left:${x1 * 100}%;top:${y1 * 100}%;width:${(x2 - x1) * 100}%;height:${(y2 - y1) * 100}%">${b.label ? `<span>${esc(b.label)}</span>` : ""}</div>`;
  }).join("");
  return w;
}
function bars(probs) {
  const ks = Object.keys(probs || {}).sort((a, b) => probs[b] - probs[a]);
  return `<div class="bars">${ks.map(k => `<div class="bar"><span>${esc(ZH[k] || CLEAN[k] || k)}</span><i><b style="width:${probs[k] * 100}%"></b></i><span>${pct(probs[k])}</span></div>`).join("")}</div>`;
}

/* ---------------- 狀態列與分頁 ---------------- */
async function loadStatus() {
  STATUS = await api("/api/status");
  const p = (label, val, ok) => `<span class="pill ${ok ? "ok" : "warn"}">${label}：${val}</span>`;
  $("#status").innerHTML =
    p("拍攝面", METHOD[STATUS.panel], ["onnx", "gemini"].includes(STATUS.panel)) +
    p("整潔度", METHOD[STATUS.cleanliness], ["onnx", "gemini"].includes(STATUS.cleanliness)) +
    p("車損", STATUS.damage === "gemini" ? "Gemini 成對比較" : "影像差異基準線", STATUS.damage === "gemini") +
    p("車輛偵測", STATUS.vehicle_detector ? "YOLO11n" : "未安裝", STATUS.vehicle_detector);
  $("#mockbar").hidden = !STATUS.mock || STATUS.gemini;
}
function show() {
  const page = (location.hash || "#capture").slice(1);
  $$(".page").forEach(s => s.classList.toggle("on", s.id === `page-${page}`));
  $$("#nav a").forEach(a => a.classList.toggle("on", a.dataset.page === page));
  if (page === "ops") loadOps();
}
window.addEventListener("hashchange", show);

/* ---------------- 1. 拍照防呆 ---------------- */
const cap = { i: 0, res: {} };  // res[code] = {file, url, data, skipped}
function carSvg(code) {
  const spot = { LF: [14, 14], RF: [106, 14], LR: [14, 186], RR: [106, 186] }[code];
  const corner = { LF: [36, 34], RF: [84, 34], LR: [36, 166], RR: [84, 166] }[code];
  let hl = "";
  if (spot) hl = `<line x1="${spot[0]}" y1="${spot[1]}" x2="${corner[0]}" y2="${corner[1]}" stroke="#0f6e56" stroke-width="3" stroke-dasharray="4 3"/><circle cx="${spot[0]}" cy="${spot[1]}" r="9" fill="#0f6e56"/><circle cx="${spot[0]}" cy="${spot[1]}" r="3.5" fill="#fff"/>`;
  if (code === "INT_F") hl = `<rect x="33" y="62" width="54" height="40" rx="6" fill="#0f6e56" opacity=".35"/>`;
  if (code === "INT_R") hl = `<rect x="33" y="106" width="54" height="38" rx="6" fill="#0f6e56" opacity=".35"/>`;
  return `<rect x="26" y="18" width="68" height="164" rx="24" fill="#f1f1ec" stroke="#888780" stroke-width="2"/>
    <path d="M34 56 Q60 44 86 56 L82 66 Q60 58 38 66Z" fill="#d3d1c7"/><path d="M36 150 Q60 158 84 150 L82 142 Q60 148 38 142Z" fill="#d3d1c7"/>
    <text x="60" y="13" text-anchor="middle" font-size="10" fill="#6c6f68">車頭</text>${hl}`;
}
function renderCapture() {
  const code = ORDER[cap.i];
  $("#cap-steps").innerHTML = ORDER.map((c, i) => {
    const r = cap.res[c]; const cls = i === cap.i ? "cur" : r?.skipped ? "skip" : r?.data?.verdict === "pass" ? "done" : "";
    return `<div class="step ${cls}" data-i="${i}">${ZH[c]}</div>`;
  }).join("");
  const done = ORDER.filter(c => cap.res[c]?.data?.verdict === "pass" || cap.res[c]?.skipped).length;
  $("#cap-progress").textContent = `${done} / 6 完成`;
  $("#car-diagram").innerHTML = carSvg(code);
  $("#cap-target").textContent = `請拍「${ZH[code]}」`;
  $("#cap-hint").textContent = code.startsWith("INT") ? "打開車門，拍攝整個" + (code === "INT_F" ? "前座與儀表板" : "後座與腳踏墊")
    : "站在車角斜 45 度，讓車頭（或車尾）、側面和車牌都入鏡";
  const r = cap.res[code];
  const pv = $("#cap-preview");
  if (r?.url) {
    pv.className = "preview"; pv.innerHTML = "";
    const v = r.data?.vehicle;
    pv.append(boxed(r.url, v ? [{ box: v.box, cls: "car", label: `車輛 ${pct(v.area_ratio)} 畫面` }] : []));
  } else { pv.className = "preview empty"; pv.innerHTML = r?.skipped ? `<span>已略過：${esc(r.skipped)}</span>` : "<span>尚未拍攝</span>"; }
  const out = $("#cap-result");
  if (r?.data) {
    const d = r.data;
    out.innerHTML = `<div class="verdict ${d.verdict}">${d.verdict === "pass" ? "✓ 照片合格" : "✕ 請重拍"}</div>
      <ul class="checks">${d.checks.map(c => `<li><span class="ic ${c.ok === true ? "ok" : c.ok === false ? "bad" : "na"}">${c.ok === true ? "✓" : c.ok === false ? "✕" : "–"}</span><span><b>${esc(c.label)}</b>　${esc(c.message)}</span></li>`).join("")}</ul>`;
    const p = d.panel || {};
    $("#cap-detail").innerHTML = p.panel ? `<div class="kv">判斷方式 <b>${METHOD[p.method] || p.method}</b>　預測 <b>${ZH[p.panel]}</b>　信心 <b>${pct(p.confidence)}</b></div>${bars(p.probs)}`
      : `<span class="muted">${p.error ? "拍攝面分類錯誤：" + esc(p.error) : "拍攝面分類未啟用"}</span>`;
  } else out.innerHTML = "";
  $("#cap-finish").hidden = done < 6;
}
async function captureFile(file) {
  const code = ORDER[cap.i];
  const url = URL.createObjectURL(file);
  cap.res[code] = { file, url, data: null };
  renderCapture();
  $("#cap-result").innerHTML = `<div class="verdict"><span class="spin"></span> 檢查中…</div>`;
  const fd = new FormData(); fd.append("file", file); fd.append("target", code); fd.append("car_no", $("#cap-carno").value);
  try {
    cap.res[code].data = await api("/api/check", { method: "POST", body: fd });
    if (cap.res[code].data.verdict === "pass") {
      const next = ORDER.findIndex((c, i) => i > cap.i && !(cap.res[c]?.data?.verdict === "pass" || cap.res[c]?.skipped));
      renderCapture();
      if (next >= 0) setTimeout(() => { cap.i = next; renderCapture(); toast(`${ZH[code]}完成，下一張：${ZH[ORDER[next]]}`); }, 1400);
      return;
    }
  } catch (e) { toast("檢查失敗：" + e.message); }
  renderCapture();
}
$("#cap-input").addEventListener("change", e => { const f = e.target.files[0]; e.target.value = ""; if (f) captureFile(f); });
$("#cap-steps").addEventListener("click", e => { const s = e.target.closest(".step"); if (s) { cap.i = +s.dataset.i; renderCapture(); } });
$("#cap-skip").addEventListener("click", () => ($("#cap-skipbox").hidden = !$("#cap-skipbox").hidden));
$("#cap-skip-ok").addEventListener("click", () => {
  const code = ORDER[cap.i];
  cap.res[code] = { skipped: $("#cap-skip-reason").value };
  $("#cap-skipbox").hidden = true;
  const next = ORDER.findIndex(c => !(cap.res[c]?.data?.verdict === "pass" || cap.res[c]?.skipped));
  if (next >= 0) cap.i = next;
  renderCapture();
});
$("#cap-finish").addEventListener("click", () => {
  for (const c of ORDER) if (cap.res[c]?.file) ins.files[`return_${c}`] = cap.res[c].file;
  if ($("#cap-carno").value) $("#ins-car").value = $("#cap-carno").value;
  renderSlots(); location.hash = "#inspect";
  toast("已帶入還車照，請補上取車照後開始判讀");
});

/* ---------------- 2. 照片自動分類 ---------------- */
const cls = { items: [] };
function renderClassify() {
  const items = cls.items;
  const done = items.filter(i => i.r);
  const mis = done.filter(i => i.r.mismatch).length;
  const withOrig = done.filter(i => i.r.original).length;
  const low = done.filter(i => i.r.panel.panel && i.r.panel.confidence < 0.6).length;
  const bad = done.filter(i => i.r.quality_issues.length).length;
  $("#cls-summary").innerHTML = !items.length ? "" : `
    <div class="kpi"><small>照片數</small><b>${done.length}</b> / ${items.length}</div>
    <div class="kpi alert"><small>與原上傳分類不一致</small><b>${mis}</b> ${withOrig ? `（${pct(mis / withOrig)}）` : ""}</div>
    <div class="kpi review"><small>低信心（&lt;60%），建議人工確認</small><b>${low}</b></div>
    <div class="kpi"><small>品質不合格</small><b>${bad}</b></div>`;
  const groups = {};
  for (const it of items) { const k = it.r ? (it.r.panel.panel || "NA") : "PENDING"; (groups[k] ||= []).push(it); }
  const keys = [...ORDER, "OTHER", "NA", "PENDING"].filter(k => groups[k]);
  $("#cls-board").innerHTML = keys.map(k => `<div class="col"><h4><span>${k === "NA" ? "未分類" : k === "PENDING" ? "分析中…" : ZH[k]}</span><span class="muted">${groups[k].length}</span></h4>
    ${groups[k].map(it => {
      const r = it.r; const idx = items.indexOf(it);
      if (!r) return `<div class="thumb"><img src="${it.url}" alt=""><div class="meta"><span class="spin"></span></div></div>`;
      const p = r.panel;
      return `<div class="thumb ${r.mismatch ? "mis" : ""}"><img src="${it.url}" alt="" title="${esc(r.filename)}"><div class="meta">
        <div><b>${p.panel ? ZH[p.panel] : "—"}</b> <span class="muted">${pct(p.confidence)}</span>${it.fixed ? ` <span class="tag">已更正</span>` : ""}</div>
        ${r.original ? `<span class="tag ${r.mismatch ? "bad" : ""}">原分類：${ZH[r.original] || r.original}</span>` : ""}
        ${p.panel && p.confidence < 0.6 ? `<span class="tag warn">低信心</span>` : ""}
        ${r.quality_issues.map(q => `<span class="tag warn">${esc(q)}</span>`).join("")}
        ${p.error ? `<span class="tag bad">${esc(p.error).slice(0, 60)}</span>` : ""}
        <select data-i="${idx}"><option value="">更正分類…</option>${[...ORDER, "OTHER"].map(c => `<option value="${c}">${ZH[c]}</option>`).join("")}</select>
      </div></div>`;
    }).join("")}</div>`).join("");
}
async function classifyFiles(files) {
  const fresh = [...files].filter(f => f.type.startsWith("image/")).map(f => ({ file: f, url: URL.createObjectURL(f), r: null }));
  if (!fresh.length) return;
  cls.items.push(...fresh); renderClassify();
  for (let i = 0; i < fresh.length; i += 6) {
    const batch = fresh.slice(i, i + 6); const fd = new FormData();
    batch.forEach(b => fd.append("files", b.file, b.file.name));
    try {
      const res = await api("/api/classify", { method: "POST", body: fd });
      res.results.forEach((r, j) => (batch[j].r = r));
    } catch (e) {
      toast("分類失敗：" + e.message);
      batch.forEach(b => (b.r = { filename: b.file.name, panel: { panel: null, confidence: 0, error: e.message }, original: null, mismatch: false, quality_issues: [] }));
    }
    renderClassify();
  }
}
const drop = $("#cls-drop");
drop.addEventListener("click", () => $("#cls-input").click());
$("#cls-input").addEventListener("change", e => { classifyFiles(e.target.files); e.target.value = ""; });
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); classifyFiles(e.dataTransfer.files); });
$("#cls-board").addEventListener("change", async e => {
  const s = e.target.closest("select[data-i]"); if (!s || !s.value) return;
  const it = cls.items[+s.dataset.i]; const ai = it.r.panel.panel || "OTHER";
  try { await jpost("/api/feedback/panel", { ai_panel: ai, human_panel: s.value, confidence: it.r.panel.confidence || 0 }); } catch (e2) { toast(e2.message); }
  it.r.panel = { ...it.r.panel, panel: s.value, confidence: 1 }; it.fixed = true;
  it.r.mismatch = !!(it.r.original && it.r.original !== s.value);
  renderClassify(); toast("已記錄更正，會回流成訓練資料");
});

/* ---------------- 3. 還車判讀 ---------------- */
const ins = { files: {} };
function renderSlots() {
  const cell = key => {
    if (!key) return `<div class="slot na">不需要</div>`;
    const f = ins.files[key];
    return `<div class="slot" data-key="${key}">${f ? `<img src="${URL.createObjectURL(f)}" alt=""><button class="x" data-del="${key}">移除</button>` : "＋ 點擊或拖曳上傳"}</div>`;
  };
  $("#ins-slots").innerHTML = `<div></div><div class="hd">取車照（上一手的基準）</div><div class="hd">還車照（本次）</div>` +
    ORDER.map(c => `<div class="lab">${ZH[c]}</div>${cell(c.startsWith("INT") ? null : `pickup_${c}`)}${cell(`return_${c}`)}`).join("");
}
function pickFor(key) {
  const inp = document.createElement("input"); inp.type = "file"; inp.accept = "image/*";
  inp.onchange = () => { if (inp.files[0]) { ins.files[key] = inp.files[0]; renderSlots(); } };
  inp.click();
}
$("#ins-slots").addEventListener("click", e => {
  const del = e.target.closest("[data-del]");
  if (del) { delete ins.files[del.dataset.del]; renderSlots(); return; }
  const s = e.target.closest(".slot[data-key]"); if (s) pickFor(s.dataset.key);
});
$("#ins-slots").addEventListener("dragover", e => { if (e.target.closest(".slot[data-key]")) e.preventDefault(); });
$("#ins-slots").addEventListener("drop", e => {
  const s = e.target.closest(".slot[data-key]"); if (!s) return; e.preventDefault();
  const f = e.dataTransfer.files[0]; if (f) { ins.files[s.dataset.key] = f; renderSlots(); }
});
$("#ins-clear").addEventListener("click", () => { ins.files = {}; renderSlots(); $("#ins-result").innerHTML = ""; });
$("#ins-run").addEventListener("click", async () => {
  if (!Object.keys(ins.files).some(k => k.startsWith("return_"))) return toast("請至少上傳一張還車照");
  const btn = $("#ins-run"); btn.disabled = true; btn.innerHTML = `<span class="spin"></span> 判讀中`;
  const fd = new FormData();
  fd.append("order_no", $("#ins-order").value); fd.append("car_no", $("#ins-car").value);
  for (const [k, f] of Object.entries(ins.files)) fd.append(k, f, f.name);
  try {
    const c = await api("/api/inspect", { method: "POST", body: fd });
    const box = $("#ins-result"); box.innerHTML = ""; box.append(renderCase(c, false));
    box.scrollIntoView({ behavior: "smooth" });
  } catch (e) { toast("判讀失敗：" + e.message); }
  btn.disabled = false; btn.textContent = "開始判讀";
});

function gauge(p, th) {
  return `<div class="gauge" style="--p:${th.pass_max * 100}%;--a:${th.alert_min * 100}%">
    ${p == null ? "" : `<div class="mk" style="left:${p * 100}%"><span>${pct(p)}</span></div>`}</div>
    <div class="row small muted"><span>0 自動放行 ≤ ${pct(th.pass_max)}</span><div class="grow"></div><span>人工審核</span><div class="grow"></div><span>預警 ≥ ${pct(th.alert_min)} 100%</span></div>`;
}
function renderItem(it, th, caseId, reviewable) {
  const el = document.createElement("div"); el.className = `item ${it.route}`;
  const r = it.result || {};
  const kind = it.kind === "damage" ? "新車損" : "整潔度";
  let body = "";
  if (it.kind === "damage" && it.result) {
    body = `<div class="kv">判讀方式 <b>${METHOD[r.method] || r.method}</b>${r.samples ? `（問 ${r.samples} 次）` : ""}　新車損機率 <b>${pct(r.p_new)}</b>　信心 <b>${pct(r.confidence)}</b></div>
      ${(r.items || []).filter(d => d.type !== "other" || d.location).map(d => `<div class="small">• ${esc(d.type_zh)} ${esc(d.severity_zh)} ${esc(d.location)}</div>`).join("")}
      <div class="small muted">${esc(r.reason || "")}</div>`;
  } else if (it.kind === "cleanliness" && it.result && r.level) {
    body = `<div class="kv">判讀方式 <b>${METHOD[r.method] || r.method}</b>　判定 <b>${CLEAN[r.level]}</b>　信心 <b>${pct(r.confidence)}</b></div>
      ${bars(r.probs)}${r.issues?.length ? `<div class="small">髒污類型：${r.issues.map(i => ISSUE[i] || i).join("、")}</div>` : ""}
      ${r.description ? `<div class="small muted">${esc(r.description)}</div>` : ""}`;
  }
  el.innerHTML = `<div class="row gap"><b>${esc(it.panel_zh)}・${kind}</b><span class="rbadge ${it.route}">${ROUTE[it.route]}</span>${it.resolved ? `<span class="tag">已審核</span>` : ""}</div>
    <div class="pair"><figure><figcaption>取車</figcaption><div class="pk"></div></figure><figure><figcaption>還車</figcaption><div class="rt"></div></figure></div>
    ${gauge(it.p_abnormal, th)}${body}${it.note ? `<div class="small" style="color:var(--review)">⚠ ${esc(it.note)}</div>` : ""}
    ${(it.feedback || []).map(f => `<div class="fb">${f.source === "next_renter" ? "下一位用戶" : f.source === "ops" ? "審核人員" : f.source}：${esc(f.label)}${f.agree == null ? "" : f.agree ? "（與 AI 一致）" : "（與 AI 不一致，已回流）"}</div>`).join("")}`;
  const boxes = (r.items || []).map(d => ({ box: d.box, label: d.type_zh }));
  $(".pk", el).append(it.pickup_url ? boxed(it.pickup_url) : Object.assign(document.createElement("div"), { className: "none", textContent: it.kind === "damage" ? "未提供" : "不需要" }));
  $(".rt", el).append(it.return_url ? boxed(it.return_url, boxes) : Object.assign(document.createElement("div"), { className: "none", textContent: "未提供" }));
  if (reviewable) {
    const rb = document.createElement("div"); rb.className = "revbtns";
    const yes = it.kind === "damage" ? "確認有新車損" : "確認髒汙需清潔";
    const no = it.kind === "damage" ? "判定無新車損" : "判定不需清潔";
    rb.innerHTML = `<button class="btn" data-a="1">${yes}</button><button class="btn" data-a="0">${no}</button>` +
      (it.kind === "cleanliness" ? `<span class="small muted" style="align-self:center;margin-left:6px">模擬下一位用戶評價：</span>` +
        ["clean", "normal", "dirty"].map(l => `<button class="btn ghost" data-l="${l}">${CLEAN[l]}</button>`).join("") : "");
    rb.addEventListener("click", async e => {
      const b = e.target.closest("button"); if (!b) return;
      const payload = b.dataset.l ? { item_key: it.key, human_level: b.dataset.l, source: "next_renter" }
        : { item_key: it.key, human_abnormal: b.dataset.a === "1", source: "ops" };
      try { await jpost(`/api/cases/${caseId}/review`, payload); toast("已記錄回饋"); await loadOps(caseId); } catch (e2) { toast(e2.message); }
    });
    el.append(rb);
  }
  return el;
}
function renderCase(c, reviewable) {
  const wrap = document.createElement("div");
  wrap.innerHTML = `<div class="banner ${c.route}"><b>${ROUTE[c.route]}</b><span>訂單 ${esc(c.order_no)}${c.car_no ? `・${esc(c.car_no)}` : ""}・${esc(c.created.replace("T", " "))}</span></div>
    ${c.actions.length ? `<div class="card"><h3>系統自動處理</h3><ul class="actions">${c.actions.map(a => `<li>${esc(a)}</li>`).join("")}</ul></div>` : ""}
    <div class="items"></div>`;
  const box = $(".items", wrap);
  c.items.forEach(it => box.append(renderItem(it, c.thresholds, c.id, reviewable)));
  return wrap;
}

/* ---------------- 4. 營運儀表板 ---------------- */
const ops = { filter: "", sel: null };
async function loadOps(selectId) {
  const [cases, m] = await Promise.all([api("/api/cases"), api("/api/metrics")]);
  const cnt = r => cases.filter(c => c.route === r).length;
  $("#ops-kpis").innerHTML = `
    <div class="kpi alert"><small>預警案件</small><b>${cnt("alert")}</b></div>
    <div class="kpi review"><small>待人工審核</small><b>${cnt("review")}</b></div>
    <div class="kpi pass"><small>自動放行</small><b>${cnt("pass")}</b></div>
    <div class="kpi"><small>照片項目送人工比例</small><b>${pct(m.review_ratio)}</b></div>
    <div class="kpi ${m.level === 2 ? "alert" : m.level === 1 ? "review" : "pass"}"><small>監控等級</small><b>${m.level === 0 ? "正常" : "第 " + m.level + " 級"}</b></div>`;
  const shown = cases.filter(c => !ops.filter || c.route === ops.filter);
  $("#ops-list").innerHTML = shown.length ? shown.map(c => `<div class="case ${c.id === ops.sel ? "on" : ""}" data-id="${c.id}"><span class="dot ${c.route}"></span>
      <div class="grow"><b>${esc(c.order_no)}</b>${c.car_no ? `　${esc(c.car_no)}` : ""}<small>${esc(c.created.replace("T", " "))}・${c.items.length} 項・${ROUTE[c.route]}</small></div></div>`).join("")
    : `<p class="muted small">還沒有案件。到「還車判讀」送出一筆，就會出現在這裡。</p>`;
  if (selectId) ops.sel = selectId;
  if (ops.sel) {
    const c = cases.find(x => x.id === ops.sel);
    if (c) { const d = $("#ops-detail"); d.innerHTML = ""; d.append(renderCase(c, true)); }
  }
  renderMonitor(m);
}
function renderMonitor(m) {
  const b = m.thresholds_base, e = m.thresholds_effective;
  $("#th-pass").value = b.pass_max; $("#th-alert").value = b.alert_min;
  $("#th-pass-v").textContent = pct(b.pass_max); $("#th-alert-v").textContent = pct(b.alert_min);
  $("#th-eff").innerHTML = (e.pass_max !== b.pass_max || e.alert_min !== b.alert_min)
    ? `目前實際套用：放行 ≤ ${pct(e.pass_max)}、預警 ≥ ${pct(e.alert_min)}（監控第 1 級自動擴大人工審核區間）` : "";
  const msg = ["正常：AI 與回收答案的不一致率在容許範圍內", "第 1 級：已自動擴大人工審核區間，先保住準確度", "第 2 級：已通知工程師，附上累積錯誤案例，由人決定是否重新訓練"][m.level];
  const T = { damage: "新車損", cleanliness: "整潔度", panel: "拍攝面" };
  $("#mon").innerHTML = `<div class="lvl l${m.level}"><b>${msg}</b>${m.reasons.length ? `<div class="small">${m.reasons.map(esc).join("；")}</div>` : ""}</div>
    <table class="mt"><tr><th>判讀項目</th><th>回饋筆數</th><th>不一致率</th></tr>
    ${Object.entries(m.tasks).map(([k, t]) => `<tr><td>${T[k]}</td><td>${t.n}</td><td>${t.disagree == null ? "—" : pct(t.disagree)}${t.n < m.monitor.min_feedback ? ` <span class="muted">（未滿 ${m.monitor.min_feedback} 筆不判斷）</span>` : ""}</td></tr>`).join("")}</table>
    <p class="small muted">門檻：不一致率 ≥ ${pct(m.monitor.level1_disagree)} 進第 1 級、≥ ${pct(m.monitor.level2_disagree)} 進第 2 級；送人工比例 ≥ ${pct(m.monitor.level1_review_ratio)} 也進第 1 級。回饋來源：審核人員確認或駁回、下一位用戶評價。</p>`;
}
$("#ops-filter").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  ops.filter = b.dataset.r; $$("#ops-filter button").forEach(x => x.classList.toggle("on", x === b)); loadOps();
});
$("#ops-list").addEventListener("click", e => { const c = e.target.closest(".case"); if (c) { ops.sel = c.dataset.id; loadOps(); } });
$("#th-pass").addEventListener("input", e => ($("#th-pass-v").textContent = pct(+e.target.value)));
$("#th-alert").addEventListener("input", e => ($("#th-alert-v").textContent = pct(+e.target.value)));
$("#th-save").addEventListener("click", async () => {
  try { renderMonitor(await jpost("/api/thresholds", { pass_max: +$("#th-pass").value, alert_min: +$("#th-alert").value }, "PUT")); toast("門檻已更新，套用到之後的新案件"); }
  catch (e) { toast(e.message); }
});
$("#sim-go").addEventListener("click", async () => {
  try { await jpost("/api/simulate-feedback", { task: $("#sim-task").value, n: +$("#sim-n").value, disagree_rate: +$("#sim-rate").value }); await loadOps(); toast("已灌入模擬回饋"); }
  catch (e) { toast(e.message); }
});
$("#reset").addEventListener("click", async () => {
  if (!confirm("確定清空所有案件、回饋與門檻設定？")) return;
  await jpost("/api/reset", {}); ops.sel = null; $("#ops-detail").innerHTML = `<p class="muted">點選左側案件查看前後對照與 AI 判讀</p>`; loadOps();
});

/* ---------------- 啟動 ---------------- */
loadStatus().catch(e => toast("無法連線後端：" + e.message));
renderCapture(); renderSlots(); show();
