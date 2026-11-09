const state = {
  sessionId: crypto.randomUUID(),
  socket: null,
  activeAssistant: null,
  pending: false,
  traces: [],
};

const el = (id) => document.getElementById(id);
const conversation = el("conversation");
const composer = el("composer");
const input = el("query");

const traceLabels = {
  intent_classification: "意图识别",
  route_decision: "动态路由",
  query_optimization: "Query 优化",
  hybrid_retrieval: "Dense / Sparse 混合召回",
  rerank: "二阶段精排",
  grounded_generation: "证据约束生成",
};

const rewriteBackendLabels = {
  llm: "LLM 改写",
  rules: "规则改写",
  "rules(llm-fallback)": "LLM 失败 → 规则兜底",
};

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  state.socket = new WebSocket(`${protocol}://${location.host}/ws/chat/${state.sessionId}`);
  state.socket.addEventListener("open", () => {
    document.querySelector(".connection").classList.add("connected");
    el("connection-text").textContent = "实时链路已连接";
  });
  state.socket.addEventListener("close", () => {
    document.querySelector(".connection").classList.remove("connected");
    el("connection-text").textContent = "正在重连";
    if (!state.pending) setTimeout(connect, 1200);
  });
  state.socket.addEventListener("message", (event) => handleEvent(JSON.parse(event.data)));
}

async function loadHealth() {
  try {
    const response = await fetch("/api/v1/health");
    const data = await response.json();
    el("service-state").textContent = data.status === "ok" ? "正常" : "降级";
    el("retrieval-mode").textContent = `${data.services.retrieval} / ${data.services.rerank}`;
    el("llm-mode").textContent = data.services.llm;
  } catch {
    el("service-state").textContent = "不可用";
  }
}

function addMessage(role, content = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `
    <div class="message-head"><strong>${role === "user" ? "现场描述" : "FiberMind"}</strong><span>${new Date().toLocaleTimeString("zh-CN", {hour: "2-digit", minute: "2-digit"})}</span></div>
    <div class="message-body"></div>`;
  article.querySelector(".message-body").textContent = content;
  conversation.appendChild(article);
  conversation.scrollTop = conversation.scrollHeight;
  return article;
}

function submitQuery(query) {
  const value = query.trim();
  if (!value || state.pending || state.socket?.readyState !== WebSocket.OPEN) return;
  el("welcome")?.remove();
  addMessage("user", value);
  state.activeAssistant = addMessage("assistant", "");
  state.pending = true;
  state.traces = [];
  el("send").disabled = true;
  el("request-state").textContent = "正在分析";
  el("session-short").textContent = value.slice(0, 18);
  resetInspector();
  state.socket.send(JSON.stringify({ query: value }));
  input.value = "";
}

function handleEvent(message) {
  const { event, data } = message;
  if (event === "intent_completed") {
    el("intent-name").textContent = data.intent;
    el("intent-confidence").textContent = `${Math.round(data.confidence * 100)}%`;
    addLiveTrace("intent_classification", data);
  } else if (event === "route_completed") {
    el("route-name").textContent = data.route;
    addLiveTrace("route_decision", data);
  } else if (event === "query_optimized") {
    addLiveTrace("query_optimization", data);
    renderRewrite(data);
  } else if (event === "retrieval_completed") {
    addLiveTrace("hybrid_retrieval", data);
  } else if (event === "rerank_completed") {
    addLiveTrace("rerank", data);
  } else if (event === "generation_completed") {
    el("grounded-state").textContent = data.grounded ? "已约束" : "证据不足";
    addLiveTrace("grounded_generation", data);
  } else if (event === "answer_delta") {
    state.activeAssistant.querySelector(".message-body").textContent += data.text;
    conversation.scrollTop = conversation.scrollHeight;
  } else if (event === "answer_completed") {
    completeRequest(data);
  } else if (event === "error") {
    state.activeAssistant.querySelector(".message-body").textContent = `请求失败：${data.message}`;
    finishPending();
  }
}

function addLiveTrace(name, detail) {
  if (state.traces.some((item) => item.name === name)) return;
  state.traces.push({ name, detail, latency_ms: null });
  renderTraces(state.traces);
}

function completeRequest(response) {
  state.activeAssistant.querySelector(".message-body").textContent = response.answer;
  const meta = document.createElement("div");
  meta.className = "answer-meta";
  meta.innerHTML = `<span class="badge success">${response.diagnosis.grounded ? "知识库有据" : "需要补充"}</span><span class="badge">${response.route}</span><span>${response.citations.length} 条引用</span>`;
  state.activeAssistant.appendChild(meta);

  el("intent-name").textContent = response.intent.name;
  el("intent-confidence").textContent = `${Math.round(response.intent.confidence * 100)}%`;
  el("route-name").textContent = response.route;
  el("grounded-state").textContent = response.diagnosis.grounded ? "已约束" : "证据不足";
  el("total-latency").textContent = `${response.total_latency_ms.toFixed(1)} ms`;
  renderTraces(response.trace);
  renderSources(response.citations);
  const rewriteStep = (response.trace || []).find((step) => step.name === "query_optimization");
  if (rewriteStep) renderRewrite(rewriteStep.detail);
  finishPending();
}

function finishPending() {
  state.pending = false;
  el("send").disabled = false;
  el("request-state").textContent = "分析完成";
  input.focus();
}

function resetInspector() {
  ["intent-name", "intent-confidence", "route-name", "grounded-state"].forEach((id) => el(id).textContent = "--");
  el("total-latency").textContent = "-- ms";
  el("source-count").textContent = "0";
  el("source-list").innerHTML = '<p class="empty-state">正在检索知识来源</p>';
  el("trace-list").innerHTML = '<li class="empty-state">链路处理中…</li>';
  el("rewrite-list").innerHTML = '<p class="empty-state">RAG 路径会在此显示改写后的检索 Query</p>';
}

function renderRewrite(detail) {
  const data = detail || {};
  const strategies = (data.strategies || []).map((s) => `<span class="badge">${escapeHtml(s)}</span>`).join("");
  const subQueries = (data.sub_queries || []).length
    ? (data.sub_queries || []).map((q, i) => `<li>${escapeHtml(q)}</li>`).join("")
    : '<li class="empty-state">单一问题，未拆分子查询</li>';
  el("rewrite-list").innerHTML = `
    <article class="rewrite-item">
      <h3>改写引擎 <span class="badge success">${escapeHtml(rewriteBackendLabels[data.backend] || data.backend || "--")}</span></h3>
      <p class="rewrite-line"><strong>原始：</strong>${escapeHtml(data.original || "--")}</p>
      <p class="rewrite-line"><strong>改写后（检索用）：</strong>${escapeHtml(data.rewritten || "--")}</p>
      ${data.context_expanded ? '<p class="rewrite-note">已结合多轮上下文消解指代</p>' : ""}
      <p class="rewrite-line"><strong>子查询：</strong></p>
      <ul>${subQueries}</ul>
      <p class="rewrite-line"><strong>HyDE 假设文档：</strong>${escapeHtml(data.hyde || "未启用")}</p>
      ${data.fallback_reason ? `<p class="rewrite-note">LLM 改写失败原因：${escapeHtml(data.fallback_reason)}</p>` : ""}
      <div class="rewrite-strategies">${strategies || '<span class="badge">无改写策略</span>'}</div>
    </article>`;
}

function formatDetail(step) {
  const detail = step.detail || {};
  if (step.name === "intent_classification") return `intent=${detail.intent} · entities=${JSON.stringify(detail.entities || {})}`;
  if (step.name === "route_decision") return detail.reason || detail.route;
  if (step.name === "query_optimization") {
    const backend = rewriteBackendLabels[detail.backend] || detail.backend || "";
    return `${backend} · ${(detail.strategies || []).join(" + ") || "原始 Query"} · ${detail.rewritten || ""}`;
  }
  if (step.name === "hybrid_retrieval") return `召回 ${detail.candidates || 0} 个候选 · ${(detail.top || []).slice(0, 2).map((x) => x.title).join(" / ")}`;
  if (step.name === "rerank") return `保留 ${detail.selected || 0} 条 · ${(detail.top || []).slice(0, 2).map((x) => `${x.title} (${x.score})`).join(" / ")}`;
  if (step.name === "grounded_generation") return `grounded=${detail.grounded} · confidence=${detail.confidence}`;
  return JSON.stringify(detail);
}

function renderTraces(traces) {
  el("trace-list").innerHTML = traces.map((step) => `
    <li class="trace-step">
      <div class="trace-title"><span>${traceLabels[step.name] || step.name}</span><time>${step.latency_ms == null ? "处理中" : `${step.latency_ms.toFixed(1)} ms`}</time></div>
      <p class="trace-detail">${escapeHtml(formatDetail(step))}</p>
    </li>`).join("");
}

function renderSources(sources) {
  el("source-count").textContent = String(sources.length);
  el("source-list").innerHTML = sources.length ? sources.map((source) => `
    <article class="source-item">
      <h3>[${source.index}] ${escapeHtml(source.title)}</h3>
      <p>${escapeHtml(source.source)}</p>
      <p>${escapeHtml(source.excerpt)}</p>
      <div class="source-score">
        <span>Dense ${source.dense_score.toFixed(3)}</span><span>Sparse ${source.sparse_score.toFixed(3)}</span>
        <span>Fusion ${source.fusion_score.toFixed(3)}</span><span>Rerank ${source.rerank_score.toFixed(3)}</span>
      </div>
    </article>`).join("") : '<p class="empty-state">该回答未使用检索来源</p>';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"})[char]);
}

composer.addEventListener("submit", (event) => { event.preventDefault(); submitQuery(input.value); });
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submitQuery(input.value); }
});
document.querySelectorAll("[data-query]").forEach((button) => button.addEventListener("click", () => submitQuery(button.dataset.query)));
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab, .panel").forEach((node) => node.classList.remove("active"));
  tab.classList.add("active");
  el(`${tab.dataset.tab}-panel`).classList.add("active");
}));
el("new-session").addEventListener("click", () => location.reload());

loadHealth();
connect();
