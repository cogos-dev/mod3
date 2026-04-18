/**
 * TracePanel — renders `trace_event` WebSocket frames into the Cycle Trace drawer.
 *
 * Event shape (ADR-083):
 *   { id, ts (RFC3339), source, cycle_id, kind, payload }
 *
 * Renders most-recent-first, caps at MAX_ENTRIES, exposes clear().
 */
(function () {
  const MAX_ENTRIES = 100;

  const KIND_META = {
    state_transition: { label: "state", color: "var(--accent)" },
    tool_dispatch:    { label: "tool",  color: "var(--orange)" },
    assessment:       { label: "asmt",  color: "var(--green)" },
  };

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
  }

  function fmtTime(ts) {
    try {
      const d = new Date(ts);
      if (isNaN(d.getTime())) return "--:--:--";
      const hh = String(d.getHours()).padStart(2, "0");
      const mm = String(d.getMinutes()).padStart(2, "0");
      const ss = String(d.getSeconds()).padStart(2, "0");
      return `${hh}:${mm}:${ss}`;
    } catch (_e) {
      return "--:--:--";
    }
  }

  function summarize(kind, payload) {
    const p = payload || {};
    try {
      if (kind === "state_transition") {
        const from = p.from != null ? p.from : "?";
        const to = p.to != null ? p.to : "?";
        return `${from} \u2192 ${to}`;
      }
      if (kind === "tool_dispatch") {
        const tool = p.tool || p.name || "tool";
        const ms = p.duration_ms != null ? p.duration_ms
                  : p.latency_ms != null ? p.latency_ms
                  : null;
        return ms != null ? `${tool} (${Math.round(ms)}ms)` : `${tool}`;
      }
      if (kind === "assessment") {
        const action = p.action || p.decision || "assess";
        const conf = p.confidence != null ? p.confidence
                    : p.conf != null ? p.conf
                    : null;
        return conf != null
          ? `${action} (conf: ${Number(conf).toFixed(2)})`
          : `${action}`;
      }
      // Unknown kind — show a compact inline JSON preview if short
      const keys = Object.keys(p);
      if (keys.length === 0) return "";
      const first = keys[0];
      const val = p[first];
      const valStr = typeof val === "object" ? "{...}" : String(val);
      return `${first}=${valStr}${keys.length > 1 ? ` +${keys.length - 1}` : ""}`;
    } catch (_e) {
      return "";
    }
  }

  class TracePanel {
    constructor(container) {
      this.container = container;
      this.count = 0;
    }

    render(event) {
      if (!event || !this.container) return;
      const kind = event.kind || "unknown";
      const meta = KIND_META[kind] || { label: kind, color: "var(--muted)" };

      const row = document.createElement("div");
      row.className = "trace-entry";

      const time = fmtTime(event.ts);
      const source = event.source || "?";
      const summary = summarize(kind, event.payload);

      let fullJson = "";
      try { fullJson = JSON.stringify(event, null, 2); } catch (_e) { fullJson = "<unserializable>"; }

      row.innerHTML =
        `<span class="trace-time">${escapeHtml(time)}</span>` +
        `<span class="trace-source">${escapeHtml(source)}</span>` +
        `<span class="trace-kind" style="color:${meta.color};border-color:${meta.color};">` +
          `${escapeHtml(meta.label)}</span>` +
        `<span class="trace-summary">${escapeHtml(summary)}</span>`;
      row.title = fullJson;

      // Most-recent-first
      if (this.container.firstChild) {
        this.container.insertBefore(row, this.container.firstChild);
      } else {
        this.container.appendChild(row);
      }
      this.count += 1;

      // Rolling window: drop oldest (last child)
      while (this.count > MAX_ENTRIES && this.container.lastChild) {
        this.container.removeChild(this.container.lastChild);
        this.count -= 1;
      }
    }

    clear() {
      if (!this.container) return;
      while (this.container.firstChild) {
        this.container.removeChild(this.container.firstChild);
      }
      this.count = 0;
    }
  }

  // --- Singleton wiring on DOMContentLoaded ---
  function init() {
    const container = document.getElementById("trace-entries");
    const panel = document.getElementById("trace-panel");
    const toggle = document.getElementById("trace-toggle");
    const toggleIcon = document.getElementById("trace-toggle-icon");
    if (!container || !panel) return;

    window.tracePanel = new TracePanel(container);

    // Persist collapse state
    const LS_KEY = "mod3.tracePanel.collapsed";
    function applyCollapsed(collapsed) {
      if (collapsed) panel.classList.add("collapsed");
      else panel.classList.remove("collapsed");
      if (toggleIcon) toggleIcon.innerHTML = collapsed ? "&#9650;" : "&#9660;";
    }
    let stored = null;
    try { stored = localStorage.getItem(LS_KEY); } catch (_e) {}
    // Default collapsed on first load
    applyCollapsed(stored === null ? true : stored === "1");

    if (toggle) {
      toggle.addEventListener("click", () => {
        const nowCollapsed = !panel.classList.contains("collapsed");
        applyCollapsed(nowCollapsed);
        try { localStorage.setItem(LS_KEY, nowCollapsed ? "1" : "0"); } catch (_e) {}
      });
    }

    // Dev hook for smoke-testing from the browser console.
    window.testTrace = function (partial) {
      const ev = Object.assign(
        {
          id: "test-" + Math.random().toString(36).slice(2, 8),
          ts: new Date().toISOString(),
          source: "cog",
          cycle_id: "test-cycle",
          kind: "state_transition",
          payload: { from: "idle", to: "active" },
        },
        partial || {}
      );
      window.tracePanel.render(ev);
      return ev;
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
