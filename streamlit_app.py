"""
DataLens AI — Streamlit Console
================================
A visual, step-by-step replacement for the Flask HTML pages.
This runs on :8501 alongside:
  • Flask  :5055  (JSON API — used by the React dashboard)
  • Vite   :5173  (React dashboard)

Start everything with:   ./run.sh
Start Streamlit only:    ./venv/bin/streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import math
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens AI — Console",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — mirrors the React dashboard editorial aesthetic ──────────────
st.markdown(
    """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* ── Background mesh ────────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(at 0% 0%,   hsla(217,100%,94%,0.7) 0px, transparent 50%),
        radial-gradient(at 100% 0%, hsla(213,100%,90%,0.8) 0px, transparent 50%),
        radial-gradient(at 100%100%,hsla(220,100%,95%,0.7) 0px, transparent 50%),
        radial-gradient(at 0% 100%,hsla(215,100%,92%,0.6) 0px, transparent 50%)
        #f5f8ff !important;
}

/* ── Cards / glass panels ───────────────────────────────────────── */
.dl-card {
    background: rgba(255,255,255,0.62);
    backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.84);
    border-radius: 1.5rem;
    box-shadow: 0 16px 45px rgba(37,99,235,0.08);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
}

/* ── Metric tiles ───────────────────────────────────────────────── */
.dl-metric {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(219,234,254,0.9);
    border-radius: 1rem;
    padding: 1rem;
    text-align: center;
}
.dl-metric .label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.28em; text-transform: uppercase;
    color: #64748b; margin-bottom: 0.25rem;
}
.dl-metric .value {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg,#1e3a8a,#2563eb,#3b82f6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.dl-metric .sub {
    font-size: 10px; color: #94a3b8; font-weight: 600;
}

/* ── Step timeline ──────────────────────────────────────────────── */
.step-done  { color:#16a34a; font-weight:700; }
.step-run   { color:#2563eb; font-weight:700; }
.step-wait  { color:#94a3b8; }
.step-fail  { color:#dc2626; font-weight:700; }

/* ── Badge chips ────────────────────────────────────────────────── */
.chip {
    display:inline-block; padding:2px 10px;
    border-radius:999px; font-size:11px; font-weight:600;
    border:1px solid rgba(191,219,254,0.9);
    background:rgba(255,255,255,0.72); color:#1d4ed8;
    margin:2px;
}
.chip-high   { background:#fef2f2; color:#dc2626; border-color:#fecaca; }
.chip-medium { background:#fffbeb; color:#d97706; border-color:#fde68a; }
.chip-low    { background:#f0f9ff; color:#2563eb; border-color:#bfdbfe; }
.chip-good   { background:#f0fdf4; color:#16a34a; border-color:#bbf7d0; }

/* ── Gradient text ──────────────────────────────────────────────── */
.grad { background:linear-gradient(135deg,#1e3a8a,#2563eb,#3b82f6,#60a5fa);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip:text; font-weight:900; }

/* ── Streamlit widget overrides ─────────────────────────────────── */
.stButton > button {
    background: linear-gradient(to right, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.75rem !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(to right, #1e40af, #1d4ed8) !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(37,99,235,0.25) !important;
}
.stFileUploader > div {
    border-radius: 1rem !important;
    border: 2px dashed #bfdbfe !important;
    background: rgba(255,255,255,0.5) !important;
}
.stSelectbox > div > div,
.stTextInput > div > div > input {
    border-radius: 0.75rem !important;
    border-color: #e2e8f0 !important;
}
/* Progress bar teal accent */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2563eb, #38bdf8) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_n(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "—"
        if v.is_integer():
            return f"{int(v):,}"
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    return str(v)


def severity_cls(s: str) -> str:
    s = (s or "").lower()
    if s in {"high", "critical"}:
        return "chip-high"
    if s in {"medium", "moderate"}:
        return "chip-medium"
    if s in {"low"}:
        return "chip-low"
    return "chip"


def save_upload(uploaded) -> tuple[str, Path, str]:
    """Save a Streamlit UploadedFile to disk and return (dataset_id, path, name)."""
    dataset_id = uuid4().hex[:12]
    suffix = Path(uploaded.name).suffix.lower() or ".csv"
    Path("uploads").mkdir(exist_ok=True)
    dest = Path("uploads") / f"{dataset_id}{suffix}"
    dest.write_bytes(uploaded.read())
    return dataset_id, dest, uploaded.name


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            """
<div style="text-align:center;padding:0.5rem 0 1rem">
  <div style="font-size:1.6rem;font-weight:900;letter-spacing:-0.03em">
    <span class="grad">DataLens AI</span>
  </div>
  <div style="font-size:10px;letter-spacing:0.28em;text-transform:uppercase;color:#64748b;margin-top:2px">
    Visual Console
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Upload ────────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:10px;font-weight:700;letter-spacing:0.28em;text-transform:uppercase;color:#64748b">01 · Dataset</p>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Drop a dataset",
            type=["csv", "tsv", "xlsx", "xls", "json"],
            label_visibility="collapsed",
        )
        if uploaded:
            sz = uploaded.size / 1024
            unit = "KB" if sz < 1024 else "MB"
            sz_str = f"{sz:.1f} {unit}" if sz < 1024 else f"{sz/1024:.2f} {unit}"
            st.caption(f"📄 **{uploaded.name}** · {sz_str}")

        st.markdown("---")

        # ── Analysis depth ────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:10px;font-weight:700;letter-spacing:0.28em;text-transform:uppercase;color:#64748b">02 · Analysis depth</p>',
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "Mode",
            options=["quick", "standard", "deep", "research"],
            index=1,
            horizontal=True,
            label_visibility="collapsed",
        )
        mode_desc = {
            "quick": "Profiling only — sub-second.",
            "standard": "Full pipeline including charts, ML, SHAP.",
            "deep": "Adds bootstrap CIs, expanded bivariate tests.",
            "research": "All analyses at maximum budget.",
        }
        st.caption(mode_desc[mode])

        st.markdown("---")

        # ── Target column ─────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:10px;font-weight:700;letter-spacing:0.28em;text-transform:uppercase;color:#64748b">03 · Target column (optional)</p>',
            unsafe_allow_html=True,
        )
        target = st.text_input(
            "Target column",
            placeholder="e.g. churned, price, is_fraud",
            label_visibility="collapsed",
        )
        st.caption("Unlocks ML leaderboard, SHAP explanations, and target-aware charts.")

        st.markdown("---")

        run = st.button("▶  Run analysis", use_container_width=True)

        st.markdown("---")

        # ── Links ─────────────────────────────────────────────────────────
        st.markdown(
            """
<div style="font-size:11px;color:#64748b;line-height:2">
  🔬 <a href="http://127.0.0.1:5173" target="_blank" style="color:#2563eb">React dashboard →</a><br>
  🔧 <a href="http://127.0.0.1:5055" target="_blank" style="color:#2563eb">Flask API →</a><br>
  💊 <a href="http://127.0.0.1:5055/api/health" target="_blank" style="color:#2563eb">Health check →</a>
</div>
""",
            unsafe_allow_html=True,
        )

    return {"uploaded": uploaded, "mode": mode, "target": target or None, "run": run}


# ── Main header ───────────────────────────────────────────────────────────────

def render_header():
    st.markdown(
        """
<div class="dl-card">
  <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;justify-content:space-between">
    <div>
      <div style="font-size:10px;font-weight:700;letter-spacing:0.32em;text-transform:uppercase;color:#64748b">
        Flask backend console
      </div>
      <div style="font-size:1.75rem;font-weight:900;letter-spacing:-0.03em;margin-top:2px">
        DataLens AI <span class="grad">v3</span> live console
      </div>
      <div style="font-size:0.85rem;color:#64748b;margin-top:4px">
        Upload a dataset and watch every pipeline phase execute step by step.
      </div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:6px">
      <span class="chip">Profiling</span>
      <span class="chip">Deep stats</span>
      <span class="chip">Anomaly ensemble</span>
      <span class="chip">ML leaderboard</span>
      <span class="chip">SHAP</span>
      <span class="chip">Time-series</span>
      <span class="chip">NLP</span>
      <span class="chip">Drift</span>
      <span class="chip">Agent Q&amp;A</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── Step-by-step pipeline runner ──────────────────────────────────────────────

PIPELINE_STEPS = [
    ("📂", "Load & save",         "Saving upload to disk and loading into pandas."),
    ("🔍", "Profiling",           "Building schema, dtypes, missing counts, correlations."),
    ("🧬", "Column roles",        "Inferring semantic roles (id, price, target candidate…)."),
    ("📡", "Dataset signals",     "Detecting structural characteristics of the dataset."),
    ("🎯", "Analysis selection",  "Choosing which analyses are worth running for this mode."),
    ("❤️", "Health score",        "Computing completeness, consistency, uniqueness, outlier safety."),
    ("🤖", "ML readiness",        "Estimating how ready the data is for supervised learning."),
    ("📊", "Advanced indicators", "Column utility, IQR anomalies, fairness flags, freshness."),
    ("⚡", "Signal engine",       "Converting indicators into human-readable signals."),
    ("🔧", "Cleaning",            "Running conservative auto-cleaning and saving cleaned CSV."),
    ("📈", "Deep statistics",     "Bivariate tests, normality, bootstrap CIs, distribution fit."),
    ("🚨", "Anomaly ensemble",    "Isolation Forest + LOF + MAD z-score + Mahalanobis."),
    ("🏆", "Model leaderboard",   "Cross-validated CV of 7 estimators for the target column."),
    ("💡", "SHAP explainability", "TreeSHAP / LinearSHAP global + per-row attributions."),
    ("⏱️", "Time-series",         "ADF + KPSS stationarity, STL decomposition, Holt-Winters."),
    ("📝", "Text profile",        "Vocabulary, n-grams, TTR, stopword-filtered bigrams."),
    ("📉", "Charts",              "Role-aware chart planner → 20+ matplotlib PNG charts."),
    ("📄", "PDF report",          "Generating the multi-page styled report with embedded charts."),
]


def run_pipeline(uploaded, mode: str, target: str | None) -> dict | None:
    from framevitals.loader import (
    load_dataset,
)
from framevitals.profiler import (
    build_profile,
)
from framevitals.column_roles import (
    infer_column_roles,
    summarize_roles,
)
from framevitals.dataset_signals import (
    detect_dataset_signals,
)
from framevitals.analysis_selector import (
    select_analyses,
)
from framevitals.health_score import (
    calculate_health_score,
)
from framevitals.ml_readiness import (
    calculate_ml_readiness,
)
from framevitals.advanced_indicators import (
    calculate_advanced_indicators,
)
from framevitals.signal_engine import (
    build_signals,
)
from framevitals.cleaner import (
    create_cleaned_dataset,
)
from framevitals.deep_statistics_v2 import (
    run_deep_statistics_v2,
)
from framevitals.anomaly_ensemble import (
    detect_anomalies_ensemble,
)
from framevitals.model_leaderboard import (
    run_model_leaderboard,
)
from framevitals.explainability import (
    explain_winner,
)
from framevitals.time_series import (
    detect_and_analyze_time_series,
)
from framevitals.text_profile import (
    profile_text_columns,
)
from framevitals.visualizer import (
    generate_charts,
)
from modules.ai_agent import (
    answer_with_agent,
)
    from modules.report_generator import generate_pdf_report
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    n_steps = len(PIPELINE_STEPS)
    progress_bar = st.progress(0, text="Initialising…")
    status_area = st.empty()

    def update(step_idx: int, icon: str, name: str, done: bool = False, err: bool = False):
        frac = (step_idx + (1 if done else 0)) / n_steps
        tone = "✅" if done else ("❌" if err else "⏳")
        progress_bar.progress(
            min(frac, 1.0),
            text=f"{tone} {name}",
        )

    # ── Timeline HTML builder ─────────────────────────────────────────────
    step_states: list[str] = ["wait"] * n_steps  # wait | run | done | fail

    def render_timeline():
        icons = {"done": "✅", "run": "⏳", "wait": "○", "fail": "❌"}
        cls = {"done": "step-done", "run": "step-run", "wait": "step-wait", "fail": "step-fail"}
        rows = "".join(
            f'<div style="margin:2px 0"><span class="{cls[step_states[i]]}">'
            f'{icons[step_states[i]]} {PIPELINE_STEPS[i][1]}</span>'
            f'<span style="font-size:11px;color:#94a3b8;margin-left:6px">'
            f'{PIPELINE_STEPS[i][2]}</span></div>'
            for i in range(n_steps)
        )
        status_area.markdown(
            f'<div class="dl-card" style="padding:1rem 1.25rem;font-size:13px">'
            f'{rows}</div>',
            unsafe_allow_html=True,
        )

    def mark(i: int, state: str):
        step_states[i] = state
        render_timeline()

    # ── Step 0: Load ──────────────────────────────────────────────────────
    mark(0, "run")
    try:
        dataset_id, file_path, original_filename = save_upload(uploaded)
        df = load_dataset(file_path)
        mark(0, "done")
    except Exception as exc:
        mark(0, "fail")
        st.error(f"❌ Load failed: {exc}")
        return None

    # ── Step 1: Profile ───────────────────────────────────────────────────
    mark(1, "run")
    profile = build_profile(df)
    mark(1, "done")

    # ── Step 2: Column roles ──────────────────────────────────────────────
    mark(2, "run")
    column_roles = infer_column_roles(df)
    roles_summary = summarize_roles(column_roles)
    mark(2, "done")

    # ── Step 3: Dataset signals ───────────────────────────────────────────
    mark(3, "run")
    dataset_signals = detect_dataset_signals(df, profile)
    mark(3, "done")

    # ── Step 4: Analysis selection ────────────────────────────────────────
    mark(4, "run")
    analysis_selection = select_analyses(
        signals=dataset_signals,
        analysis_mode=mode,
        target_column=target,
    )
    mark(4, "done")

    # ── Step 5: Health ────────────────────────────────────────────────────
    mark(5, "run")
    health = calculate_health_score(df, profile)
    mark(5, "done")

    # ── Step 6: ML readiness ──────────────────────────────────────────────
    mark(6, "run")
    ml_readiness = calculate_ml_readiness(df)
    mark(6, "done")

    # ── Step 7: Advanced indicators ───────────────────────────────────────
    mark(7, "run")
    advanced = calculate_advanced_indicators(df)
    mark(7, "done")

    # ── Step 8: Signal engine ─────────────────────────────────────────────
    mark(8, "run")
    signals = build_signals(profile, health, ml_readiness, advanced)
    mark(8, "done")

    # ── Step 9: Cleaning ──────────────────────────────────────────────────
    mark(9, "run")
    cleaning = create_cleaned_dataset(dataset_id, df)
    mark(9, "done")

    # ── Phase 3: parallel (steps 10-15) ──────────────────────────────────
    deep_statistics_v2 = None
    anomalies_v2 = None
    time_series_analysis = None
    text_profile_result = None

    if mode in {"standard", "deep", "research"}:
        # Mark all four as running simultaneously
        for i in [10, 11, 14, 15]:
            step_states[i] = "run"
        render_timeline()

        tasks = {
            "deep_statistics_v2": lambda: run_deep_statistics_v2(df),
            "anomalies_v2":       lambda: detect_anomalies_ensemble(df),
            "time_series":        lambda: detect_and_analyze_time_series(df, target_column=target),
            "text_profile":       lambda: profile_text_columns(df),
        }
        parallel_results: dict[str, object] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    parallel_results[name] = fut.result()
                except Exception as exc:
                    parallel_results[name] = {"available": False, "error": str(exc)}

        deep_statistics_v2 = parallel_results.get("deep_statistics_v2")
        anomalies_v2       = parallel_results.get("anomalies_v2")
        time_series_analysis = parallel_results.get("time_series")
        text_profile_result  = parallel_results.get("text_profile")

        for i in [10, 11, 14, 15]:
            step_states[i] = "done"
        render_timeline()
    else:
        for i in [10, 11, 14, 15]:
            step_states[i] = "done"  # skipped → show done so UI is clean

    # ── Steps 12/13: ML leaderboard + SHAP ───────────────────────────────
    model_leaderboard = None
    explainability = None

    if target and mode in {"standard", "deep", "research"}:
        mark(12, "run")
        try:
            model_leaderboard = run_model_leaderboard(df, target_column=target)
        except Exception as exc:
            model_leaderboard = {"available": False, "error": str(exc)}
        mark(12, "done")

        if (
            isinstance(model_leaderboard, dict)
            and model_leaderboard.get("available")
            and model_leaderboard.get("winner")
        ):
            mark(13, "run")
            try:
                explainability = explain_winner(
                    df,
                    target_column=target,
                    leaderboard_result=model_leaderboard,
                    dataset_id=dataset_id,
                )
            except Exception as exc:
                explainability = {"available": False, "error": str(exc)}
            mark(13, "done")
        else:
            step_states[13] = "done"
            render_timeline()
    else:
        step_states[12] = "done"
        step_states[13] = "done"
        render_timeline()

    # ── Step 16: Charts ───────────────────────────────────────────────────
    charts: list[dict] = []
    if mode in {"standard", "deep", "research"}:
        mark(16, "run")
        try:
            charts = generate_charts(
                dataset_id, df, health, advanced, cleaning,
                target_column=target,
                model_leaderboard=model_leaderboard,
                explainability=explainability,
                time_series=time_series_analysis,
                deep_statistics_v2=deep_statistics_v2,
            )
        except Exception as exc:
            st.warning(f"Chart generation warning: {exc}")
        mark(16, "done")
    else:
        step_states[16] = "done"
        render_timeline()

    # ── Step 17: PDF ──────────────────────────────────────────────────────
    mark(17, "run")
    ai_report = {"source": "deferred", "text": "", "deferred": True}
    result = {
        "dataset_id": dataset_id,
        "filename": original_filename,
        "analysis_mode": mode,
        "profile": profile,
        "column_roles": column_roles,
        "roles_summary": roles_summary,
        "dataset_signals": dataset_signals,
        "analysis_selection": analysis_selection,
        "health": health,
        "signals": signals,
        "ml_readiness": ml_readiness,
        "advanced": advanced,
        "deep_statistics_v2": deep_statistics_v2,
        "anomalies_v2": anomalies_v2,
        "model_leaderboard": model_leaderboard,
        "explainability": explainability,
        "time_series": time_series_analysis,
        "text_profile": text_profile_result,
        "cleaning": cleaning,
        "charts": charts,
        "ai_report": ai_report,
        "timings_ms": {},
    }
    pdf_path: Path | None = None
    try:
        pdf_path = generate_pdf_report(result)
    except Exception as exc:
        st.warning(f"PDF generation warning: {exc}")
    mark(17, "done")

    progress_bar.progress(1.0, text="✅ Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_area.empty()

    result["_pdf_path"] = pdf_path
    result["_df"] = df
    return result


# ── Result renderer ───────────────────────────────────────────────────────────

def render_results(result: dict):
    profile = result.get("profile", {})
    shape = profile.get("shape", {})
    health = result.get("health", {})
    ml = result.get("ml_readiness", {})
    signals = result.get("signals", []) or []
    cleaning = result.get("cleaning", {}) or {}
    charts = result.get("charts", []) or []
    pdf_path = result.get("_pdf_path")
    df: pd.DataFrame | None = result.get("_df")

    # ── KPI tiles ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    tiles = [
        (c1, "Rows",         fmt_n(shape.get("rows", 0)),   ""),
        (c2, "Columns",      fmt_n(shape.get("columns", 0)), ""),
        (c3, "Health",       fmt_n(health.get("overall_score")), health.get("label", "")),
        (c4, "ML Readiness", fmt_n(ml.get("score")),         ml.get("label", "")),
        (c5, "Charts",       fmt_n(len(charts)),              "Generated"),
    ]
    for col, label, val, sub in tiles:
        with col:
            st.markdown(
                f'<div class="dl-metric"><div class="label">{label}</div>'
                f'<div class="value">{val}</div>'
                f'<div class="sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download buttons ─────────────────────────────────────────────────
    dl_col1, dl_col2, _ = st.columns([1, 1, 4])
    cleaned_path = Path("cleaned") / f"{result['dataset_id']}_cleaned.csv"
    if cleaned_path.exists():
        with dl_col1:
            st.download_button(
                "⬇ Cleaned CSV",
                data=cleaned_path.read_bytes(),
                file_name=f"{result['dataset_id']}_cleaned.csv",
                mime="text/csv",
                use_container_width=True,
            )
    if pdf_path and Path(pdf_path).exists():
        with dl_col2:
            st.download_button(
                "⬇ PDF Report",
                data=Path(pdf_path).read_bytes(),
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                use_container_width=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview",
        "🧬 Column Roles",
        "⚡ Signals",
        "📈 Charts",
        "🔧 Cleaning",
        "🏆 ML Lab",
        "📋 Data Preview",
        "⏱️ Timings",
    ])

    # ── TAB: Overview ─────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="dl-card">', unsafe_allow_html=True)
        st.markdown("### 🏥 Health Components")
        comps = health.get("components", {}) or {}
        if comps:
            hc = st.columns(len(comps))
            for col, (k, v) in zip(hc, comps.items()):
                color = "#16a34a" if float(v or 0) >= 80 else "#d97706" if float(v or 0) >= 60 else "#dc2626"
                with col:
                    st.markdown(
                        f'<div class="dl-metric">'
                        f'<div class="label">{k.replace("_"," ")}</div>'
                        f'<div style="font-size:2rem;font-weight:900;color:{color}">{fmt_n(v)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        st.markdown("</div>", unsafe_allow_html=True)

        # Analysis plan
        sel = result.get("analysis_selection") or {}
        summary = sel.get("summary", {})
        if summary:
            st.markdown('<div class="dl-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Analysis plan")
            a1, a2, a3 = st.columns(3)
            a1.metric("Selected", summary.get("selected_count", 0))
            a2.metric("Recommended", summary.get("recommended_count", 0))
            a3.metric("Skipped", summary.get("skipped_count", 0))
            with st.expander("View selected analyses"):
                for a in sel.get("selected_analyses", []):
                    st.markdown(f"✅ **{a.get('name')}** · `{a.get('category')}`")
            st.markdown("</div>", unsafe_allow_html=True)

        # Numeric / categorical columns
        nc, cc = st.columns(2)
        with nc:
            st.markdown('<div class="dl-card">', unsafe_allow_html=True)
            st.markdown("**Numeric columns**")
            nums = profile.get("numeric_columns", [])
            if nums:
                st.markdown(" ".join(f'<span class="chip">{c}</span>' for c in nums), unsafe_allow_html=True)
            else:
                st.caption("None detected")
            st.markdown("</div>", unsafe_allow_html=True)
        with cc:
            st.markdown('<div class="dl-card">', unsafe_allow_html=True)
            st.markdown("**Categorical columns**")
            cats = profile.get("categorical_columns", [])
            if cats:
                st.markdown(" ".join(f'<span class="chip" style="background:#f5f3ff;color:#7c3aed;border-color:#ddd6fe">{c}</span>' for c in cats), unsafe_allow_html=True)
            else:
                st.caption("None detected")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB: Column Roles ─────────────────────────────────────────────────
    with tabs[1]:
        roles_summary = result.get("roles_summary") or {}
        if roles_summary:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Numeric", roles_summary.get("numeric_count", 0))
            r2.metric("Categorical", roles_summary.get("categorical_count", 0))
            r3.metric("ID-like", len(roles_summary.get("id_like", [])))
            r4.metric("Target candidates", len(roles_summary.get("target_candidates", [])))
            st.markdown("<br>", unsafe_allow_html=True)

        column_roles = result.get("column_roles", {}) or {}
        if column_roles:
            rows = []
            for col_name, info in column_roles.items():
                rows.append({
                    "Column": col_name,
                    "Dtype": info.get("dtype", ""),
                    "Roles": ", ".join(info.get("roles", [])),
                    "Missing %": f'{info.get("missing_percent", 0):.1f}%',
                    "Unique": fmt_n(info.get("unique_count")),
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    # ── TAB: Signals ──────────────────────────────────────────────────────
    with tabs[2]:
        if signals:
            for sig in signals:
                sev = (sig.get("severity") or "").lower()
                color = "#fef2f2" if sev in {"high","critical"} else "#fffbeb" if sev=="medium" else "#f0f9ff"
                border = "#fecaca" if sev in {"high","critical"} else "#fde68a" if sev=="medium" else "#bfdbfe"
                text_c = "#dc2626" if sev in {"high","critical"} else "#d97706" if sev=="medium" else "#2563eb"
                st.markdown(
                    f'<div style="background:{color};border:1px solid {border};border-radius:1rem;padding:1rem 1.25rem;margin-bottom:0.75rem">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                    f'<span style="font-size:1.1rem">{sig.get("icon","⚡")}</span>'
                    f'<strong style="color:#1e293b">{sig.get("name","Signal")}</strong>'
                    f'<span style="font-size:11px;font-weight:700;color:{text_c};background:white;border:1px solid {border};border-radius:999px;padding:1px 8px">{sig.get("severity","")}</span>'
                    f'</div>'
                    f'<p style="font-size:13px;color:#475569;margin:0">{sig.get("evidence","")}</p>'
                    f'<p style="font-size:12px;color:#2563eb;margin-top:4px;font-weight:600">💡 {sig.get("recommendation","")}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No signals generated.")

        # Dataset signal flags
        ds = result.get("dataset_signals") or {}
        flags = [k.replace("has_","").replace("_"," ").title() for k, v in ds.items() if v is True and str(k).startswith("has_")]
        if flags:
            st.markdown("**Detected dataset flags**")
            st.markdown(" ".join(f'<span class="chip">{f}</span>' for f in flags), unsafe_allow_html=True)

    # ── TAB: Charts ───────────────────────────────────────────────────────
    with tabs[3]:
        if charts:
            st.caption(f"{len(charts)} charts generated by the role-aware chart planner.")
            col_a, col_b = st.columns(2)
            for i, chart in enumerate(charts):
                img_path_str = chart.get("path", "")
                img_path = None
                for candidate in [
                    Path(img_path_str),
                    Path("static") / img_path_str,
                    Path("static") / Path(img_path_str).name,
                ]:
                    if candidate.exists():
                        img_path = candidate
                        break
                target_col = col_a if i % 2 == 0 else col_b
                with target_col:
                    with st.container(border=True):
                        st.markdown(f"**{chart.get('title','Chart')}**")
                        if chart.get("description"):
                            st.caption(chart["description"])
                        if img_path:
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.warning(f"Image not found: {img_path_str}")
                        if chart.get("planner_reason"):
                            st.caption(f"📌 {chart['planner_reason']}")
        else:
            st.info("No charts generated. Use standard / deep / research mode.")

    # ── TAB: Cleaning ─────────────────────────────────────────────────────
    with tabs[4]:
        if cleaning:
            bh = (cleaning.get("before_health") or {}).get("overall_score", 0)
            ah = (cleaning.get("after_health") or {}).get("overall_score", 0)
            cl1, cl2, cl3, cl4 = st.columns(4)
            cl1.metric("Missing Before", fmt_n(cleaning.get("missing_before", 0)))
            cl2.metric("Missing After",  fmt_n(cleaning.get("missing_after", 0)),
                       delta=f"{cleaning.get('missing_after',0)-cleaning.get('missing_before',0):+,}")
            cl3.metric("Dupes Before",   fmt_n(cleaning.get("duplicates_before", 0)))
            cl4.metric("Dupes After",    fmt_n(cleaning.get("duplicates_after", 0)))

            st.metric("Health: before → after", f"{fmt_n(bh)} → {fmt_n(ah)}", delta=f"{ah-bh:+.1f}")

            actions = cleaning.get("actions", []) or []
            if actions:
                st.markdown("**Cleaning actions**")
                for a in actions:
                    risk = a.get("risk","Low")
                    color = "#dc2626" if risk=="High" else "#d97706" if risk=="Medium" else "#16a34a"
                    st.markdown(
                        f'<div style="padding:8px 12px;border-left:3px solid {color};margin-bottom:6px;background:rgba(255,255,255,0.7);border-radius:0 0.5rem 0.5rem 0">'
                        f'<strong style="color:#1e293b">{a.get("action","")}</strong>'
                        f' <span style="font-size:11px;color:{color};font-weight:700">[{risk}]</span><br>'
                        f'<span style="font-size:12px;color:#64748b">{a.get("details","")}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── TAB: ML Lab ───────────────────────────────────────────────────────
    with tabs[5]:
        lb = result.get("model_leaderboard") or {}
        ex = result.get("explainability") or {}

        if not (isinstance(lb, dict) and lb.get("available")):
            st.info("Set a target column to enable the ML leaderboard.")
        else:
            winner = lb.get("winner") or {}
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Task",    lb.get("task_type","").title())
            m2.metric("Winner",  winner.get("model","—"))
            m3.metric("Score",   fmt_n(winner.get("primary_score")))
            m4.metric("Metric",  lb.get("primary_metric",""))

            leaderboard_rows = [r for r in (lb.get("leaderboard") or []) if r.get("primary_score") is not None]
            if leaderboard_rows:
                st.markdown("**Cross-validated leaderboard**")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Model": r.get("model"),
                            "Primary Score": fmt_n(r.get("primary_score")),
                            "Fit Time (s)": fmt_n(r.get("fit_time_s")),
                        }
                        for r in leaderboard_rows
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

        if isinstance(ex, dict) and ex.get("available"):
            st.markdown("**Feature importance (SHAP / permutation)**")
            gi = ex.get("global_importance") or []
            if gi:
                imp_df = pd.DataFrame(gi[:12]).rename(columns={"feature":"Feature","importance":"Importance"})
                st.dataframe(imp_df, use_container_width=True, hide_index=True)
                # Show SHAP chart if saved
                shap_path = ex.get("summary_chart_path")
                if shap_path and Path(shap_path).exists():
                    st.image(shap_path, caption="SHAP summary plot", use_container_width=True)

    # ── TAB: Data Preview ─────────────────────────────────────────────────
    with tabs[6]:
        if df is not None and not df.empty:
            st.markdown("**First 15 rows**")
            st.dataframe(df.head(15), use_container_width=True)

            num_sum = profile.get("numeric_summary") or {}
            if num_sum:
                st.markdown("**Numeric summary**")
                st.dataframe(
                    pd.DataFrame(num_sum).T.round(3),
                    use_container_width=True,
                )

            cat_sum = profile.get("categorical_summary") or {}
            if cat_sum:
                st.markdown("**Categorical summary**")
                rows = [
                    {"Column": col, "Unique": v.get("unique_values","—"), "Top value": list(v.get("top_values",{}).keys())[0] if v.get("top_values") else "—"}
                    for col, v in cat_sum.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No preview available.")

    # ── TAB: Timings ──────────────────────────────────────────────────────
    with tabs[7]:
        timings = result.get("timings_ms") or {}
        if timings:
            rows = [
                {"Phase": k, "Time (ms)": round(float(v), 1) if isinstance(v, (int, float)) else "—"}
                for k, v in timings.items()
                if isinstance(v, (int, float))
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No timing data available.")


# ── Ask Anything ─────────────────────────────────────────────────────────────

def render_ask(result: dict):
    st.markdown("---")
    st.markdown('<div class="dl-card">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask About Your Data")
    st.caption("Backed by RAG over the analysis result + an Ollama / OpenRouter LLM.")

    question = st.text_input(
        "Question",
        placeholder="e.g. What columns should I use as features?",
        label_visibility="collapsed",
        key="ask_question",
    )
    if st.button("Ask", key="ask_btn") and question.strip():
        from modules.ai_agent import answer_with_agent
        df = result.get("_df")
        with st.spinner("Thinking…"):
            try:
                resp = answer_with_agent(
                    question=question.strip(),
                    df=df,
                    analysis_result=result,
                    fast=True,
                )
                answer = resp.get("answer", "")
                source = resp.get("source", "unknown")
                st.markdown(
                    f'<div style="background:#f0f9ff;border:1px solid #bfdbfe;border-radius:1rem;padding:1rem 1.25rem;margin-top:0.75rem">'
                    f'<p style="font-size:11px;color:#2563eb;font-weight:700;margin-bottom:4px">Source: {source}</p>'
                    f'<p style="font-size:14px;color:#1e293b;white-space:pre-wrap">{answer}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Agent error: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    controls = sidebar()
    render_header()

    # ── Idle state ────────────────────────────────────────────────────────
    if not controls["uploaded"]:
        st.markdown(
            """
<div class="dl-card" style="text-align:center;padding:3rem 2rem">
  <div style="font-size:3rem;margin-bottom:0.5rem">🔬</div>
  <div style="font-size:1.25rem;font-weight:700;color:#1e293b">
    Upload a dataset to begin
  </div>
  <div style="font-size:0.9rem;color:#64748b;margin-top:0.5rem;max-width:480px;margin-inline:auto">
    CSV, TSV, JSON, or Excel. The pipeline will run step by step and
    you'll see every phase as it completes — profiling, quality checks,
    ML leaderboard, SHAP, time-series, charts, and the PDF report.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    # ── Run on button click ───────────────────────────────────────────────
    if controls["run"]:
        st.session_state["result"] = None  # clear previous
        with st.spinner(""):
            result = run_pipeline(
                controls["uploaded"],
                controls["mode"],
                controls["target"],
            )
        if result:
            st.session_state["result"] = result
            st.session_state["mode"] = controls["mode"]
        else:
            st.error("Pipeline failed — see error above.")
            return

    # ── Render results ────────────────────────────────────────────────────
    result = st.session_state.get("result")
    if result:
        render_results(result)
        render_ask(result)
    else:
        st.info("Click **▶ Run analysis** in the sidebar to start.")


if __name__ == "__main__":
    main()
