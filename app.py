"""
Flask Application — Main Frontend Server
==========================================
Serves the Cluely-inspired TailwindCSS frontend.
Streamlit remains as a separate admin/debug portal.
"""

import os
import math
from pathlib import Path
from copy import deepcopy
from time import perf_counter
from threading import Lock, Thread

from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify
from werkzeug.exceptions import ClientDisconnected

from framevitals.loader import (
    load_dataset,
    save_uploaded_file,
)
from framevitals.pipeline import (
    run_full_analysis,
)
from framevitals.ai_insights import (
    answer_dataset_question,
)

from modules.ai_agent import (
    answer_with_agent,
)
from modules.report_generator import (
    generate_pdf_report,
)
from modules.frontend_api import (
    build_dashboard_payload,
)
from framevitals.drift_analysis import (
    compare_datasets,
    split_by_date,
)


# ---------------------------------------------------------------------------
# JSON sanitizer
# ---------------------------------------------------------------------------
# Flask's jsonify emits the JavaScript-only tokens NaN, Infinity, -Infinity by
# default. Browsers reject these in strict-parse mode (response.json() and
# JSON.parse both do), which makes the frontend silently fall back to an
# empty payload. We walk every payload recursively and replace those values
# with None so the wire format is RFC-8259 compliant.

def _is_nonfinite(v) -> bool:
    return isinstance(v, float) and not math.isfinite(v)


def _json_safe(value):
    if _is_nonfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in value]
    return value


def safe_jsonify(payload):
    """Drop-in replacement for jsonify that is strict-JSON safe."""
    return jsonify(_json_safe(payload))


app = Flask(__name__)
app.secret_key = os.environ.get(
    "DATALENS_SECRET_KEY",
    "development-only-secret"
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

ANALYSIS_CACHE = {}
REPORT_JOBS = {}
REPORT_LOCK = Lock()


class DotDict(dict):
    """Allow dict.key access for Jinja templates."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def from_dict(d):
        if isinstance(d, dict):
            return DotDict({k: DotDict.from_dict(v) for k, v in d.items()})
        if isinstance(d, list):
            return [DotDict.from_dict(i) for i in d]
        return d


def _report_path(dataset_id: str) -> Path:
    return REPORT_DIR / f"{dataset_id}_report.pdf"


def _get_report_job(dataset_id: str) -> dict:
    with REPORT_LOCK:
        return dict(REPORT_JOBS.get(dataset_id, {}))


def _set_report_job(dataset_id: str, status: str, pdf_path: Path | None = None, error: str | None = None) -> dict:
    job = {
        "status": status,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "error": error,
    }
    with REPORT_LOCK:
        REPORT_JOBS[dataset_id] = job
    return dict(job)


def _queue_pdf_generation(dataset_id: str, result: dict | None = None) -> dict:
    if result is None:
        with REPORT_LOCK:
            cached_result = ANALYSIS_CACHE.get(dataset_id)
    else:
        cached_result = result

    if cached_result is None:
        return _set_report_job(dataset_id, "missing", error="Analysis result not available.")

    current_job = _get_report_job(dataset_id)
    if current_job.get("status") in {"queued", "running", "ready"}:
        return current_job

    _set_report_job(dataset_id, "queued")

    def worker():
        _set_report_job(dataset_id, "running")
        try:
            pdf_path = generate_pdf_report(deepcopy(cached_result))
            _set_report_job(dataset_id, "ready", pdf_path=pdf_path)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            _set_report_job(dataset_id, "failed", error=str(exc))

    Thread(target=worker, daemon=True).start()
    return _get_report_job(dataset_id)


def _report_status_payload(dataset_id: str) -> dict:
    job = _get_report_job(dataset_id)
    report_path = _report_path(dataset_id)

    if job.get("status") == "ready" and report_path.exists() and report_path.stat().st_size > 0:
        return {
            "status": "ready",
            "ready": True,
            "pdf_path": str(report_path),
            "error": None,
        }

    if not job and report_path.exists() and report_path.stat().st_size > 0:
        return {
            "status": "ready",
            "ready": True,
            "pdf_path": str(report_path),
            "error": None,
        }

    status = job.get("status") or "pending"
    return {
        "status": status,
        "ready": status == "ready",
        "pdf_path": job.get("pdf_path"),
        "error": job.get("error"),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        try:
            uploaded_file = request.files.get("dataset")
            analysis_mode = request.form.get("analysis_mode", "standard")
            target_column = request.form.get("target_column") or None
        except ClientDisconnected:
            return render_template(
                "error.html",
                message="The upload was interrupted before Flask finished reading the file. Please try again.",
            ), 400

        if not uploaded_file or uploaded_file.filename == "":
            return render_template(
                "error.html",
                message="Please upload a valid dataset file.",
            )

        dataset_id, file_path, original_filename = save_uploaded_file(uploaded_file)

        result = run_full_analysis(
            dataset_id=dataset_id,
            file_path=file_path,
            original_filename=original_filename,
            analysis_mode=analysis_mode,
            target_column=target_column,
        )

        with REPORT_LOCK:
            ANALYSIS_CACHE[dataset_id] = deepcopy(result)

        report_job = _queue_pdf_generation(dataset_id, result)
        result["report_status"] = _report_status_payload(dataset_id)

        session["dataset_id"] = dataset_id
        session["file_path"] = str(file_path)
        session["original_filename"] = original_filename
        session["analysis_mode"] = analysis_mode
        session["target_column"] = target_column

        # Convert to DotDict for Jinja template dot-access
        result_dot = DotDict.from_dict(result)

        return render_template("report.html", result=result_dot)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return render_template("error.html", message=str(exc))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        start = perf_counter()
        try:
            uploaded_file = request.files.get("dataset")
            analysis_mode = request.form.get("analysis_mode", "standard")
            target_column = request.form.get("target_column") or None
        except ClientDisconnected:
            return jsonify({"error": "The upload was interrupted before the server finished reading it. Please try again."}), 400

        if analysis_mode not in {"quick", "standard", "deep", "research"}:
            analysis_mode = "standard"

        if not uploaded_file or uploaded_file.filename == "":
            return jsonify({"error": "Please upload a valid dataset file."}), 400

        dataset_id, file_path, original_filename = save_uploaded_file(uploaded_file)

        result = run_full_analysis(
            dataset_id=dataset_id,
            file_path=file_path,
            original_filename=original_filename,
            analysis_mode=analysis_mode,
            target_column=target_column,
        )

        df = load_dataset(file_path)
        elapsed_ms = (perf_counter() - start) * 1000

        payload = build_dashboard_payload(
            result=result,
            df=df,
            file_path=file_path,
            analysis_mode=analysis_mode,
            elapsed_ms=elapsed_ms,
            target_column=target_column,
        )

        with REPORT_LOCK:
            ANALYSIS_CACHE[dataset_id] = deepcopy(result)

        _queue_pdf_generation(dataset_id, result)
        report_status = _report_status_payload(dataset_id)
        payload["reportStatus"] = report_status
        payload.setdefault("downloadLinks", {})["reportReady"] = report_status["ready"]
        payload["downloadLinks"]["reportStatus"] = report_status["status"]

        session["dataset_id"] = dataset_id
        session["file_path"] = str(file_path)
        session["original_filename"] = original_filename
        session["analysis_mode"] = analysis_mode
        session["target_column"] = target_column

        return safe_jsonify(payload)

    except Exception as exc:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        dataset_id = session.get("dataset_id")
        file_path = session.get("file_path")
        original_filename = session.get("original_filename", "dataset")
        analysis_mode = session.get("analysis_mode", "standard")
        target_column = session.get("target_column")
        question = request.form.get("question", "")

        if not dataset_id or not file_path:
            return redirect(url_for("index"))

        result = run_full_analysis(
            dataset_id=dataset_id,
            file_path=Path(file_path),
            original_filename=original_filename,
            analysis_mode=analysis_mode,
            target_column=target_column,
            skip_ai=True,
        )

        with REPORT_LOCK:
            ANALYSIS_CACHE[dataset_id] = deepcopy(result)

        _queue_pdf_generation(dataset_id, result)
        result["report_status"] = _report_status_payload(dataset_id)

        # Try agentic answer first; fall back to the legacy single-shot answerer
        # if anything goes wrong (Ollama offline, model errors, etc.).
        try:
            agent_response = answer_with_agent(
                question=question,
                df=load_dataset(Path(file_path)),
                analysis_result=result,
            )
            answer = {
                "source": agent_response.get("source", "agent"),
                "answer": agent_response.get("answer", ""),
                "trace": agent_response.get("trace", {}),
            }
        except Exception:
            answer = answer_dataset_question(
                question=question,
                profile=result["profile"],
                health=result["health"],
                signals=result["signals"],
                ml_readiness=result["ml_readiness"],
                advanced=result.get("advanced"),
            )

        result["chat_answer"] = answer
        result["chat_question"] = question

        result_dot = DotDict.from_dict(result)
        return render_template("report.html", result=result_dot)

    except Exception as exc:
        return render_template("error.html", message=str(exc))


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """JSON endpoint for the agentic Q&A loop. Used by the React frontend."""
    try:
        body = request.get_json(silent=True) or {}
        question = (body.get("question") or "").strip()
        dataset_id = body.get("dataset_id") or session.get("dataset_id")

        if not question:
            return jsonify({"error": "Missing 'question' in request body."}), 400

        # Look up the cached analysis result first
        with REPORT_LOCK:
            cached_result = ANALYSIS_CACHE.get(dataset_id)

        if cached_result is None:
            return jsonify({
                "error": "No cached analysis was found for this dataset. Run /api/analyze first.",
            }), 404

        # Reload dataframe (cheap; cleaned files live under uploads/)
        file_path = session.get("file_path")
        df = None
        if file_path:
            try:
                df = load_dataset(Path(file_path))
            except Exception:
                df = None

        try:
            # Fast mode by default (single writer call, no critic/repair).
            # Pass {"mode": "full"} in the body to opt back in to the full
            # planner→executor→critic→writer loop.
            mode = (body.get("mode") or "fast").lower().strip()
            response = answer_with_agent(
                question=question,
                df=df,
                analysis_result=cached_result,
                fast=(mode != "full"),
            )
        except Exception as exc:
            response = answer_dataset_question(
                question=question,
                profile=cached_result["profile"],
                health=cached_result["health"],
                signals=cached_result["signals"],
                ml_readiness=cached_result["ml_readiness"],
                advanced=cached_result.get("advanced"),
            )
            response = {"source": response.get("source", "fallback"), "answer": response.get("answer", str(exc)), "trace": {}}

        return safe_jsonify({
            "question": question,
            "dataset_id": dataset_id,
            "source": response.get("source"),
            "answer": response.get("answer"),
            "trace": response.get("trace", {}),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500



@app.route("/api/ai-report", methods=["POST"])
def api_ai_report():
    """
    On-demand AI report generation. The pipeline skips this phase by default
    (set DATALENS_ANALYZE_AI=1 to run it during /api/analyze). The frontend
    calls this endpoint when the user clicks "Generate AI report" on the
    AI Report tab.

    Body: {"dataset_id": "..."}
    """
    try:
        body = request.get_json(silent=True) or {}
        dataset_id = body.get("dataset_id") or session.get("dataset_id")

        if not dataset_id:
            return jsonify({"error": "Missing dataset_id."}), 400

        with REPORT_LOCK:
            cached = ANALYSIS_CACHE.get(dataset_id)

        if cached is None:
            return jsonify({"error": "No cached analysis for this dataset. Re-run /api/analyze."}), 404

        from framevitals.ai_insights import (
            generate_ai_report,
            )

        try:
            ai_report = generate_ai_report(
                profile=cached.get("profile") or {},
                health=cached.get("health") or {},
                signals=cached.get("signals") or [],
                ml_readiness=cached.get("ml_readiness") or {},
                advanced=cached.get("advanced") or {},
                column_roles_summary=cached.get("roles_summary") or {},
                dataset_signals=cached.get("dataset_signals") or {},
            )
        except Exception as exc:
            ai_report = {"source": f"error: {exc}", "text": str(exc)}

        # Re-cache so subsequent /api/analyze calls (or PDF regen) include it.
        with REPORT_LOCK:
            cached["ai_report"] = ai_report
            ANALYSIS_CACHE[dataset_id] = cached

        return safe_jsonify(ai_report)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def api_health():
    """
    Backend status check used by the live console to populate the status row.

    Reports:
        - Flask backend reachable (always true if this returns)
        - Ollama reachable (best-effort socket probe)
        - OpenRouter API key present
        - Cached analyses count
        - Pipeline modules loaded
    """
    import socket

    def _probe(host: str, port: int, timeout: float = 0.4) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    ollama_reachable = _probe("127.0.0.1", 11434)
    openrouter_configured = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())

    return jsonify({
        "flask": True,
        "ollama_reachable": ollama_reachable,
        "openrouter_configured": openrouter_configured,
        "cached_analyses": len(ANALYSIS_CACHE),
        "pdf_jobs": len(REPORT_JOBS),
        "version": "v3",
    })


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    Compare two uploaded datasets. Multipart form fields:
        reference: file (older / training)
        current:   file (newer / production)
        columns:   optional comma-separated list to restrict comparison
    """
    try:
        try:
            ref_file = request.files.get("reference")
            cur_file = request.files.get("current")
        except ClientDisconnected:
            return jsonify({"error": "Upload was interrupted."}), 400

        if not ref_file or not ref_file.filename:
            return jsonify({"error": "Missing 'reference' file."}), 400
        if not cur_file or not cur_file.filename:
            return jsonify({"error": "Missing 'current' file."}), 400

        _, ref_path, _ = save_uploaded_file(ref_file)
        _, cur_path, _ = save_uploaded_file(cur_file)

        df_ref = load_dataset(ref_path)
        df_cur = load_dataset(cur_path)

        columns_param = request.form.get("columns", "").strip()
        columns = [c.strip() for c in columns_param.split(",") if c.strip()] if columns_param else None

        report = compare_datasets(df_ref, df_cur, columns=columns)
        report["reference_filename"] = ref_file.filename
        report["current_filename"] = cur_file.filename
        return safe_jsonify(report)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/compare-self", methods=["POST"])
def api_compare_self():
    """
    Compare a single dataset against itself by splitting on a date column.
    Multipart fields:
        dataset:     file
        date_column: str (column to split on)
        ratio:       float in (0, 1), default 0.5
    """
    try:
        try:
            ds_file = request.files.get("dataset")
        except ClientDisconnected:
            return jsonify({"error": "Upload was interrupted."}), 400

        if not ds_file or not ds_file.filename:
            return jsonify({"error": "Missing 'dataset' file."}), 400

        date_column = (request.form.get("date_column") or "").strip()
        if not date_column:
            return jsonify({"error": "Missing 'date_column' field."}), 400

        try:
            ratio = float(request.form.get("ratio", "0.5"))
        except ValueError:
            ratio = 0.5
        ratio = max(0.1, min(0.9, ratio))

        _, ds_path, ds_name = save_uploaded_file(ds_file)
        df = load_dataset(ds_path)

        try:
            df_ref, df_cur = split_by_date(df, date_column, ratio=ratio)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        report = compare_datasets(df_ref, df_cur)
        report["reference_filename"] = f"{ds_name} (older {ratio:.0%})"
        report["current_filename"] = f"{ds_name} (newer {1 - ratio:.0%})"
        report["split_by"] = date_column
        report["split_ratio"] = ratio
        return safe_jsonify(report)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/download-cleaned/<dataset_id>")
def download_cleaned(dataset_id):
    path = Path("cleaned") / f"{dataset_id}_cleaned.csv"
    if path.exists():
        return send_file(path, as_attachment=True)
    return render_template("error.html", message="Cleaned dataset not found.")


@app.route("/api/report-status/<dataset_id>")
def api_report_status(dataset_id):
    status = _report_status_payload(dataset_id)
    status["downloadUrl"] = f"/download-report/{dataset_id}"
    status["dataset_id"] = dataset_id
    return safe_jsonify(status)


@app.route("/download-report/<dataset_id>")
def download_report(dataset_id):
    try:
        pdf_path = _report_path(dataset_id)
        report_status = _report_status_payload(dataset_id)

        if report_status["ready"] and pdf_path.exists() and pdf_path.stat().st_size > 0:
            return send_file(pdf_path, as_attachment=True)

        result = ANALYSIS_CACHE.get(dataset_id)
        if result is not None:
            _queue_pdf_generation(dataset_id, result)

        report_status = _report_status_payload(dataset_id)

        if report_status["ready"] and pdf_path.exists() and pdf_path.stat().st_size > 0:
            return send_file(pdf_path, as_attachment=True)

        message = "The PDF report is generating in the background. Please try again in a few seconds."
        if report_status["status"] == "failed":
            message = f"PDF generation failed: {report_status.get('error', 'Unknown error')}"
        elif result is None:
            message = "No cached analysis was found for this dataset. Please run analysis again first."

        return render_template("error.html", message=message), 202

    except Exception as exc:
        return render_template("error.html", message=str(exc))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5055)
