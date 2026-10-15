"""FrameVitals Flask API and generated-artifact server.

The React application is the single product UI. This backend stays thin: it
reuses the same protocol/execution policy as the public Python API, bounds
in-process cache state, and keeps filesystem/network side effects inside
explicit API and artifact handlers.
"""

from __future__ import annotations

import os
import re
import secrets
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import BoundedSemaphore, Lock
from time import perf_counter

from flask import Flask, g, jsonify, redirect, request, send_file, session
from werkzeug.exceptions import ClientDisconnected, HTTPException

from framevitals import __version__ as FRAMEVITALS_VERSION
from framevitals.ai_insights import answer_dataset_question
from framevitals.drift_analysis import split_by_date
from framevitals.frontend_api import build_dashboard_payload
from framevitals.loader import UPLOAD_DIR, load_dataset, save_uploaded_file
from framevitals.pipeline import run_full_analysis
from framevitals.planner import effective_disabled_modules

_VALID_ANALYSIS_MODES = {"quick", "standard", "deep", "research"}
_DATASET_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")
CLEANED_DIR = Path("cleaned")
REPORT_DIR = Path("reports")


def _bounded_positive_env(name: str, default: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 1:
        return default
    return min(value, maximum)


_WEB_CACHE_LIMIT = _bounded_positive_env("FRAMEVITALS_WEB_CACHE_LIMIT", 16, 128)
_REPORT_JOB_LIMIT = max(_WEB_CACHE_LIMIT, 32)


def safe_jsonify(payload):
    from framevitals.serialization import json_safe

    return jsonify(json_safe(payload))


app = Flask(__name__)
app.secret_key = os.environ.get("FRAMEVITALS_SECRET_KEY") or secrets.token_urlsafe(32)
app.config.update(
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FRAMEVITALS_SECURE_COOKIES") == "1",
)
ANALYSIS_SLOTS = BoundedSemaphore(2)
PDF_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="framevitals-report")
PDF_SLOTS = BoundedSemaphore(4)

ANALYSIS_CACHE: OrderedDict[str, dict] = OrderedDict()
UPLOAD_PATHS: OrderedDict[str, str] = OrderedDict()
REPORT_JOBS: OrderedDict[str, dict] = OrderedDict()
REPORT_LOCK = Lock()


def _normalize_analysis_mode(value: str | None) -> str:
    mode = (value or "standard").strip().lower()
    if mode not in _VALID_ANALYSIS_MODES:
        raise ValueError(f"Unknown analysis mode: {mode}")
    return mode


def _validate_dataset_id(dataset_id: str | None) -> str:
    if not isinstance(dataset_id, str) or not _DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise ValueError("Invalid dataset identifier.")
    return dataset_id


def _trusted_generated_path(value: str | Path | None, root: Path) -> Path | None:
    """Resolve a server-generated artifact path and keep it inside ``root``."""
    if value is None:
        return None
    try:
        root_path = root.resolve()
        candidate = Path(value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if candidate.parent != root_path:
        return None
    return candidate


def _is_nonempty_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _cache_upload_path(dataset_id: str, file_path: Path) -> None:
    dataset_id = _validate_dataset_id(dataset_id)
    trusted = _trusted_generated_path(file_path, UPLOAD_DIR)
    if trusted is None:
        raise ValueError("Upload path escaped the managed upload directory.")
    with REPORT_LOCK:
        UPLOAD_PATHS[dataset_id] = str(trusted)
        UPLOAD_PATHS.move_to_end(dataset_id)
        while len(UPLOAD_PATHS) > _WEB_CACHE_LIMIT:
            _, expired_path = UPLOAD_PATHS.popitem(last=False)
            _unlink_quietly(Path(expired_path))


def _get_upload_path(dataset_id: str | None) -> Path | None:
    try:
        dataset_id = _validate_dataset_id(dataset_id)
    except ValueError:
        return None
    with REPORT_LOCK:
        value = UPLOAD_PATHS.get(dataset_id)
        if value is not None:
            UPLOAD_PATHS.move_to_end(dataset_id)
    return _trusted_generated_path(value, UPLOAD_DIR)


def _cache_analysis(dataset_id: str, result: dict) -> None:
    """Store a defensive result copy while bounding process memory growth."""
    dataset_id = _validate_dataset_id(dataset_id)
    with REPORT_LOCK:
        ANALYSIS_CACHE[dataset_id] = deepcopy(result)
        ANALYSIS_CACHE.move_to_end(dataset_id)
        while len(ANALYSIS_CACHE) > _WEB_CACHE_LIMIT:
            evicted_id, _ = ANALYSIS_CACHE.popitem(last=False)
            REPORT_JOBS.pop(evicted_id, None)
            expired_path = UPLOAD_PATHS.pop(evicted_id, None)
            if expired_path:
                _unlink_quietly(Path(expired_path))


def _get_cached_analysis(dataset_id: str | None) -> dict | None:
    try:
        dataset_id = _validate_dataset_id(dataset_id)
    except ValueError:
        return None
    with REPORT_LOCK:
        result = ANALYSIS_CACHE.get(dataset_id)
        if result is None:
            return None
        ANALYSIS_CACHE.move_to_end(dataset_id)
        return deepcopy(result)


def _get_report_job(dataset_id: str) -> dict:
    dataset_id = _validate_dataset_id(dataset_id)
    with REPORT_LOCK:
        job = REPORT_JOBS.get(dataset_id, {})
        if job:
            REPORT_JOBS.move_to_end(dataset_id)
        return dict(job)


def _set_report_job(
    dataset_id: str,
    status: str,
    pdf_path: Path | None = None,
    error: str | None = None,
) -> dict:
    dataset_id = _validate_dataset_id(dataset_id)
    trusted_pdf = _trusted_generated_path(pdf_path, REPORT_DIR) if pdf_path else None
    if pdf_path is not None and trusted_pdf is None:
        raise ValueError("Report path escaped the managed report directory.")
    job = {
        "status": status,
        "pdf_path": str(trusted_pdf) if trusted_pdf else None,
        "error": error,
    }
    with REPORT_LOCK:
        REPORT_JOBS[dataset_id] = job
        REPORT_JOBS.move_to_end(dataset_id)
        while len(REPORT_JOBS) > _REPORT_JOB_LIMIT:
            REPORT_JOBS.popitem(last=False)
    return dict(job)


def _run_web_analysis(
    *,
    dataset_id: str,
    original_filename: str,
    analysis_mode: str,
    target_column: str | None,
    file_path: Path | None = None,
    dataframe=None,
    skip_ai: bool = True,
) -> dict:
    """Run the materialized web pipeline with the canonical mode policy."""
    mode = _normalize_analysis_mode(analysis_mode)
    return run_full_analysis(
        dataset_id=_validate_dataset_id(dataset_id),
        file_path=file_path,
        original_filename=original_filename,
        analysis_mode=mode,
        target_column=target_column,
        dataframe=dataframe,
        skip_ai=skip_ai,
        disabled_modules=effective_disabled_modules(mode, ()),
    )


def _queue_pdf_generation(dataset_id: str, result: dict | None = None) -> dict:
    dataset_id = _validate_dataset_id(dataset_id)
    cached_result = deepcopy(result) if result is not None else _get_cached_analysis(dataset_id)
    if cached_result is None:
        return _set_report_job(
            dataset_id,
            "missing",
            error="Analysis result not available.",
        )

    current_job = _get_report_job(dataset_id)
    if current_job.get("status") in {"queued", "running"}:
        return current_job
    if current_job.get("status") == "ready":
        report_path = _trusted_generated_path(current_job.get("pdf_path"), REPORT_DIR)
        if _is_nonempty_file(report_path):
            return current_job

    if not PDF_SLOTS.acquire(blocking=False):
        return _set_report_job(
            dataset_id, "deferred", error="Report queue is busy; retry download."
        )
    _set_report_job(dataset_id, "queued")

    def worker() -> None:
        _set_report_job(dataset_id, "running")
        try:
            from framevitals.report_generator import generate_pdf_report

            pdf_path = generate_pdf_report(cached_result)
            _set_report_job(dataset_id, "ready", pdf_path=pdf_path)
        except Exception:  # optional report generation must fail soft
            app.logger.exception("PDF generation failed for dataset %s", dataset_id)
            _set_report_job(
                dataset_id,
                "failed",
                error="PDF report generation failed. Check server logs for details.",
            )

        finally:
            PDF_SLOTS.release()

    PDF_POOL.submit(worker)
    return _get_report_job(dataset_id)


def _report_status_payload(dataset_id: str) -> dict:
    dataset_id = _validate_dataset_id(dataset_id)
    job = _get_report_job(dataset_id)
    report_path = _trusted_generated_path(job.get("pdf_path"), REPORT_DIR)

    if job.get("status") == "ready" and _is_nonempty_file(report_path):
        return {
            "status": "ready",
            "ready": True,
            "pdf_path": None,
            "error": None,
        }

    status = job.get("status") or "pending"
    if status == "ready":
        status = "missing"
    return {
        "status": status,
        "ready": False,
        "pdf_path": None,
        "error": job.get("error") if status == "failed" else None,
    }


def _store_session(
    *,
    dataset_id: str,
    original_filename: str,
    analysis_mode: str,
    target_column: str | None,
) -> None:
    # Keep paths out of client-side session state. The upload path is held in a
    # bounded server-side map keyed by the generated dataset identifier.
    session["datasets"] = [*session.get("datasets", [])[-15:], dataset_id]
    session["dataset_id"] = _validate_dataset_id(dataset_id)
    session["original_filename"] = original_filename
    session["analysis_mode"] = _normalize_analysis_mode(analysis_mode)
    session["target_column"] = target_column


def _unlink_quietly(path: Path | None) -> None:
    if path is None:
        return
    trusted = _trusted_generated_path(path, UPLOAD_DIR)
    if trusted is None:
        app.logger.warning("Refusing to remove path outside the managed upload directory")
        return
    try:
        trusted.unlink(missing_ok=True)
    except OSError:
        app.logger.warning("Could not remove temporary upload")


@app.route("/")
def index():
    """Redirect the backend root to the single React product surface."""
    frontend_url = os.environ.get(
        "FRAMEVITALS_FRONTEND_URL",
        "https://framevitals.vercel.app/",
    )
    return redirect(frontend_url, code=302)


@app.route("/analyze", methods=["POST"])
def analyze():
    """Compatibility tombstone for the removed server-rendered dashboard."""
    return jsonify(
        {
            "error": "The legacy HTML analysis endpoint has been removed.",
            "use": "/api/analyze",
        }
    ), 410


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        start = perf_counter()
        try:
            uploaded_file = request.files.get("dataset")
            analysis_mode = _normalize_analysis_mode(request.form.get("analysis_mode"))
            target_column = request.form.get("target_column") or None
        except ClientDisconnected:
            return jsonify(
                {
                    "error": (
                        "The upload was interrupted before the server finished reading it. "
                        "Please try again."
                    )
                }
            ), 400

        if not uploaded_file or uploaded_file.filename == "":
            return jsonify({"error": "Please upload a valid dataset file."}), 400

        dataset_id, file_path, original_filename = save_uploaded_file(uploaded_file)
        _cache_upload_path(dataset_id, file_path)
        df = load_dataset(file_path)
        result = _run_web_analysis(
            dataset_id=dataset_id,
            dataframe=df,
            original_filename=original_filename,
            analysis_mode=analysis_mode,
            target_column=target_column,
        )
        elapsed_ms = (perf_counter() - start) * 1000
        payload = build_dashboard_payload(
            result=result,
            df=df,
            file_path=file_path,
            analysis_mode=analysis_mode,
            elapsed_ms=elapsed_ms,
            target_column=target_column,
        )

        _cache_analysis(dataset_id, result)
        _queue_pdf_generation(dataset_id, result)
        report_status = _report_status_payload(dataset_id)
        payload["reportStatus"] = report_status
        payload.setdefault("downloadLinks", {})["reportReady"] = report_status["ready"]
        payload["downloadLinks"]["reportStatus"] = report_status["status"]
        _store_session(
            dataset_id=dataset_id,
            original_filename=original_filename,
            analysis_mode=analysis_mode,
            target_column=target_column,
        )
        return safe_jsonify(payload)
    except (ValueError, ImportError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    except HTTPException:
        raise
    except Exception:
        app.logger.exception("API analysis failed")
        return jsonify({"error": "Dataset analysis failed."}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Compatibility tombstone for the removed server-rendered Q&A route."""
    return jsonify(
        {
            "error": "The legacy HTML Q&A endpoint has been removed.",
            "use": "/api/ask",
        }
    ), 410


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """JSON endpoint for the optional agentic Q&A loop."""
    try:
        body = request.get_json(silent=True) or {}
        question = (body.get("question") or "").strip()
        session_dataset_id = session.get("dataset_id")
        dataset_id = body.get("dataset_id") or session_dataset_id

        if not question:
            return jsonify({"error": "Missing 'question' in request body."}), 400
        try:
            dataset_id = _validate_dataset_id(dataset_id)
        except ValueError:
            return jsonify({"error": "Invalid dataset identifier."}), 400

        cached_result = _get_cached_analysis(dataset_id)
        if cached_result is None:
            return jsonify(
                {
                    "error": "No cached analysis was found for this dataset. Run /api/analyze first.",
                }
            ), 404

        file_path = _get_upload_path(dataset_id)
        df = None
        if file_path is not None:
            try:
                df = load_dataset(file_path)
            except Exception:
                app.logger.warning("Could not reload cached upload for agent analysis")
                df = None

        try:
            from framevitals.ai_agent import answer_with_agent

            mode = (body.get("mode") or "fast").lower().strip()
            response = answer_with_agent(
                question=question,
                df=df,
                analysis_result=cached_result,
                fast=(mode != "full"),
            )
        except Exception:
            fallback = answer_dataset_question(
                question=question,
                profile=cached_result["profile"],
                health=cached_result["health"],
                signals=cached_result["signals"],
                ml_readiness=cached_result["ml_readiness"],
                advanced=cached_result.get("advanced"),
            )
            response = {
                "source": fallback.get("source", "fallback"),
                "answer": fallback.get("answer") or "Question answering is unavailable.",
                "trace": {},
            }

        return safe_jsonify(
            {
                "question": question,
                "dataset_id": dataset_id,
                "source": response.get("source"),
                "answer": response.get("answer"),
                "trace": response.get("trace", {}),
            }
        )
    except Exception:
        app.logger.exception("API question answering failed")
        return jsonify({"error": "Question answering failed."}), 500


@app.route("/api/ai-report", methods=["POST"])
def api_ai_report():
    """Generate the optional AI narrative for an existing cached analysis."""
    try:
        body = request.get_json(silent=True) or {}
        dataset_id = body.get("dataset_id") or session.get("dataset_id")
        try:
            dataset_id = _validate_dataset_id(dataset_id)
        except ValueError:
            return jsonify({"error": "Invalid dataset identifier."}), 400

        cached = _get_cached_analysis(dataset_id)
        if cached is None:
            return jsonify(
                {"error": "No cached analysis for this dataset. Re-run /api/analyze."}
            ), 404

        from framevitals.ai_insights import generate_ai_report

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
        except Exception:
            app.logger.exception("AI report model generation failed")
            ai_report = {
                "source": "error",
                "text": "AI report generation failed. Check server logs for details.",
            }

        cached["ai_report"] = ai_report
        _cache_analysis(dataset_id, cached)
        return safe_jsonify(ai_report)
    except Exception:
        app.logger.exception("AI report generation failed")
        return jsonify({"error": "AI report generation failed."}), 500


@app.route("/api/health")
def api_health():
    import socket

    def _probe(host: str, port: int, timeout: float = 0.4) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    with REPORT_LOCK:
        cached_count = len(ANALYSIS_CACHE)
        job_count = len(REPORT_JOBS)

    return jsonify(
        {
            "flask": True,
            "ollama_reachable": _probe("127.0.0.1", 11434),
            "openrouter_configured": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
            "cached_analyses": cached_count,
            "pdf_jobs": job_count,
            "version": FRAMEVITALS_VERSION,
        }
    )


@app.route("/api/compare", methods=["POST"])
def api_compare():
    ref_path = None
    cur_path = None
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

        _, ref_path, ref_name = save_uploaded_file(ref_file)
        _, cur_path, cur_name = save_uploaded_file(cur_file)
        columns_param = request.form.get("columns", "").strip()
        columns = (
            [column.strip() for column in columns_param.split(",") if column.strip()]
            if columns_param
            else None
        )

        from framevitals.operations import compare

        report = compare(ref_path, cur_path, columns=columns)
        report["reference_filename"] = ref_name
        report["current_filename"] = cur_name
        return safe_jsonify(report)
    except (ValueError, TypeError, ImportError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    except HTTPException:
        raise
    except Exception:
        app.logger.exception("Dataset comparison failed")
        return jsonify({"error": "Dataset comparison failed."}), 500
    finally:
        _unlink_quietly(ref_path)
        _unlink_quietly(cur_path)


@app.route("/api/compare-self", methods=["POST"])
def api_compare_self():
    ds_path = None
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

        from framevitals.operations import compare

        report = compare(df_ref, df_cur)
        report["reference_filename"] = f"{ds_name} (older {ratio:.0%})"
        report["current_filename"] = f"{ds_name} (newer {1 - ratio:.0%})"
        report["split_by"] = date_column
        report["split_ratio"] = ratio
        return safe_jsonify(report)
    except Exception:
        app.logger.exception("Self-comparison failed")
        return jsonify({"error": "Dataset self-comparison failed."}), 500
    finally:
        _unlink_quietly(ds_path)


@app.route("/download-cleaned/<dataset_id>")
def download_cleaned(dataset_id):
    try:
        dataset_id = _validate_dataset_id(dataset_id)
    except ValueError:
        return jsonify({"error": "Invalid dataset identifier."}), 400

    result = _get_cached_analysis(dataset_id)
    cleaning = result.get("cleaning", {}) if isinstance(result, dict) else {}
    output_value = cleaning.get("output_path") if isinstance(cleaning, dict) else None
    path = _trusted_generated_path(output_value, CLEANED_DIR)
    if _is_nonempty_file(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "Cleaned dataset not found."}), 404


@app.route("/api/report-status/<dataset_id>")
def api_report_status(dataset_id):
    try:
        dataset_id = _validate_dataset_id(dataset_id)
    except ValueError:
        return jsonify({"error": "Invalid dataset identifier."}), 400
    status = _report_status_payload(dataset_id)
    status["downloadUrl"] = f"/download-report/{dataset_id}"
    status["dataset_id"] = dataset_id
    return safe_jsonify(status)


@app.route("/download-report/<dataset_id>")
def download_report(dataset_id):
    try:
        dataset_id = _validate_dataset_id(dataset_id)
        report_status = _report_status_payload(dataset_id)
        pdf_path = _trusted_generated_path(_get_report_job(dataset_id).get("pdf_path"), REPORT_DIR)

        if report_status["ready"] and _is_nonempty_file(pdf_path):
            return send_file(pdf_path, as_attachment=True)

        result = _get_cached_analysis(dataset_id)
        if result is not None:
            _queue_pdf_generation(dataset_id, result)

        report_status = _report_status_payload(dataset_id)
        pdf_path = _trusted_generated_path(_get_report_job(dataset_id).get("pdf_path"), REPORT_DIR)
        if report_status["ready"] and _is_nonempty_file(pdf_path):
            return send_file(pdf_path, as_attachment=True)

        message = "The PDF report is generating in the background. Please try again shortly."
        if report_status["status"] == "failed":
            message = "PDF report generation failed. Check the server logs for details."
        elif result is None:
            message = (
                "No cached analysis was found for this dataset. Please run analysis again first."
            )
        return jsonify({"error": message, "status": report_status["status"]}), 202
    except ValueError:
        return jsonify({"error": "Invalid dataset identifier."}), 400
    except Exception:
        app.logger.exception("PDF download failed")
        return jsonify({"error": "PDF download failed. Check the server logs for details."}), 500


def serve(*, port: int = 5055) -> None:
    app.run(debug=False, host="127.0.0.1", port=port)


@app.errorhandler(HTTPException)
def http_error(error):
    return jsonify({"error": error.description}), error.code


@app.before_request
def protect_requests():
    """Same-origin mutations and uniform ownership for cached private outputs."""
    if request.content_length and request.content_length > app.config["MAX_CONTENT_LENGTH"]:
        return jsonify({"error": "Upload exceeds the configured request limit."}), 413
    if request.path in {"/api/analyze", "/api/compare", "/api/compare-self"}:
        if not ANALYSIS_SLOTS.acquire(blocking=False):
            return jsonify({"error": "Analysis server is busy. Try again shortly."}), 429
        g.analysis_slot = True
    if request.method == "POST":
        origin = request.headers.get("Origin")
        allowed = {request.host_url.rstrip("/")}
        allowed.update(filter(None, os.environ.get("FRAMEVITALS_ALLOWED_ORIGINS", "").split(",")))
        if request.host.split(":")[0] in {"localhost", "127.0.0.1"}:
            allowed.update({"http://localhost:5173", "http://127.0.0.1:5173"})
        if origin and origin not in allowed:
            return jsonify({"error": "This origin is not allowed to submit data."}), 403
    dataset_id = (request.view_args or {}).get("dataset_id")
    static_name = (request.view_args or {}).get("filename", "")
    if request.path.startswith("/static/charts/"):
        dataset_id = Path(static_name).name.split("_")[0]
    if request.path in {"/api/ask", "/api/ai-report"}:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Expected a JSON object."}), 400
        if body.get("allow_ai") is not True:
            return jsonify(
                {
                    "error": "AI may send data to the configured provider. Explicit allow_ai consent is required."
                }
            ), 403
        dataset_id = body.get("dataset_id") or session.get("dataset_id")
    if dataset_id is not None:
        owned = session.get("datasets", [])
        if dataset_id not in owned and dataset_id != session.get("dataset_id"):
            return jsonify({"error": "Analysis is not available in this session."}), 403


@app.after_request
def security_headers(response):
    if getattr(g, "analysis_slot", False):
        ANALYSIS_SLOTS.release()
        g.analysis_slot = False
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.path.startswith(("/api/", "/download-")):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/capabilities")
def api_capabilities():
    from framevitals.capabilities import capabilities

    return safe_jsonify(capabilities())


def _receive_source(field="dataset"):
    uploaded = request.files.get(field)
    if not uploaded or not uploaded.filename:
        raise ValueError(f"Choose a file for '{field}'.")
    return save_uploaded_file(uploaded)


@app.post("/api/inspect")
def api_inspect():
    """Inspect one temporary source; always remove its upload after preview."""
    path = None
    if not ANALYSIS_SLOTS.acquire(blocking=False):
        return jsonify({"error": "Analysis server is busy. Try again shortly."}), 429
    try:
        from framevitals.capabilities import capabilities
        from framevitals.file_formats import prepare_file_source

        identifier, path, filename = _receive_source()
        info = capabilities(path)
        preview = None
        if info["kind"] == "tabular" and path.suffix.lower() in {".csv", ".tsv"}:
            from framevitals.loader import _read_csv_tolerant

            frame = _read_csv_tolerant(
                path, sep="\t" if path.suffix.lower() == ".tsv" else ",", nrows=6
            )
            if len(frame.columns) > 2000:
                raise ValueError("Browser preview is limited to 2,000 columns.")
            preview = {
                "columns": list(frame.columns[:40]),
                "rows": frame.iloc[:6, :40]
                .astype(object)
                .where(frame.iloc[:6, :40].notna(), None)
                .values.tolist(),
                "limited": True,
            }
        elif info["kind"] == "tensor":
            array = prepare_file_source(path)
            preview = {"shape": list(array.shape), "dtype": str(array.dtype)}
        return safe_jsonify(
            {
                "filename": filename,
                "size_bytes": path.stat().st_size,
                "source": info,
                "preview": preview,
            }
        )
    except (ValueError, TypeError, ImportError, OSError, RecursionError) as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        _unlink_quietly(path)
        ANALYSIS_SLOTS.release()


@app.post("/api/prism")
def api_prism():
    """Unified browser protocol: same source dispatcher and calculations as Python/CLI."""
    path = reference_path = None
    if not ANALYSIS_SLOTS.acquire(blocking=False):
        return jsonify({"error": "Analysis server is busy. Try again shortly."}), 429
    try:
        from framevitals.capabilities import capabilities
        from framevitals.protocols import prism
        from framevitals.result_contract import build_envelope

        start = perf_counter()
        mode = _normalize_analysis_mode(request.form.get("analysis_mode"))
        dataset_id, path, filename = _receive_source()
        source = capabilities(path)
        target = request.form.get("target_column") or None
        reference = request.files.get("reference")
        if reference and reference.filename:
            _, reference_path, _ = _receive_source("reference")
        result = prism(
            path,
            reference=reference_path,
            depth=mode,
            focus=target,
            artifacts=False,
            disabled_modules=["ai", "charts", "cleaning"],
        )
        # Generated storage names must not replace user-facing provenance.
        result.analysis["filename"] = filename
        result.analysis["dataset_id"] = dataset_id
        payload = build_envelope(result, source=source, elapsed_ms=(perf_counter() - start) * 1000)
        payload.update(id=dataset_id, filename=filename)
        # Pure results are sufficient; never retain raw structured uploads.
        _cache_analysis(dataset_id, result.analysis)
        _store_session(
            dataset_id=dataset_id,
            original_filename=filename,
            analysis_mode=mode,
            target_column=target,
        )
        return safe_jsonify(payload)
    except (ValueError, TypeError, ImportError, OSError, RecursionError) as exc:
        return jsonify({"error": str(exc)}), 400
    except HTTPException:
        raise
    except Exception:
        app.logger.exception("Prism analysis failed")
        return jsonify({"error": "Analysis failed. Check the server logs for details."}), 500
    finally:
        _unlink_quietly(path)
        _unlink_quietly(reference_path)
        ANALYSIS_SLOTS.release()
