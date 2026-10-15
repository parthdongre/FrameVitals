# Python module inventory

Generated from the actual source AST after Phase 1. “Consumers” includes direct package imports, including lazy function imports; dynamic plugins and third-party callers are not represented. This is an integration map, not a proof of dead code.

| Module | Responsibility / entry points | Direct internal consumers |
| --- | --- | --- |
| `__init__` | FrameVitals public package interface.; capabilities, inspect_source, profile, roles, health, ml_readiness, quality, statistics, anomalies | Public entry, framework callback, or indirect use; verify before removal |
| `__main__` | Run FrameVitals directly with python -m framevitals.;  | Public entry, framework callback, or indirect use; verify before removal |
| `acceleration` | Hardware and optional acceleration discovery for FrameVitals.; GpuDevice, SystemCapabilities, detect_system_capabilities, system_info | __init__, cli |
| `advanced_indicators` | calculate_column_utility, calculate_anomalies, detect_fairness_review, calculate_freshness, calculate_advanced_indicators | pipeline, streaming_bounded_pipeline |
| `agent_brief` | Dataset brief packer.; build_dataset_brief, render_brief_block | agent_tools, ai_agent |
| `agent_tools` | Agent Tool Registry (WS-9); AgentContext, list_tools, run_tool | ai_agent |
| `ai_agent` | FrameVitals — Agentic AI Layer (WS-9); ToolCall, Plan, Verdict, answer_with_agent | webapp |
| `ai_insights` | Optional AI interpretation with deterministic local fallbacks.; compact_context, generate_ai_report, answer_dataset_question | pipeline, webapp |
| `analysis.__init__` | Structured analysis engines used by protocol dispatch.;  | Public entry, framework callback, or indirect use; verify before removal |
| `analysis.architectures` | Architecture-aware diagnostics for deep-learning models.; analyze_transformer_architecture, analyze_recurrent_architecture, analyze_architecture | analysis.model |
| `analysis.change` | Structured Tide comparisons for graphs, tensors, and neural models.; compare_tensors, compare_graphs, compare_models | structured_analysis |
| `analysis.document` | Bounded, non-executing inspection of common document file types.; inspect_document, analyze_document, compare_documents | analysis.structured_contracts, structured_analysis |
| `analysis.graph` | Bounded graph diagnostics for NetworkX-compatible graph objects.; analyze_graph | analysis.change, analysis.structured_contracts, structured_analysis |
| `analysis.model` | Static and lightweight runtime diagnostics for PyTorch-style models.; analyze_model | structured_analysis |
| `analysis.nested` | Bounded diagnostics for nested Python/JSON-like structures.; inspect_nested, analyze_nested, compare_nested | analysis.structured_contracts, structured_analysis |
| `analysis.onnx_model` | ONNX model diagnostics built on optional onnx dependency.; inspect_onnx, analyze_onnx, compare_onnx | analysis.structured_contracts, structured_analysis |
| `analysis.optimizer` | Optimizer-state diagnostics for PyTorch-style optimizers.; inspect_optimizer | analysis.model |
| `analysis.relational` | Diagnostics for related collections of tabular datasets.; inspect_relational, analyze_relational, compare_relational | analysis.structured_contracts, structured_analysis |
| `analysis.runtime_model` | Bounded runtime observation for PyTorch-style neural networks.; observe_model_runtime | analysis.model |
| `analysis.safetensors` | Metadata-first diagnostics for local Safetensors checkpoint files.; inspect_safetensors, analyze_safetensors, compare_safetensors | analysis.structured_contracts, structured_analysis |
| `analysis.structured_contracts` | Axiom contracts for non-tabular structured sources.; infer_structured_contract, validate_structured | protocols |
| `analysis.tensor` | Tensor diagnostics shared by NumPy arrays and framework tensor adapters.; tensor_metrics, analyze_tensor | analysis.change, analysis.model, analysis.structured_contracts, structured_analysis |
| `analysis_api` | Public full-analysis dispatcher.; analyze | __init__, api, cli, protocols |
| `analysis_inventory` | Analysis Inventory;  | analysis_selector |
| `analysis_selector` | Analysis Selector; select_analyses | pipeline, planner, streaming_pipeline |
| `analysis_state` | Mergeable analysis-state primitives.; NumericColumnState, AnalysisState | streaming_profile |
| `anomaly_ensemble` | Multi-detector anomaly scoring for tabular data.; detect_anomalies_ensemble | budgeted_analysis |
| `api` | Backward-compatible public API facade.; inspect_source, profile, roles, health, ml_readiness, quality, statistics, anomalies, relationships | Public entry, framework callback, or indirect use; verify before removal |
| `backends` | Backend routing for FrameVitals native and NumPy kernels.; native_available, resolve_numeric_backend, numeric_state, numeric_profile, create_numeric_accumulator, create_arrow_batch_profile_accumulator, create_string_accumulator, backend_status | analysis_state, profiler, streaming_profile |
| `baseline_model` | classification_metrics, regression_metrics, run_baseline_model | Public entry, framework callback, or indirect use; verify before removal |
| `budgeted_analysis` | Resource-bounded adapters around expensive analysis modules.; run_budgeted_deep_statistics, run_budgeted_anomalies, run_budgeted_time_series | focused, pipeline, streaming_bounded_pipeline |
| `capabilities` | Authoritative source capabilities shared by Python, terminal and HTTP.; capabilities | __init__, cli, protocols, terminal_ui, webapp |
| `chart_planner` | Chart Planner; build_chart_plan | visualizer |
| `checks` | Extensible user-defined checks for FrameVitals quality gates.; DataCheck, check, run_checks | __init__, api, operations, plugins |
| `cleaner` | create_cleaned_dataset | pipeline |
| `cleaning_plan` | Explicit, conservative cleaning plans for FrameVitals datasets.; CleaningPlan, infer_cleaning_plan, apply_cleaning_plan, simulate_cleaning_plan | __init__, api, cleaner, operations |
| `cli` | build_parser, main | __main__, cli_entry |
| `cli_entry` | Backward-compatible CLI entrypoint.; main | Public entry, framework callback, or indirect use; verify before removal |
| `column_roles` | Column Role Inference Engine; infer_column_roles, get_columns_with_role, get_meaningful_numeric_columns, get_meaningful_categorical_columns, summarize_roles | chart_planner, dataset_signals, focused, pipeline, planning_api, quality_diagnostics, streaming_pipeline, streaming_quality, streaming_roles, streaming_target, target_intelligence, visualizer |
| `config` | Runtime configuration for FrameVitals analysis.; AnalysisConfig, available_presets, available_modules, resolve_config, with_overrides | __init__, analysis_api, cli, execution_context, pipeline, planner, planning_api, streaming_bounded_pipeline, streaming_pipeline |
| `contracts` | Inference and validation helpers for reusable data contracts.; infer_contract, validate_contract | operations |
| `core.__init__` | Core structured-source primitives used by FrameVitals protocols.;  | Public entry, framework callback, or indirect use; verify before removal |
| `core.beacons` | Small common finding schema shared by structured-data analyzers.; beacon | analysis.architectures, analysis.document, analysis.graph, analysis.model, analysis.nested, analysis.onnx_model, analysis.optimizer, analysis.relational, analysis.runtime_model, analysis.safetensors, analysis.tensor, core.__init__ |
| `core.source` | Recognition of structured sources without importing heavyweight frameworks.; SourceKind, SourceDescriptor, recognize_source | __init__, analysis.structured_contracts, api, capabilities, core.__init__, protocols, structured_analysis |
| `dataset_signals` | Dataset Signal Detector; detect_dataset_signals | pipeline, planning_api, streaming_pipeline |
| `deep_statistics` | safe_float, classify_skewness, classify_kurtosis, normality_test, numeric_deep_statistics, correlation_insights, classify_correlation, categorical_deep_statistics, chi_square_relationships | Public entry, framework callback, or indirect use; verify before removal |
| `deep_statistics_v2` | Deep Statistics v2 (WS-1); run_deep_statistics_v2 | budgeted_analysis, fast_deep_statistics, streaming_exact_reuse |
| `deep_triage` | Cheap column triage for expensive deep-analysis routines.; DeepTriageResult, triage_deep_columns | budgeted_analysis |
| `drift_analysis` | Dataset drift and change analysis.; severity_at_least, compare_datasets, split_by_date | cli, operations, webapp |
| `duckdb_source` | Optional DuckDB relation adapter for FrameVitals source-aware execution.; DuckDBRelationSource, resolve_duckdb_source | sources |
| `execution` | Adaptive execution budgets for FrameVitals analyses.; ExecutionPolicy, current_execution_policy, use_execution_policy, ExecutionBudget, derive_streaming_profile_column_limit, derive_execution_budget, deterministic_sample_frame | analysis_api, budgeted_analysis, execution_context, focused, pipeline, planning_api, profiler, streaming_bounded_pipeline, streaming_pipeline, streaming_profile, streaming_quality, streaming_target |
| `execution_context` | Per-run reusable execution context for FrameVitals analyses.; AnalysisContext | planning_api |
| `explainability` | Explainability (WS-4); explain_winner | pipeline, streaming_bounded_pipeline |
| `fast_anomaly` | Fast bounded anomaly screening for standard and deep analysis modes.; fast_anomaly_scan | budgeted_analysis |
| `fast_deep_statistics` | Fast deep-statistics execution for non-research analysis modes.; fast_mean_ci, fast_median_ci, run_fast_deep_statistics_v2 | budgeted_analysis |
| `feature_importance` | collapse_onehot_importances, calculate_mutual_information, run_feature_importance | Public entry, framework callback, or indirect use; verify before removal |
| `file_formats` | Single source of truth for CLI and terminal-supported file formats.; FormatSpec, format_for, supported_extensions, is_available, check_file_format, format_catalog, render_formats, prepare_file_source | capabilities, cli, core.source, loader, security, structured_analysis, terminal_ui, webapp |
| `findings` | Normalized findings for FrameVitals analysis results.; normalize_severity, findings_from_signals, findings_from_target_intelligence, findings_from_quality_diagnostics, merge_findings, recommendations_from_findings | result |
| `focused` | Focused FrameVitals analysis entry points.; profile, roles, health, ml_readiness, quality, statistics, anomalies, relationships, target_analysis | __init__, api, ml_readiness |
| `frontend_api` | format_bytes, format_elapsed_label, build_dashboard_payload | webapp |
| `health_score` | calculate_outlier_percent, calculate_health_score_from_profile_sample, calculate_health_score | cleaner, cleaning_plan, focused, pipeline, streaming_pipeline |
| `input_safety` | Parser preflight limits; no extraction or executable deserialization.; validate_input | analysis.document, analysis.graph, file_formats, loader |
| `loader` | save_uploaded_file, load_dataset | pipeline, sources, webapp |
| `ml_preprocessing` | Unified ML preprocessing used by every FrameVitals model workflow.; prepare_ml_matrix, build_sklearn_preprocessor, get_transformed_feature_names | baseline_model, explainability, feature_importance, model_leaderboard, predictive_diagnostics |
| `ml_readiness` | ML-readiness scoring and compatibility helpers.; calculate_ml_readiness_from_profile, calculate_ml_readiness | focused, pipeline, streaming_pipeline |
| `model_diagnostics` | Compatibility shim for legacy tabular predictive diagnostics.;  | Public entry, framework callback, or indirect use; verify before removal |
| `model_leaderboard` | Cross-validated baseline model leaderboard for FrameVitals.; run_model_leaderboard | pipeline, streaming_bounded_pipeline |
| `multicollinearity` | prepare_numeric_matrix, calculate_vif_scores, detect_redundant_feature_groups, run_multicollinearity_analysis | Public entry, framework callback, or indirect use; verify before removal |
| `neural_anomaly` | Small neural reconstruction detector for research-mode anomaly analysis.; neural_reconstruction_anomalies | budgeted_analysis |
| `operations` | Lightweight public data operations that do not require the full pipeline.; plan_cleaning, clean, compare, infer_contract, validate, gate | __init__, api, cli, protocols, webapp |
| `pdf_report_builder` | PDF Report Builder (v3 — editorial dark); safe_text, fmt_number, resolve_chart_path, Composer, generate_pdf_report | report_generator |
| `pipeline` | FrameVitals analysis pipeline.; run_full_analysis | analysis_api, streaming_bounded_pipeline, webapp |
| `planner` | Declarative execution planning for FrameVitals.; ModuleRule, effective_disabled_modules, plan_execution_modules, build_execution_plan | analysis_api, planning_api, webapp |
| `planning` | Analysis planning result objects and human-readable explanations.; AnalysisPlan | __init__, api, planning_api |
| `planning_api` | Planning-only public execution path.; plan | __init__, api, cli |
| `plugins` | Opt-in discovery for third-party FrameVitals extensions.; discover_checks | __init__, api |
| `predictive_diagnostics` | run_regression_diagnostics, run_classification_diagnostics, run_predictive_diagnostics | model_diagnostics |
| `presentation` | Bounded visualization data, without new analytical calculations.; graph_excerpt | analysis.graph |
| `profiler` | clean_value, series_to_dict, detect_column_types, build_profile | cleaner, cleaning_plan, focused, operations, pipeline, planning_api, quality_diagnostics, streaming_profile, visualizer |
| `protocols` | Branded workflow protocols for the public FrameVitals experience.; PrismResult, AxiomResult, ForgeResult, prism, axiom, tide, forge, pulse | __init__, api, cli, terminal_ui, webapp |
| `provenance` | Shared execution-provenance helpers for public FrameVitals results.; load_fully_materializes, execution_provenance, normalize_execution | budgeted_analysis, checks, focused, health_score, ml_readiness, operations, streaming_quality, streaming_roles |
| `quality_diagnostics` | Deterministic data-quality diagnostics for tabular datasets.; run_quality_diagnostics | focused, pipeline, streaming_quality |
| `quality_results` | Dict-compatible public results for checks, validation, drift, and quality gates.; CheckResult, ValidationResult, DriftResult, GateResult | __init__, analysis.change, analysis.document, analysis.nested, analysis.onnx_model, analysis.relational, analysis.safetensors, analysis.structured_contracts, api, checks, operations, protocols |
| `rag_index` | RAG Fact Index; Fact, build_fact_index, retrieve, render_facts_block | agent_tools, ai_agent |
| `relationship_graph` | Sparse feature-relationship discovery for wide numeric datasets.; build_numeric_relationship_graph | focused |
| `report_generator` | generate_pdf_report | webapp |
| `reporting.__init__` | Human-facing renderers for FrameVitals results.;  | Public entry, framework callback, or indirect use; verify before removal |
| `reporting.html` | Self-contained HTML rendering for FrameVitals analysis results.; render_notebook_summary, render_html_report | reporting.__init__, result |
| `reporting.terminal` | Dependency-light terminal rendering for FrameVitals results.; render_terminal_summary | reporting.__init__, result |
| `result` | Public result objects for FrameVitals.; ColumnResult, DiagnosticResult, AnalysisResult | __init__, analysis.document, analysis.graph, analysis.model, analysis.nested, analysis.onnx_model, analysis.relational, analysis.safetensors, analysis.tensor, analysis_api, api, focused, protocols, structured_analysis |
| `result_contract` | Additive v1 result envelope. Never infer success, exactness, or zero from absence.; build_envelope | protocols, result, webapp |
| `safe_pandas` | Safe, read-only pandas expression evaluator for the optional agent layer.; UnsafeExpression, safe_eval | agent_tools |
| `security` | validate_file, make_safe_filename, sanitize_csv_value | cleaner, cli, loader |
| `segment_analysis` | choose_segment_columns, run_segment_analysis | Public entry, framework callback, or indirect use; verify before removal |
| `semantic_types` | Deterministic semantic type inference for text-like columns.; infer_semantic_types | column_roles |
| `serialization` | Strict, shared JSON conversion for results and transport boundaries.; json_safe, dumps | cli, protocols, quality_results, result, result_contract, snapshots, terminal_ui, webapp |
| `signal_engine` | Signal Engine (Display Layer); severity_from_percent, build_signals | pipeline, streaming_pipeline |
| `snapshots` | Compact versioned snapshots and lightweight local monitoring history.; AnalysisSnapshot, create_snapshot, load_snapshot, compare_snapshots, SnapshotHistory | __init__, cli, protocols, result |
| `sources` | Dataset source abstractions for FrameVitals.; DatasetMetadata, DatasetSource, StreamingDatasetSource, PandasSource, ArrowTableSource, FileSource, DelimitedTextSource, ParquetSource, resolve_source | __init__, analysis_api, api, checks, cli, duckdb_source, focused, operations, planning_api, streaming_pipeline, streaming_profile, streaming_target |
| `stream_change` | Low-overhead change detection for streaming and ordered numeric data.; PageHinkleyMeanShift, scan_ordered_mean_shift | budgeted_analysis |
| `streaming_bounded_pipeline` | Bounded module scheduler for streaming analyses.; run_streaming_bounded_modules | streaming_pipeline |
| `streaming_exact_reuse` | Reuse full-stream sufficient statistics in bounded downstream analyses.; reuse_streaming_exact_statistics | analysis_api |
| `streaming_pipeline` | Full FrameVitals orchestration for streaming dataset sources.; run_streaming_analysis | analysis_api |
| `streaming_profile` | Streaming profile construction for Arrow-capable dataset sources.; numeric_columns_for_streaming_source, sample_streaming_source, build_streaming_profile | focused, operations, streaming_pipeline, streaming_target |
| `streaming_quality` | Quality diagnostics for streaming dataset sources.; run_streaming_quality_diagnostics | focused, streaming_pipeline |
| `streaming_roles` | Column-role inference for streaming dataset sources.; infer_streaming_column_roles | focused, streaming_pipeline |
| `streaming_sketches` | Pure-NumPy bounded-memory sketches for non-native streaming execution.; should_use_full_stream_numpy_sketch, NumpyLogQuantileSketch | streaming_profile |
| `streaming_target` | Target-aware diagnostics for streaming dataset sources.; run_streaming_target_analysis | focused |
| `structured_analysis` | Protocol-level dispatcher for non-tabular structured sources.; normalize_structured_input, analyze_structured, compare_structured | protocols |
| `target_analyzer` | detect_task_type, analyze_classification_target, analyze_regression_target, analyze_target | target_intelligence |
| `target_intelligence` | Unified target-aware diagnostics for supervised-learning datasets.; run_target_intelligence | focused, pipeline, streaming_bounded_pipeline, streaming_target |
| `target_leakage` | same_ratio_non_missing, numeric_correlation, categorical_mapping_purity, classify_target_leakage_risk, run_target_leakage_analysis | target_intelligence |
| `terminal_ui` | Local, terminal-first FrameVitals interface.; TerminalReport, TerminalState, run_diagnostic, save_report, view_lines, launch_terminal | cli |
| `text_profile` | Text / NLP Profiling (WS-6); detect_text_columns, profile_text_columns | pipeline, streaming_bounded_pipeline |
| `time_series` | Time-Series Auto-Detection (WS-5); detect_date_column, detect_and_analyze_time_series | budgeted_analysis |
| `visualizer` | Visualizer (v3 — editorial dark); generate_charts | pipeline |
| `webapp` | FrameVitals Flask API and generated-artifact server.; safe_jsonify, index, analyze, api_analyze, ask, api_ask, api_ai_report, api_health, api_compare | cli |
