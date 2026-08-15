"""
Analysis Selector
==================
Signal-driven engine that decides which analyses to run, skip, or recommend.
No hardcoded domain logic — decisions are purely based on signals + inventory rules.
"""

from framevitals.analysis_inventory import ANALYSIS_INVENTORY


def _mode_allowed(analysis, mode):
    return mode in analysis.get("modes", [])


def _check_requires(analysis, signals):
    for key, expected in analysis.get("requires", {}).items():
        if signals.get(key) != expected:
            return False, f"Requires {key}={expected}, got {signals.get(key)}."
    min_rows = analysis.get("min_rows")
    if min_rows and signals.get("row_count", 0) < min_rows:
        return False, f"Requires at least {min_rows} rows."
    return True, "OK"


def _check_requires_any(analysis, signals):
    any_reqs = analysis.get("requires_any")
    if not any_reqs:
        return True, "No any-requirements."
    for group in any_reqs:
        if all(signals.get(k) == v for k, v in group.items()):
            return True, "Matched."
    return False, "No any-requirement group matched."


def select_analyses(signals, analysis_mode="standard", target_column=None):
    selected, skipped, recommended = [], [], []

    for a in ANALYSIS_INVENTORY:
        if not _mode_allowed(a, analysis_mode):
            skipped.append({"id": a["id"], "name": a["name"], "category": a["category"],
                            "reason": f"Not in {analysis_mode} mode."})
            continue

        ok, reason = _check_requires(a, signals)
        if not ok:
            skipped.append({"id": a["id"], "name": a["name"], "category": a["category"], "reason": reason})
            continue

        ok2, reason2 = _check_requires_any(a, signals)
        if not ok2:
            skipped.append({"id": a["id"], "name": a["name"], "category": a["category"], "reason": reason2})
            continue

        if a.get("requires_user_target") and not target_column:
            recommended.append({"id": a["id"], "name": a["name"], "category": a["category"],
                                "reason": "Recommended after selecting a target column."})
            continue

        selected.append({"id": a["id"], "name": a["name"], "category": a["category"],
                         "priority": a["priority"], "outputs": a["outputs"]})

    return {
        "selected_analyses": selected,
        "skipped_analyses": skipped,
        "recommended_analyses": recommended,
        "summary": {"selected_count": len(selected), "skipped_count": len(skipped),
                     "recommended_count": len(recommended)},
    }
