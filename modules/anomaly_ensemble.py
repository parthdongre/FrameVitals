"""
Backward-compatible anomaly detection imports.

Deprecated: use framevitals.anomaly_ensemble instead.
"""

from framevitals.anomaly_ensemble import (
    detect_anomalies_ensemble,
)

__all__ = [
    "detect_anomalies_ensemble",
]
