"""
Backward-compatible dataset loader imports.

Deprecated: use framevitals.loader instead.
"""

from framevitals.loader import (
    UPLOAD_DIR,
    load_dataset,
    save_uploaded_file,
)


__all__ = [
    "UPLOAD_DIR",
    "load_dataset",
    "save_uploaded_file",
]
