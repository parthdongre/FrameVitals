"""Core structured-source primitives used by FrameVitals protocols."""

from framevitals.core.source import SourceDescriptor, SourceKind, recognize_source
from framevitals.core.beacons import beacon

__all__ = ["SourceDescriptor", "SourceKind", "recognize_source", "beacon"]
