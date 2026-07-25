"""Empty annotation scaffold for later post-hoc coding phases (assumption archaeology,
question categorization). Phase 0 only produces the empty container.
"""
from __future__ import annotations

from harness.record.schema import Annotations


def empty_annotations() -> Annotations:
    return Annotations()
