"""Freezing: canonical hashing and reproducible payload construction.

The content hash covers the canonical JSON of (arrangement, solution, pins,
metrics) — everything that determines the manufacturing output, and nothing
else (no timestamps, no version ids). Recomputing a frozen version therefore
reproduces the exact same hash and the exact same SVG.
"""

from __future__ import annotations

import hashlib
import json

from . import engine
from .models import ArrangementRequest, Diagnostics, FreezeRequest, SolutionSpec
from .svg import render_unrolled_svg


class InfeasibleSolution(Exception):
    """Raised when a freeze request describes a layout that still conflicts."""

    def __init__(self, diagnostics: Diagnostics) -> None:
        super().__init__("solution is not manufacturable")
        self.diagnostics = diagnostics


def canonical_json(obj: object) -> str:
    """Deterministic JSON: sorted keys, tight separators, unicode kept."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_solution(arr: ArrangementRequest, sol: SolutionSpec) -> None:
    known = {n.id for n in arr.notes}
    locked = {n.id for n in arr.notes if n.locked}
    unknown = [i for i in sol.deleted_note_ids if i not in known]
    if unknown:
        raise ValueError(f"deleted_note_ids not present in the score: {sorted(unknown)}")
    locked_deleted = [i for i in sol.deleted_note_ids if i in locked]
    if locked_deleted:
        raise ValueError(f"locked notes must not be adjusted: {sorted(locked_deleted)}")


def build_payload(
    arr: ArrangementRequest, sol: SolutionSpec
) -> tuple[str, dict, dict]:
    """Compute a frozen version. Returns (content_hash, request_dict, result_dict)
    where result_dict contains pins, metrics and the printable SVG."""
    _validate_solution(arr, sol)
    placed, missing, kept = engine.place(arr, sol)
    diagnostics = engine.build_diagnostics(engine.diagnose(arr, placed, missing))
    if not diagnostics.ok:
        raise InfeasibleSolution(diagnostics)

    pins = engine.build_pins(arr, placed)
    metrics = engine.compute_metrics(arr, sol, kept, placed)

    request_dict = {
        "arrangement": arr.model_dump(mode="json"),
        "solution": sol.model_dump(mode="json"),
    }
    core = {
        **request_dict,
        "pins": [p.model_dump(mode="json") for p in pins],
        "metrics": metrics.model_dump(mode="json"),
    }
    content_hash = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()

    svg = render_unrolled_svg(
        content_hash=content_hash,
        req=arr,
        sol=sol,
        pins=pins,
        deleted_marks=engine.deleted_markers(arr, sol),
        metrics=metrics,
    )
    result_dict = {
        "pins": core["pins"],
        "metrics": core["metrics"],
        "svg": svg,
    }
    return content_hash, request_dict, result_dict


def recompute(request_dict: dict) -> tuple[str, dict]:
    """Rebuild a frozen version from its stored request; pure and deterministic."""
    freeze = FreezeRequest.model_validate(request_dict)
    content_hash, _, result_dict = build_payload(freeze.arrangement, freeze.solution)
    return content_hash, result_dict
