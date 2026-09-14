"""Freezing of calibration batches and confirmed calibration plans.

Two hash levels mirror the trial / plan split of the dynamics service:

- a *batch* hashes (source content hash, measurements, axis fit, baseline
  analysis): the immutable 采集中 dataset referencing a frozen
  pin-arrangement version;
- a *plan* hashes (source snapshot, measurements, search limits, resolved
  adjustment, axis fit, resulting analysis): the 已确认 operating point whose
  recomputation is self-contained and bit-identical.

No timestamps or database ids enter either hash.
"""

from __future__ import annotations

import hashlib

from . import calibration_engine as ce
from .calibration_models import (
    Adjustment,
    CalibrationSearchLimits,
    CalibrationSpec,
)
from .freeze import canonical_json

ZERO = Adjustment()


def build_batch_payload(
    source: ce.CalSource, spec: CalibrationSpec
) -> tuple[str, dict, dict]:
    """Validate the measurements against the source and build the immutable
    batch: axis fit plus the baseline (zero-adjustment) per-pin analysis."""
    ce.validate_spec(spec, source)
    model = ce.fit_axis_model(spec)
    fit = ce.axis_fit_out(model, spec)
    analysis = ce.analyze(spec, source, model, ZERO)

    request_dict = {
        "source_version_id": source.version_id,
        "spec": spec.model_dump(mode="json"),
    }
    core = {
        "source_content_hash": source.content_hash,
        "spec": request_dict["spec"],
        "fit": fit.model_dump(mode="json"),
        "pins": [p.model_dump(mode="json") for p in analysis.pins],
        "diagnostics": analysis.diagnostics.model_dump(mode="json"),
    }
    content_hash = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    result_dict = {
        "source": source.snapshot_dict(),
        "source_content_hash": source.content_hash,
        "fit": core["fit"],
        "pins": core["pins"],
        "diagnostics": core["diagnostics"],
        "summary": analysis.summary.model_dump(mode="json"),
    }
    return content_hash, request_dict, result_dict


def build_plan_payload(
    source: ce.CalSource,
    spec: CalibrationSpec,
    limits: CalibrationSearchLimits,
    adjustment: Adjustment,
) -> tuple[str, dict, dict]:
    """Freeze a confirmed adjustment. Raises CalibrationError if the
    adjustment violates the locks/ranges or a shim total is not achievable."""
    ce.validate_spec(spec, source)
    resolved = ce.resolve_adjustment(limits, adjustment)
    snapped = ce.snap_adjustment(resolved)
    model = ce.fit_axis_model(spec)
    fit = ce.axis_fit_out(model, spec)
    analysis = ce.analyze(spec, source, model, snapped)

    request_dict = {
        "source": source.snapshot_dict(),
        "spec": spec.model_dump(mode="json"),
        "limits": limits.model_dump(mode="json"),
        "adjustment": snapped.model_dump(mode="json"),
    }
    core = {
        "source_content_hash": source.content_hash,
        "source": request_dict["source"],
        "spec": request_dict["spec"],
        "limits": request_dict["limits"],
        "adjustment": request_dict["adjustment"],
        "fit": fit.model_dump(mode="json"),
        "pins": [p.model_dump(mode="json") for p in analysis.pins],
        "diagnostics": analysis.diagnostics.model_dump(mode="json"),
    }
    content_hash = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    result_dict = {
        "fit": core["fit"],
        "adjustment": resolved.model_dump(mode="json"),
        "pins": core["pins"],
        "diagnostics": core["diagnostics"],
        "summary": analysis.summary.model_dump(mode="json"),
    }
    return content_hash, request_dict, result_dict


def recompute_plan(request_dict: dict) -> tuple[str, dict]:
    """Rebuild a frozen plan from its stored source snapshot; pure."""
    source = ce.CalSource.from_snapshot_dict(request_dict["source"])
    spec = CalibrationSpec.model_validate(request_dict["spec"])
    limits = CalibrationSearchLimits.model_validate(request_dict["limits"])
    adjustment = Adjustment.model_validate(request_dict["adjustment"])
    content_hash, _, result_dict = build_plan_payload(source, spec, limits, adjustment)
    return content_hash, result_dict
