"""Freezing of balance plans: canonical hashing and payload construction.

The content hash covers the canonical JSON of (source content hash, spec,
locked weights, chosen weights, resolved pin contributions, final imbalance)
— everything that determines the balancing output, and nothing else (no
timestamps, no plan ids, no database-local source ids). The stored request
carries a full snapshot of the source version, so recomputing a frozen plan
is self-contained and reproduces the exact same hash and SVG.
"""

from __future__ import annotations

import hashlib

from . import balance_engine as be
from .balance_models import (
    BalanceFreezeRequest,
    BalanceSpec,
    WeightRef,
)
from .freeze import canonical_json
from .svg import render_balance_svg


def build_plan_payload(
    source: be.SourceInfo,
    spec: BalanceSpec,
    locked: list[WeightRef],
    weights: list[WeightRef],
) -> tuple[str, dict, dict]:
    """Compute a frozen balance plan. Returns (content_hash, request_dict,
    result_dict); result_dict holds baseline/final imbalance, per-pin
    contributions and the SVG. Raises BalancePlanError on invalid input."""
    be.validate_spec(spec, source)
    be.validate_weights(spec, source, locked, weights)

    pins = be.pin_points(spec, source)
    locked_pts = be.weight_points(spec, source, locked, "locked")
    chosen_pts = be.weight_points(spec, source, weights, "weight")

    baseline = be.compute_imbalance(spec, source, pins + locked_pts)
    final = be.compute_imbalance(spec, source, pins + locked_pts + chosen_pts)
    contributions = be.pin_contributions(spec, source, pins)

    request_dict = {
        "source": source.snapshot_dict(),
        "spec": spec.model_dump(mode="json"),
        "locked_weights": [w.model_dump(mode="json") for w in locked],
        "weights": [w.model_dump(mode="json") for w in weights],
    }
    core = {
        "source_content_hash": source.content_hash,
        "spec": request_dict["spec"],
        "locked_weights": request_dict["locked_weights"],
        "weights": request_dict["weights"],
        "pins": [c.model_dump(mode="json") for c in contributions],
        "imbalance": final.model_dump(mode="json"),
    }
    content_hash = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()

    svg = render_balance_svg(
        content_hash=content_hash,
        source=source,
        spec=spec,
        locked=be.weight_points(spec, source, locked, "locked"),
        weights=be.weight_points(spec, source, weights, "weight"),
        weight_diameters={t.id: t.diameter_mm for t in spec.inventory},
        imbalance=final,
    )
    result_dict = {
        "baseline": baseline.model_dump(mode="json"),
        "imbalance": core["imbalance"],
        "pins": core["pins"],
        "svg": svg,
    }
    return content_hash, request_dict, result_dict


def recompute_plan(request_dict: dict) -> tuple[str, dict]:
    """Rebuild a frozen plan from its stored snapshot; pure and deterministic."""
    req = BalanceFreezeRequest.model_validate(
        {
            "source_version_id": request_dict["source"]["version_id"],
            "spec": request_dict["spec"],
            "locked_weights": request_dict["locked_weights"],
            "weights": request_dict["weights"],
        }
    )
    source = be.SourceInfo.from_snapshot_dict(request_dict["source"])
    content_hash, _, result_dict = build_plan_payload(
        source, req.spec, req.locked_weights, req.weights
    )
    return content_hash, result_dict
