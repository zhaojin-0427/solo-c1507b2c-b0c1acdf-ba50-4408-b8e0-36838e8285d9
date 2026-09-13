"""Freezing of dynamics trials and chosen scenarios.

Two hash levels mirror the pin-version / balance-plan split:

- a *trial* hashes (source content hash, spec, derived, pins): the immutable
  powertrain dataset referencing a frozen pin-arrangement version;
- a *plan* hashes (source snapshot, input curves, integration step, resolved
  scenario, full simulation result): a selected operating point whose
  recomputation is self-contained and bit-identical.

No timestamps or database ids enter either hash.
"""

from __future__ import annotations

import hashlib

from . import dynamics_engine as de
from .dynamics_models import (
    DynamicsSpec,
    ScenarioParams,
)
from .freeze import canonical_json


def build_trial_payload(
    source: de.DynamicsSource, spec: DynamicsSpec
) -> tuple[str, dict, dict]:
    """Validate the spec against the source and build the immutable trial."""
    de.validate_spec(spec, source)
    derived = de.derived_info(spec, source)
    request_dict = {
        "source_version_id": source.version_id,
        "spec": spec.model_dump(mode="json"),
    }
    core = {
        "source_content_hash": source.content_hash,
        "spec": request_dict["spec"],
        "derived": derived,
        "pins": [p.model_dump(mode="json") for p in source.pins],
    }
    content_hash = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    result_dict = {
        "derived": derived,
        "pins": core["pins"],
        "source_content_hash": source.content_hash,
    }
    return content_hash, request_dict, result_dict


def build_plan_payload(
    source: de.DynamicsSource, spec: DynamicsSpec, scenario: ScenarioParams
) -> tuple[str, dict, dict]:
    """Run the selected scenario and freeze the complete result. Raises
    DynamicsError if the scenario is infeasible (e.g. prewind beyond travel)."""
    de.validate_spec(spec, source)
    result = de.simulate(spec, source, scenario, keep_curves=True)
    request_dict = {
        "source": source.snapshot_dict(),
        "spec": spec.model_dump(mode="json"),
        "scenario": scenario.model_dump(mode="json", exclude_none=True),
    }
    result_dict = result.model_dump(mode="json")
    core = {
        "source_content_hash": source.content_hash,
        "source": request_dict["source"],
        "spec": request_dict["spec"],
        "dt_s": spec.dt_s,
        "scenario": result_dict["scenario"],
        "result": result_dict,
    }
    content_hash = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    return content_hash, request_dict, result_dict


def recompute_plan(request_dict: dict) -> tuple[str, dict]:
    """Rebuild a frozen plan from its stored source snapshot; pure."""
    source = de.DynamicsSource.from_snapshot_dict(request_dict["source"])
    spec = DynamicsSpec.model_validate(request_dict["spec"])
    scenario = ScenarioParams.model_validate(request_dict["scenario"])
    content_hash, _, result_dict = build_plan_payload(source, spec, scenario)
    return content_hash, result_dict
