"""Pydantic request/response schemas for the powertrain dynamics trial service.

A dynamics trial references a *frozen* pin-arrangement version and stores the
powertrain data independently: the sampled mainspring torque curve, the gear
train (stage ratios and efficiencies), the rotating inertia, the governor
drag curve and the single-pluck energy of every comb reed.

Unit convention (inputs carry explicit unit strings and are converted to SI
inside the engine):

- torque:    mN*m  (millinewton-metre, 1e-3 N*m)
- energy:    uJ    (microjoule, 1e-6 J)
- inertia:   g*cm^2 (1e-7 kg*m^2)
- angles:    rad internally; curve abscissae use "turns" or "rpm"
- time:      seconds
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class SampledCurve(BaseModel):
    """A piecewise-linear sampled curve. Abscissae must be strictly increasing
    — a non-increasing x series makes the lookup ambiguous and is rejected."""

    x: list[float] = Field(min_length=2)
    y: list[float] = Field(min_length=2)
    x_unit: Literal["turns", "rpm"]
    y_unit: Literal["mN*m"]

    @model_validator(mode="after")
    def _checks(self) -> "SampledCurve":
        if len(self.x) != len(self.y):
            raise ValueError("curve x and y must have equal length")
        for a, b in zip(self.x, self.x[1:]):
            if not b > a:
                raise ValueError(
                    "curve abscissae must be strictly increasing "
                    f"(got {a} followed by {b})"
                )
        for v in self.y:
            if v < 0:
                raise ValueError("curve ordinates must be non-negative")
        return self


class GearStage(BaseModel):
    """One gear stage on the spring-barrel -> cylinder path.

    ``ratio`` is output speed over input speed (cylinder side over barrel
    side); the total ratio is the product of all stages."""

    ratio: float = Field(gt=0, le=10000.0)
    efficiency: float = Field(gt=0, le=1.0)


class DynamicsSpec(BaseModel):
    """Powertrain parameters, stored independently of the pin version."""

    spring_torque: SampledCurve = Field(
        description=(
            "mainspring torque vs released barrel turns (x_unit=turns): "
            "x[0] is the fully-wound reference (0 turns released), x[-1] the "
            "fully-run-down end; the span x[-1]-x[0] is the usable travel"
        )
    )
    gear_train: list[GearStage] = Field(min_length=1)
    base_inertia_g_cm2: float = Field(
        gt=0, le=1e9, description="inertia of the rotating assembly on the cylinder shaft"
    )
    governor_drag: SampledCurve = Field(
        description="governor resistance torque vs cylinder rpm (x_unit=rpm)"
    )
    pluck_energy_uJ: dict[int, float] = Field(
        description="single-pluck energy per comb reed, keyed by MIDI pitch"
    )
    dt_s: float = Field(
        0.001, gt=5e-5, le=0.02, description="fixed integration time step"
    )
    stall_rpm_ratio: float = Field(
        0.6,
        gt=0.0,
        lt=1.0,
        description="stall threshold as a fraction of the design rpm",
    )
    overspeed_rpm_ratio: float = Field(
        1.25, gt=1.0, le=3.0, description="overspeed threshold as a fraction of design rpm"
    )
    beat_drift_limit: float = Field(
        0.1,
        gt=0.0,
        le=1.0,
        description="maximum allowed |actual-design|/design interval between plucks",
    )

    @model_validator(mode="after")
    def _cross_checks(self) -> "DynamicsSpec":
        if self.spring_torque.x_unit != "turns":
            raise ValueError("spring_torque curve abscissa unit must be 'turns'")
        if self.governor_drag.x_unit != "rpm":
            raise ValueError("governor_drag curve abscissa unit must be 'rpm'")
        for pitch, energy in self.pluck_energy_uJ.items():
            if not 0 <= pitch <= 127:
                raise ValueError(f"pluck energy pitch {pitch} outside MIDI range")
            if energy <= 0:
                raise ValueError(f"pluck energy for pitch {pitch} must be positive")
        return self


class TrialCreateRequest(BaseModel):
    source_version_id: int = Field(ge=1)
    spec: DynamicsSpec


class ScenarioParams(BaseModel):
    """One tunable operating point for a trial simulation."""

    prewind_turns: float = Field(gt=0, le=100000.0, description="wound reserve at run start")
    governor_coefficient: float = Field(
        1.0, ge=0.0, le=100.0, description="multiplier on the sampled governor drag curve"
    )
    flywheel_inertia_g_cm2: float = Field(
        0.0, ge=0.0, le=1e9, description="added flywheel inertia reflected to the cylinder shaft"
    )
    gear_ratio: Optional[float] = Field(
        None, gt=0.0, le=10000.0,
        description="overrides the stored total gear ratio; None keeps the stored train",
    )


class ScenarioSearchRequest(BaseModel):
    """Candidate grids for the scenario search. Callers may lock the gear
    ratio (only that value is used); otherwise ratio candidates are taken from
    ``gear_ratio_candidates`` (defaulting to the stored total ratio)."""

    gear_ratio_lock: Optional[float] = Field(None, gt=0.0, le=10000.0)
    gear_ratio_candidates: list[float] = Field(default_factory=list)
    prewind_turns: list[float] = Field(min_length=1, max_length=50)
    governor_coefficients: list[float] = Field(default_factory=lambda: [1.0], min_length=1)
    flywheel_inertia_g_cm2: list[float] = Field(default_factory=lambda: [0.0], min_length=1)
    max_candidates: int = Field(10, ge=1, le=20)

    @model_validator(mode="after")
    def _checks(self) -> "ScenarioSearchRequest":
        # Candidate grids share the single-scenario bounds, so an out-of-range
        # candidate is rejected at the request (422), not deep in the search.
        for v in self.gear_ratio_candidates:
            if not 0.0 < v <= 10000.0:
                raise ValueError(
                    f"gear_ratio_candidates must be within (0, 10000], got {v}"
                )
        for v in self.prewind_turns:
            if not 0.0 < v <= 100000.0:
                raise ValueError(
                    f"prewind_turns must be within (0, 100000], got {v}"
                )
        for v in self.governor_coefficients:
            if not 0.0 <= v <= 100.0:
                raise ValueError(
                    f"governor_coefficients must be within [0, 100], got {v}"
                )
        for v in self.flywheel_inertia_g_cm2:
            if not 0.0 <= v <= 1e9:
                raise ValueError(
                    f"flywheel_inertia_g_cm2 must be within [0, 1e9], got {v}"
                )
        return self


class PlanFreezeRequest(BaseModel):
    trial_id: int = Field(ge=1)
    scenario: ScenarioParams


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class DynamicsPin(BaseModel):
    note_id: str
    pitch: int
    design_time_s: float
    angle_deg: float
    phase_rev: float = Field(description="unwrapped design phase in cylinder revolutions")


class TrialDerived(BaseModel):
    design_rpm: float
    total_ratio: float = Field(description="product of stage ratios (cylinder speed / barrel speed)")
    total_efficiency: float
    inertia_kg_m2: float
    required_cylinder_revolutions: float
    required_barrel_turns: float
    available_spring_turns: float
    pluck_event_count: int
    pin_count: int


class TrialSummary(BaseModel):
    id: int
    content_hash: str
    created_at: str
    source_version_id: int
    pin_count: int


class TrialResponse(BaseModel):
    id: int
    content_hash: str = Field(description="sha256 of source version hash + canonical spec")
    created_at: str
    source_version_id: int
    source_content_hash: str
    spec: DynamicsSpec
    derived: TrialDerived
    pins: list[DynamicsPin]


class ViolationLoc(BaseModel):
    """First occurrence of a violation kind, located on the pin timeline."""

    kind: Literal["stall", "overspeed", "beat_drift"]
    pin_index: Optional[int] = Field(description="pin event index the violation is attributed to")
    note_ids: list[str]
    time_s: float
    rpm: float
    detail: str


class PluckEventOut(BaseModel):
    index: int
    note_ids: list[str]
    pitches: list[int]
    phase_rev: float
    design_time_s: float
    time_s: float = Field(description="actual crossing time")
    rpm_before: float
    rpm_after: float = Field(description="immediately after the pluck energy is taken")
    min_rpm_until_next: float = Field(description="lowest rpm after this pluck until the next")
    beat_drift_ratio: float = Field(description="|actual-design|/design interval vs previous pluck")
    energy_uJ: float


class SimCurve(BaseModel):
    t_s: list[float]
    rpm: list[float]
    torque_margin_mNm: list[float] = Field(
        description="drive torque minus governor drag (plucks act as impulses)"
    )


class SimSummary(BaseModel):
    completed: bool = Field(description="every pin was plucked before the run ended")
    total_time_s: float = Field(description="accumulated running time (走时)")
    pins_plucked: int
    pin_count: int
    used_spring_turns: float
    remaining_spring_turns: float
    min_rpm: float
    min_torque_margin_mNm: float
    max_beat_drift_ratio: float
    stall: bool
    overspeed: bool
    drift_violations: int = Field(description="number of inter-pluck intervals over the drift limit")
    violation_count: int


class ScenarioResolved(BaseModel):
    prewind_turns: float
    governor_coefficient: float
    flywheel_inertia_g_cm2: float
    gear_ratio: float = Field(description="ratio actually used (override or stored train)")


class SimResult(BaseModel):
    design_rpm: float
    total_efficiency: float
    total_inertia_g_cm2: float
    dt_s: float
    scenario: ScenarioResolved
    summary: SimSummary
    curves: SimCurve
    pluck_events: list[PluckEventOut]
    first_violations: dict[str, Optional[ViolationLoc]]


class SimulateResponse(SimResult):
    trial_id: int
    source_content_hash: str


class CandidateMetrics(BaseModel):
    violation_count: int
    max_beat_drift_ratio: float
    min_torque_margin_mNm: float
    remaining_spring_turns: float
    min_rpm: float
    total_time_s: float
    completed: bool


class ScenarioCandidate(BaseModel):
    rank: int
    scenario: ScenarioResolved
    metrics: CandidateMetrics
    pluck_min_rpm: list[float] = Field(
        description="lowest rpm after each pluck until the next one"
    )
    first_violations: dict[str, Optional[ViolationLoc]]


class ScenarioSearchResponse(BaseModel):
    feasible: bool = Field(description="a candidate with zero violations exists")
    combinations_evaluated: int
    candidates: list[ScenarioCandidate]


class DynamicsPlanSummary(BaseModel):
    id: int
    content_hash: str
    created_at: str
    trial_content_hash: str
    violation_count: int


class DynamicsPlanResponse(BaseModel):
    id: int
    content_hash: str = Field(description="sha256 of source snapshot, curves, dt, scenario, result")
    created_at: str
    trial_content_hash: str
    source_version_id: int
    source_content_hash: str
    spec: DynamicsSpec
    scenario: ScenarioResolved
    dt_s: float
    result: SimResult


class DynamicsRecomputeResponse(BaseModel):
    id: int
    match: bool
    stored_hash: str
    recomputed_hash: str
