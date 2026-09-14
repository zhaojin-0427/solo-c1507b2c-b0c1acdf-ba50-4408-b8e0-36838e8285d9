"""Pydantic request/response schemas for the assembly-calibration service.

A calibration batch takes its pin layout from a frozen pin-arrangement
version and adds the assembly measurements: radial runout of the shell at
several datum stations and angles, both bearing heights, default or per-pin
pin heights and the measured comb reed tips (axial position, tip height,
width and allowed pluck depth). A batch starts out 采集中 ("collecting");
confirming a chosen adjustment freezes an immutable calibration plan and the
batch becomes 已确认 ("confirmed").

Machine frame of reference: z along the cylinder axis, x toward the comb.
Bearing heights locate the rotation axis in the frame; runout is the signed
radial deviation of the shell surface about the rotation axis (positive =
larger radius), measured at cylinder angles that share the pin version's
angular reference (0 deg = seam). Pluck depth = pin tip height - reed tip
height.

All lengths are millimetres, angles are degrees.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

EPS = 1e-9
# upper bound on shim x shift x height combinations in one search
MAX_COMBINATIONS = 20000

ViolationKind = Literal["missed", "wrong_reed", "double_contact", "over_depth"]
VIOLATION_KINDS = ("missed", "wrong_reed", "double_contact", "over_depth")

BatchStatus = Literal["collecting", "confirmed"]


def symmetric_offsets_mm(range_mm: float, step_mm: float) -> list[float]:
    """Symmetric adjustment offsets within +/-range_mm. Always contains 0 and
    both endpoints; interior samples sit at +/-k*step_mm, so the grid mirrors
    around 0 even when the step does not divide the range. Shared by the
    search-space validator and the engine so their counts never disagree."""
    offsets = {0.0, -range_mm, range_mm}
    n = int(range_mm / step_mm)  # floor for positive floats
    for k in range(1, n + 1):
        offsets.add(round(k * step_mm, 9))
        offsets.add(round(-k * step_mm, 9))
    return sorted(offsets)


def achievable_shim_totals(thicknesses: list[float], max_pieces: int) -> list[float]:
    """Every shim-stack total achievable with up to ``max_pieces`` shims of the
    stocked thicknesses (unlimited quantity per size). Totals are rounded to
    1e-9 so float noise cannot split one physical total into two."""
    totals = {0.0}
    for r in range(1, max_pieces + 1):
        for combo in combinations_with_replacement(sorted(thicknesses), r):
            totals.add(round(sum(combo), 9))
    return sorted(totals)


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class RunoutGrid(BaseModel):
    """Radial runout of the shell about the rotation axis: one row of signed
    readings per datum station, sampled at every measurement angle."""

    datum_points_mm: list[float] = Field(
        min_length=2, max_length=16,
        description="axial stations of the runout measurements (strictly increasing)",
    )
    angles_deg: list[float] = Field(
        min_length=3, max_length=64,
        description="measurement angles, cylinder reference (strictly increasing)",
    )
    values_mm: list[list[float]] = Field(
        min_length=2,
        description="runout rows: one per datum station, one reading per angle",
    )

    @model_validator(mode="after")
    def _checks(self) -> "RunoutGrid":
        for a, b in zip(self.datum_points_mm, self.datum_points_mm[1:]):
            if not b > a:
                raise ValueError("datum_points_mm must be strictly increasing")
        if self.datum_points_mm[0] < 0:
            raise ValueError("datum_points_mm must be non-negative")
        for a in self.angles_deg:
            if not 0.0 <= a < 360.0:
                raise ValueError(f"runout angle {a} outside [0, 360) deg")
        for a, b in zip(self.angles_deg, self.angles_deg[1:]):
            if not b > a:
                raise ValueError("angles_deg must be strictly increasing")
        if len(self.values_mm) != len(self.datum_points_mm):
            raise ValueError(
                "runout grid must have one row per datum point "
                f"({len(self.values_mm)} rows for {len(self.datum_points_mm)} stations)"
            )
        for row in self.values_mm:
            if len(row) != len(self.angles_deg):
                raise ValueError(
                    "every runout row must have one reading per angle "
                    f"({len(self.angles_deg)})"
                )
        return self


class ReedTip(BaseModel):
    """One measured comb reed tip."""

    pitch: int = Field(ge=0, le=127)
    axial_mm: float = Field(ge=0, description="tip centre axial position")
    height_mm: float = Field(ge=0, description="tip height in the machine frame")
    width_mm: float = Field(gt=0, description="axial width of the tip")
    max_pluck_depth_mm: float = Field(gt=0, description="allowed pluck depth (允许拨入量)")


class CalibrationSpec(BaseModel):
    """Assembly measurements of one calibration batch, stored independently of
    the source pin-arrangement version."""

    length_unit: Literal["mm"] = "mm"
    angle_unit: Literal["deg"] = "deg"
    runout: RunoutGrid
    bearing_a_mm: float = Field(description="axial position of bearing A (may be outboard)")
    bearing_b_mm: float = Field(description="axial position of bearing B (may be outboard)")
    bearing_a_height_mm: float = Field(0.0, description="measured height of bearing A")
    bearing_b_height_mm: float = Field(0.0, description="measured height of bearing B")
    default_pin_height_mm: float = Field(gt=0)
    pin_heights_mm: dict[str, float] = Field(
        default_factory=dict,
        description="per-pin height overrides keyed by note_id of the source version",
    )
    reeds: list[ReedTip] = Field(min_length=1)
    min_engagement_mm: float = Field(
        0.2, ge=0, description="minimum pluck depth that still sounds the reed"
    )

    @model_validator(mode="after")
    def _cross_checks(self) -> "CalibrationSpec":
        if self.bearing_a_mm == self.bearing_b_mm:
            raise ValueError("bearing_a_mm and bearing_b_mm must differ")
        for nid, h in self.pin_heights_mm.items():
            if not nid:
                raise ValueError("pin_heights_mm keys must be non-empty note ids")
            if h <= 0:
                raise ValueError(f"pin height for '{nid}' must be positive")
        pitches = [r.pitch for r in self.reeds]
        if len(set(pitches)) != len(pitches):
            raise ValueError("reed pitches must be unique")
        # the reed order carries no meaning: canonicalise so the same physical
        # comb always hashes identically
        self.reeds = sorted(self.reeds, key=lambda r: (r.axial_mm, r.pitch))
        for a, b in zip(self.reeds, self.reeds[1:]):
            gap = b.axial_mm - a.axial_mm - (a.width_mm + b.width_mm) / 2.0
            if gap < -EPS:
                raise ValueError(
                    f"reed tips at {a.axial_mm}mm and {b.axial_mm}mm overlap axially"
                )
        for r in self.reeds:
            if self.min_engagement_mm >= r.max_pluck_depth_mm:
                raise ValueError(
                    f"min_engagement_mm {self.min_engagement_mm} must be smaller "
                    f"than the allowed pluck depth of reed {r.pitch} "
                    f"({r.max_pluck_depth_mm}mm)"
                )
        return self


class Adjustment(BaseModel):
    """One concrete assembly adjustment."""

    shim_a_mm: float = Field(0.0, ge=0, le=20, description="shim stack under bearing A")
    shim_b_mm: float = Field(0.0, ge=0, le=20, description="shim stack under bearing B")
    comb_shift_mm: float = Field(0.0, ge=-50, le=50, description="comb lateral (axial) shift")
    comb_height_mm: float = Field(0.0, ge=-50, le=50, description="comb height adjustment")


class CalibrationSearchLimits(BaseModel):
    """Search envelope for the adjustment search. Locks mark the bearings or
    the comb as unchangeable: the corresponding dimension is fixed at 0."""

    shim_thicknesses_mm: list[float] = Field(
        default_factory=lambda: [0.05, 0.1, 0.2],
        min_length=1,
        max_length=8,
        description="stocked shim thicknesses (unlimited quantity per size)",
    )
    max_shims_per_end: int = Field(2, ge=0, le=4)
    comb_shift_range_mm: float = Field(0.6, ge=0, le=20)
    comb_shift_step_mm: float = Field(0.1, gt=0, le=5)
    comb_height_range_mm: float = Field(0.4, ge=0, le=20)
    comb_height_step_mm: float = Field(0.05, gt=0, le=5)
    lock_bearing_a: bool = False
    lock_bearing_b: bool = False
    lock_comb_shift: bool = False
    lock_comb_height: bool = False
    max_candidates: int = Field(5, ge=1, le=20)

    @model_validator(mode="after")
    def _bounded_search_space(self) -> "CalibrationSearchLimits":
        # shim sizes are a set: duplicates must not change the search or the hash
        self.shim_thicknesses_mm = sorted(set(self.shim_thicknesses_mm))
        for t in self.shim_thicknesses_mm:
            if not 0.0 < t <= 5.0:
                raise ValueError(f"shim thicknesses must be within (0, 5] mm, got {t}")
        n_shim = len(achievable_shim_totals(self.shim_thicknesses_mm, self.max_shims_per_end))
        n_a = 1 if self.lock_bearing_a else n_shim
        n_b = 1 if self.lock_bearing_b else n_shim
        n_s = 1 if self.lock_comb_shift else len(
            symmetric_offsets_mm(self.comb_shift_range_mm, self.comb_shift_step_mm)
        )
        n_h = 1 if self.lock_comb_height else len(
            symmetric_offsets_mm(self.comb_height_range_mm, self.comb_height_step_mm)
        )
        combos = n_a * n_b * n_s * n_h
        if combos > MAX_COMBINATIONS:
            raise ValueError(
                f"search space too large ({combos} combinations, max "
                f"{MAX_COMBINATIONS}); narrow the shim set, the ranges or the steps"
            )
        return self


class CalibrationBatchRequest(BaseModel):
    source_version_id: int = Field(ge=1)
    spec: CalibrationSpec


class CalibrationSearchRequest(BaseModel):
    limits: CalibrationSearchLimits = Field(default_factory=CalibrationSearchLimits)


class CalibrationConfirmRequest(BaseModel):
    """Confirm a chosen adjustment: freezes an immutable calibration plan and
    marks the batch 已确认. ``limits`` is the envelope the adjustment was
    chosen under (locks and ranges are enforced, and kept for traceability)."""

    adjustment: Adjustment = Field(default_factory=Adjustment)
    limits: CalibrationSearchLimits = Field(default_factory=CalibrationSearchLimits)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class DatumFit(BaseModel):
    """Fitted runout harmonic at one datum station."""

    axial_mm: float
    mean_deviation_mm: float = Field(description="mean radius deviation at this station")
    eccentricity_mm: float = Field(description="shell eccentricity magnitude")
    eccentricity_angle_deg: float = Field(description="eccentricity direction")


class AxisFit(BaseModel):
    """Fitted cylinder axis: the shell eccentricity vector and the mean radius
    deviation as straight lines along z. x points toward the comb."""

    radius_intercept_mm: float = Field(description="mean radius deviation at z=0")
    radius_slope: float = Field(description="radius deviation per mm of axis (taper)")
    ecc_x_intercept_mm: float
    ecc_x_slope: float
    ecc_y_intercept_mm: float
    ecc_y_slope: float
    ecc_at_bearing_a_mm: float
    ecc_at_bearing_a_angle_deg: float
    ecc_at_bearing_b_mm: float
    ecc_at_bearing_b_angle_deg: float
    max_residual_mm: float = Field(description="worst runout reading minus fit")
    datums: list[DatumFit]


class CalibrationPin(BaseModel):
    """Per-pin resolution against the measured comb."""

    pin_index: int = Field(description="index in the source version's pin order")
    note_id: str
    pitch: int
    angle_deg: float
    axial_mm: float
    pin_height_mm: float
    runout_mm: float = Field(description="fitted shell runout at the pin")
    tip_height_mm: float = Field(description="pin tip height in the machine frame")
    intended_reed_pitch: int
    contacted_pitches: list[int] = Field(description="reeds the pin actually touches")
    axial_deviation_mm: float = Field(description="pin centre minus intended reed centre")
    pluck_depth_mm: float
    axial_margin_mm: float = Field(
        description="min(overlap with the intended reed, gap to the nearest other reed)"
    )
    depth_margin_mm: float = Field(
        description="min(depth - min engagement, allowed depth - depth)"
    )
    margin_mm: float = Field(description="worst of the axial and depth margins")
    violations: list[ViolationKind]


class CalIssue(BaseModel):
    kind: ViolationKind
    pin_index: int
    note_id: str
    pitch: int
    angle_deg: float
    axial_mm: float
    measured: Optional[float] = Field(None, description="measured offending value")
    required: Optional[float] = Field(None, description="required limit value")
    unit: Optional[str] = Field(None, description='"mm"')
    message: str


class CalDiagnostics(BaseModel):
    ok: bool
    issue_counts: dict[str, int]
    first_by_kind: dict[str, Optional[CalIssue]] = Field(
        description="first pin (in source order) of each violation kind, null when clean"
    )
    issues: list[CalIssue]


class CalSummary(BaseModel):
    pin_count: int
    pins_with_violations: int
    violation_count: int
    min_margin_mm: float
    ok: bool


class CalibrationBatchSummary(BaseModel):
    id: int
    content_hash: str
    created_at: str
    status: BatchStatus
    source_version_id: int
    pin_count: int
    violation_count: int


class CalibrationBatchResponse(BaseModel):
    id: int
    content_hash: str = Field(description="sha256 of source hash + measurements + fit + analysis")
    created_at: str
    status: BatchStatus = Field(
        description="collecting (采集中) until an adjustment is confirmed, then confirmed (已确认)"
    )
    confirmed_plan_id: Optional[int]
    source_version_id: int
    source_content_hash: str
    spec: CalibrationSpec
    fit: AxisFit
    pins: list[CalibrationPin]
    diagnostics: CalDiagnostics
    summary: CalSummary


class ResolvedAdjustment(Adjustment):
    """The chosen adjustment with its shim stacks resolved to stocked sizes."""

    shim_a_pieces_mm: list[float]
    shim_b_pieces_mm: list[float]
    shim_varieties: int = Field(description="distinct shim sizes used across both ends")
    adjustment_total_mm: float = Field(
        description="shim_a + shim_b + |comb shift| + |comb height|"
    )


class CalibrationCandidate(BaseModel):
    rank: int
    adjustment: ResolvedAdjustment
    violation_count: int
    min_margin_mm: float
    issue_counts: dict[str, int]
    first_by_kind: dict[str, Optional[CalIssue]]


class CalibrationSearchResponse(BaseModel):
    feasible: bool = Field(description="a candidate with zero violations exists")
    combinations_evaluated: int
    candidates: list[CalibrationCandidate]


class CalibrationPlanSummary(BaseModel):
    id: int
    content_hash: str
    created_at: str
    batch_content_hash: str
    violation_count: int
    min_margin_mm: float


class CalibrationPlanResponse(BaseModel):
    id: int
    content_hash: str = Field(
        description="sha256 of source snapshot, measurements, limits, adjustment, fit and analysis"
    )
    created_at: str
    batch_content_hash: str
    source_version_id: int
    source_content_hash: str
    spec: CalibrationSpec
    limits: CalibrationSearchLimits
    adjustment: ResolvedAdjustment
    fit: AxisFit
    pins: list[CalibrationPin]
    diagnostics: CalDiagnostics
    summary: CalSummary


class CalibrationRecomputeResponse(BaseModel):
    id: int
    match: bool
    stored_hash: str
    recomputed_hash: str
