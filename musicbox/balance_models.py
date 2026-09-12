"""Pydantic request/response schemas for the dynamic-balancing service.

A balance plan takes its pin layout from a frozen pin-arrangement version and
adds the mechanical data needed for two-plane dynamic balancing: shell mass,
pin height and masses, working speed, bearing and correction-plane positions,
a residual-imbalance limit and the balance-weight inventory.

All masses are grams, lengths millimetres, angles degrees, speeds rpm.
Imbalance magnitudes are g*mm (static) and g*mm^2 (couple about mid-plane);
bearing loads are millinewtons (peak of the rotating reaction force).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class WeightType(BaseModel):
    """One balance-weight kind in stock."""

    id: str = Field(min_length=1, max_length=64)
    mass_g: float = Field(gt=0)
    diameter_mm: float = Field(gt=0, description="footprint diameter on the shell")
    quantity: int = Field(ge=0, le=100000)


class WeightRef(BaseModel):
    """A weight mounted on a correction plane at a given angle."""

    weight_id: str = Field(min_length=1, max_length=64)
    plane: Literal[1, 2]
    angle_deg: float = Field(ge=0, lt=360)


def _canonical_weights(weights: list[WeightRef]) -> list[WeightRef]:
    """Weights are a set: exact duplicates carry no extra meaning and must not
    change the plan, the hash or the resulting version."""
    unique = {(w.plane, w.angle_deg, w.weight_id): w for w in weights}
    return [unique[k] for k in sorted(unique)]


class BalanceSpec(BaseModel):
    """Mechanical parameters of the rotor, stored independently of the source
    pin-arrangement version."""

    shell_mass_g: float = Field(gt=0, description="bare cylinder shell mass")
    pin_height_mm: float = Field(gt=0, description="pin height above the shell surface")
    default_pin_mass_g: float = Field(gt=0)
    pin_masses: dict[str, float] = Field(
        default_factory=dict,
        description="per-pin mass overrides keyed by note_id of the source version",
    )
    working_rpm: float = Field(gt=0, description="speed at which loads are computed")
    bearing_a_mm: float = Field(description="axial position of bearing A (may be outboard)")
    bearing_b_mm: float = Field(description="axial position of bearing B (may be outboard)")
    plane_1_mm: float = Field(ge=0, description="axial position of correction plane 1")
    plane_2_mm: float = Field(ge=0, description="axial position of correction plane 2")
    residual_limit_gmm: float = Field(ge=0, description="residual static imbalance limit")
    inventory: list[WeightType] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cross_checks(self) -> "BalanceSpec":
        for nid, mass in self.pin_masses.items():
            if not nid:
                raise ValueError("pin_masses keys must be non-empty note ids")
            if mass <= 0:
                raise ValueError(f"pin mass for '{nid}' must be positive")
        if self.bearing_a_mm == self.bearing_b_mm:
            raise ValueError("bearing_a_mm and bearing_b_mm must differ")
        if self.plane_1_mm >= self.plane_2_mm:
            raise ValueError("plane_1_mm must be smaller than plane_2_mm")
        ids = [w.id for w in self.inventory]
        if len(set(ids)) != len(ids):
            raise ValueError("inventory weight ids must be unique")
        return self


class BalanceSearchLimits(BaseModel):
    angle_step_deg: float = Field(
        15.0, gt=0, le=180, description="placement angle grid step"
    )
    pin_clearance_mm: float = Field(
        1.0, ge=0, description="minimum edge-to-edge distance from any pin"
    )
    seam_clearance_mm: float = Field(
        2.0, ge=0, description="minimum edge-to-seam-line distance"
    )
    max_weights: int = Field(4, ge=0, le=12, description="maximum added weights")
    max_candidates: int = Field(5, ge=1, le=20)


class BalanceAnalyzeRequest(BaseModel):
    source_version_id: int = Field(ge=1)
    spec: BalanceSpec
    locked_weights: list[WeightRef] = Field(
        default_factory=list, description="already mounted weights (fixed)"
    )

    @model_validator(mode="after")
    def _canonical(self) -> "BalanceAnalyzeRequest":
        self.locked_weights = _canonical_weights(self.locked_weights)
        return self


class BalanceSearchRequest(BaseModel):
    source_version_id: int = Field(ge=1)
    spec: BalanceSpec
    locked_weights: list[WeightRef] = Field(default_factory=list)
    limits: BalanceSearchLimits = Field(default_factory=BalanceSearchLimits)

    @model_validator(mode="after")
    def _canonical(self) -> "BalanceSearchRequest":
        self.locked_weights = _canonical_weights(self.locked_weights)
        return self


class BalanceFreezeRequest(BaseModel):
    source_version_id: int = Field(ge=1)
    spec: BalanceSpec
    locked_weights: list[WeightRef] = Field(default_factory=list)
    weights: list[WeightRef] = Field(
        default_factory=list, description="chosen new weights from the search"
    )

    @model_validator(mode="after")
    def _canonical(self) -> "BalanceFreezeRequest":
        self.locked_weights = _canonical_weights(self.locked_weights)
        self.weights = _canonical_weights(self.weights)
        return self


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class PinContribution(BaseModel):
    """Rotating-mass vector of one pin, resolved with its effective mass."""

    note_id: str
    mass_g: float
    radius_mm: float = Field(description="centre-of-mass radius of the pin")
    angle_deg: float
    axial_mm: float
    static_x_gmm: float
    static_y_gmm: float
    couple_x_gmm2: float = Field(description="couple about the mid-plane, x component")
    couple_y_gmm2: float = Field(description="couple about the mid-plane, y component")


class Imbalance(BaseModel):
    """Rotor imbalance state (pins + locked weights + any chosen weights)."""

    total_mass_g: float
    com_offset_mm: float = Field(description="centre-of-mass offset from the axis")
    static_gmm: float = Field(description="static imbalance magnitude")
    static_angle_deg: float = Field(description="direction of the static imbalance")
    couple_gmm2: float = Field(description="couple imbalance magnitude about mid-plane")
    couple_angle_deg: float
    bearing_a_load_mn: float = Field(description="peak rotating load on bearing A")
    bearing_b_load_mn: float = Field(description="peak rotating load on bearing B")
    within_limit: bool


class PlaneCorrection(BaseModel):
    mass_g: float = Field(description="ideal correction mass at shell radius")
    angle_deg: float


class IdealCorrection(BaseModel):
    """Theoretical two-plane correction that would zero both residuals."""

    plane_1: PlaneCorrection
    plane_2: PlaneCorrection


class BalanceDerived(BaseModel):
    circumference_mm: float
    pin_radius_mm: float
    weight_radius_mm: float
    mid_plane_mm: float
    omega_rad_s: float
    pin_count: int


class BalanceAnalyzeResponse(BaseModel):
    source_version_id: int
    source_content_hash: str
    derived: BalanceDerived
    imbalance: Imbalance
    ideal_correction: IdealCorrection
    pins: list[PinContribution]
    locked_weights: list[WeightRef]


class WeightPlacement(BaseModel):
    """A concrete new weight in a candidate plan."""

    plane: int
    weight_id: str
    angle_deg: float
    axial_mm: float
    mass_g: float
    diameter_mm: float


class Residual(BaseModel):
    static_gmm: float
    couple_gmm2: float
    added_mass_g: float
    weight_count: int
    within_limit: bool


class BalanceCandidate(BaseModel):
    rank: int
    weights: list[WeightPlacement]
    residual: Residual
    imbalance: Imbalance


class BalanceSearchResponse(BaseModel):
    feasible: bool = Field(description="any candidate meets the residual limit")
    baseline: Imbalance = Field(description="imbalance with pins + locked weights only")
    ideal_correction: IdealCorrection
    candidates: list[BalanceCandidate]
    plans_evaluated: int


class BalancePlanSummary(BaseModel):
    id: int
    content_hash: str
    created_at: str
    source_version_id: int
    weight_count: int
    static_gmm: float


class BalancePlanResponse(BaseModel):
    id: int
    content_hash: str = Field(description="sha256 of the canonical plan input + result")
    created_at: str
    source_version_id: int
    source_content_hash: str
    spec: BalanceSpec
    locked_weights: list[WeightRef]
    weights: list[WeightRef]
    baseline: Imbalance = Field(description="before the chosen weights")
    imbalance: Imbalance = Field(description="final state with all weights")
    pins: list[PinContribution]
    svg: str = Field(description="unrolled cylinder with pins, weights and planes")


class BalanceRecomputeResponse(BaseModel):
    id: int
    match: bool
    stored_hash: str
    recomputed_hash: str
