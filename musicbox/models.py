"""Pydantic request/response schemas for the pin-arrangement service.

All physical dimensions are millimetres, angles are degrees, times are
seconds and musical positions are beats (quarter-note = 1 beat).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class NoteIn(BaseModel):
    """A score note. ``beat`` is the onset in beats from song start,
    ``duration`` the note value in beats, ``pitch`` a MIDI note number."""

    id: str = Field(min_length=1, max_length=64)
    beat: float = Field(ge=0)
    duration: float = Field(gt=0)
    pitch: int = Field(ge=0, le=127)
    locked: bool = False


class CombReed(BaseModel):
    """One comb tooth: which pitch it plays and where it sits on the axis."""

    pitch: int = Field(ge=0, le=127)
    axial_mm: float = Field(ge=0)


class CylinderSpec(BaseModel):
    diameter_mm: float = Field(gt=0)
    effective_length_mm: float = Field(gt=0)
    rpm: float = Field(gt=0, description="cylinder rotation speed, revolutions per minute")


class Constraints(BaseModel):
    pin_diameter_mm: float = Field(gt=0)
    seam_zone_mm: float = Field(
        ge=0, description="forbidden arc width centred on the cylinder seam (angle 0/360)"
    )
    min_rebound_seconds: float = Field(
        ge=0, description="minimum time between two plucks of the same reed"
    )
    min_clearance_mm: float = Field(
        ge=0, description="minimum edge-to-edge distance between any two pins"
    )


class ArrangementRequest(BaseModel):
    bpm: float = Field(gt=0)
    notes: list[NoteIn] = Field(min_length=1)
    cylinder: CylinderSpec
    comb: list[CombReed] = Field(min_length=1)
    constraints: Constraints

    @model_validator(mode="after")
    def _cross_checks(self) -> "ArrangementRequest":
        ids = [n.id for n in self.notes]
        if len(set(ids)) != len(ids):
            raise ValueError("note ids must be unique")
        pitches = [r.pitch for r in self.comb]
        if len(set(pitches)) != len(pitches):
            raise ValueError("comb pitches must be unique")
        limit = self.cylinder.effective_length_mm
        for r in self.comb:
            if r.axial_mm > limit:
                raise ValueError(
                    f"reed pitch {r.pitch} axial_mm {r.axial_mm} exceeds "
                    f"effective length {limit}"
                )
        return self


class SolutionSpec(BaseModel):
    """A concrete manufacturing choice applied on top of the raw score."""

    transpose_semitones: int = Field(0, ge=-24, le=24)
    tempo_factor: float = Field(1.0, gt=0.5, le=2.0)
    quantize_grid_beats: Optional[float] = Field(None, gt=0, le=4)
    deleted_note_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _canonical_deletions(self) -> "SolutionSpec":
        # Deletions are a set: duplicates must not change the pin layout, the
        # deletion count, the content hash or the resulting version.
        self.deleted_note_ids = sorted(set(self.deleted_note_ids))
        return self


def tempo_offsets_percent(float_percent: float, step_percent: float) -> list[float]:
    """Symmetric tempo offsets (percent) within ±float_percent.

    Always contains the base offset 0 and both endpoints; interior samples sit
    at ±k·step_percent, so the grid mirrors around the base even when the step
    does not divide the interval. Shared by the search-space validator and the
    engine so their counts never disagree."""
    offsets = {0.0, -float_percent, float_percent}
    n = int(float_percent / step_percent)  # floor for positive floats
    for k in range(1, n + 1):
        offsets.add(round(k * step_percent, 9))
        offsets.add(round(-k * step_percent, 9))
    return sorted(offsets)


class SearchLimits(BaseModel):
    max_transpose_semitones: int = Field(3, ge=0, le=12)
    tempo_float_percent: float = Field(5.0, ge=0, le=25)
    tempo_step_percent: float = Field(1.0, gt=0, le=25)
    quantize_grids_beats: list[float] = Field(default_factory=lambda: [0.5, 0.25, 0.125])
    max_quantize_error_beats: float = Field(0.25, ge=0, le=1)
    max_deletions: int = Field(30, ge=0, le=500)
    max_candidates: int = Field(5, ge=1, le=20)

    @model_validator(mode="after")
    def _bounded_search_space(self) -> "SearchLimits":
        for g in self.quantize_grids_beats:
            if g <= 0 or g > 4:
                raise ValueError("quantize grids must be within (0, 4] beats")
        n_t = 2 * self.max_transpose_semitones + 1
        n_f = len(tempo_offsets_percent(self.tempo_float_percent, self.tempo_step_percent))
        n_g = 1 + len(set(self.quantize_grids_beats))
        if n_t * n_f * n_g > 5000:
            raise ValueError(
                f"search space too large ({n_t * n_f * n_g} combinations, max 5000); "
                "narrow transpose range, tempo float or grid list"
            )
        return self


class SearchRequest(BaseModel):
    arrangement: ArrangementRequest
    limits: SearchLimits = Field(default_factory=SearchLimits)


class FreezeRequest(BaseModel):
    arrangement: ArrangementRequest
    solution: SolutionSpec = Field(default_factory=SolutionSpec)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

IssueKind = Literal[
    "missing_pitch",
    "chord_collision",
    "seam_crossing",
    "rebound_too_dense",
    "pin_clearance",
]


class Issue(BaseModel):
    kind: IssueKind
    beat: float = Field(description="earliest beat involved in the issue")
    note_ids: list[str]
    message: str
    angle_deg: Optional[float] = None
    axial_mm: Optional[float] = None
    measured: Optional[float] = Field(None, description="measured offending value")
    required: Optional[float] = Field(None, description="required limit value")
    unit: Optional[str] = Field(None, description='"mm" or "s"')


class PinMargins(BaseModel):
    """Slack of this pin against every constraint (negative => violation)."""

    clearance_mm: Optional[float] = Field(
        None, description="edge-to-edge distance to the nearest other pin minus the minimum"
    )
    rebound_s: Optional[float] = Field(
        None, description="time to the nearest same-reed neighbour minus the rebound interval"
    )
    seam_mm: float = Field(description="arc distance from pin edge to the seam zone edge")
    ok: bool


class PinOut(BaseModel):
    note_id: str
    pitch: int
    pitch_name: str
    locked: bool
    beat: float
    duration: float
    time_seconds: float
    angle_deg: float
    x_mm: float = Field(description="position along the circumference (unrolled)")
    axial_mm: float = Field(description="position along the cylinder axis")
    margins: PinMargins


class MissingNote(BaseModel):
    note_id: str
    pitch: int
    pitch_name: str
    beat: float
    angle_deg: float
    reason: str


class DerivedInfo(BaseModel):
    circumference_mm: float
    seconds_per_beat: float
    degrees_per_second: float
    degrees_per_beat: float
    song_seconds: float
    revolutions: float


class Diagnostics(BaseModel):
    ok: bool
    issue_counts: dict[str, int]
    first_by_kind: dict[str, Optional[Issue]] = Field(
        description="first (earliest-beat) issue of each kind, null when clean"
    )
    issues: list[Issue]


class CheckResponse(BaseModel):
    ok: bool
    derived: DerivedInfo
    pins: list[PinOut]
    missing_notes: list[MissingNote]
    diagnostics: Diagnostics


class Metrics(BaseModel):
    deleted_count: int
    pitch_deviation_semitones: float = Field(
        description="mean absolute semitone shift over kept notes"
    )
    rhythm_error_seconds: float = Field(
        description="max absolute onset-time shift over kept notes (tempo + quantize)"
    )
    min_clearance_mm: Optional[float] = Field(
        None, description="smallest edge-to-edge pin distance in the layout"
    )


class Candidate(BaseModel):
    rank: int
    solution: SolutionSpec
    metrics: Metrics
    pins: list[PinOut]
    diagnostics: Diagnostics


class SearchResponse(BaseModel):
    feasible: bool
    combinations_evaluated: int
    candidates: list[Candidate]


class VersionSummary(BaseModel):
    id: int
    content_hash: str
    created_at: str
    pin_count: int
    deleted_count: int


class VersionResponse(BaseModel):
    id: int
    content_hash: str = Field(description="sha256 of the canonical freeze input + result")
    created_at: str
    arrangement: ArrangementRequest
    solution: SolutionSpec
    pins: list[PinOut]
    metrics: Metrics
    svg: str = Field(description="printable unrolled-cylinder drawing (1 unit = 1 mm)")


class RecomputeResponse(BaseModel):
    id: int
    match: bool
    stored_hash: str
    recomputed_hash: str
