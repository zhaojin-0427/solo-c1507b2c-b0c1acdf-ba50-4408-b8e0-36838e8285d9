"""Deterministic assembly-calibration engine.

From the measurements of a calibration batch the engine fits the cylinder
axis — a straight line for the shell eccentricity vector plus a linear
radius deviation along z — resolves every pin of the source version against
the measured comb reeds (actual contacted reed, axial deviation, pluck depth
and margins) and locates the first pin of each violation kind (missed pluck,
wrong reed, double contact, over-depth). The adjustment search enumerates
bearing shim stacks x comb lateral shift x comb height adjustment within the
maker's locks and ranks candidates by violation count, minimum margin,
adjustment amount and shim variety count.

Everything here is a pure function of its inputs: no clocks, no randomness,
no ambient state, so recomputing a frozen plan yields identical output.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement, product

from .calibration_models import (
    VIOLATION_KINDS,
    Adjustment,
    AxisFit,
    CalDiagnostics,
    CalIssue,
    CalibrationCandidate,
    CalibrationPin,
    CalibrationSearchLimits,
    CalibrationSpec,
    CalSummary,
    DatumFit,
    ResolvedAdjustment,
    achievable_shim_totals,
    symmetric_offsets_mm,
)

EPS = 1e-9
TOL = 1e-6  # tolerance for matching user-supplied adjustment values
_KIND_ORDER = {k: i for i, k in enumerate(VIOLATION_KINDS)}


def r6(x: float) -> float:
    """Round for stable, tidy output; normalise -0.0 to 0.0."""
    v = round(float(x), 6)
    return 0.0 if v == 0 else v


class CalibrationError(ValueError):
    """Raised when a calibration request contradicts its source version or
    the physical limits (uncovered pins, unknown note ids, datum stations
    outside the cylinder, adjustments outside the search envelope)."""


# ---------------------------------------------------------------------------
# Source snapshot (from a frozen pin-arrangement version)
# ---------------------------------------------------------------------------


@dataclass
class CalSourcePin:
    note_id: str
    pitch: int
    angle_deg: float
    axial_mm: float


@dataclass
class CalSource:
    """Everything a calibration batch needs from the frozen pin version."""

    version_id: int
    content_hash: str
    diameter_mm: float
    length_mm: float
    pin_diameter_mm: float
    pins: list[CalSourcePin]

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0

    def snapshot_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "content_hash": self.content_hash,
            "cylinder": {
                "diameter_mm": self.diameter_mm,
                "effective_length_mm": self.length_mm,
                "pin_diameter_mm": self.pin_diameter_mm,
            },
            "pins": [
                {
                    "note_id": p.note_id,
                    "pitch": p.pitch,
                    "angle_deg": p.angle_deg,
                    "axial_mm": p.axial_mm,
                }
                for p in self.pins
            ],
        }

    @staticmethod
    def from_snapshot_dict(d: dict) -> "CalSource":
        c = d["cylinder"]
        return CalSource(
            version_id=d["version_id"],
            content_hash=d["content_hash"],
            diameter_mm=c["diameter_mm"],
            length_mm=c["effective_length_mm"],
            pin_diameter_mm=c["pin_diameter_mm"],
            pins=[
                CalSourcePin(
                    note_id=p["note_id"],
                    pitch=p["pitch"],
                    angle_deg=p["angle_deg"],
                    axial_mm=p["axial_mm"],
                )
                for p in d["pins"]
            ],
        )


def source_from_version_row(row) -> CalSource:
    """Build the source snapshot from a stored pin-arrangement version row."""
    request = json.loads(row["request_json"])
    result = json.loads(row["result_json"])
    arr = request["arrangement"]
    return CalSource(
        version_id=row["id"],
        content_hash=row["content_hash"],
        diameter_mm=arr["cylinder"]["diameter_mm"],
        length_mm=arr["cylinder"]["effective_length_mm"],
        pin_diameter_mm=arr["constraints"]["pin_diameter_mm"],
        pins=[
            CalSourcePin(
                note_id=p["note_id"],
                pitch=p["pitch"],
                angle_deg=p["angle_deg"],
                axial_mm=p["axial_mm"],
            )
            for p in result["pins"]
        ],
    )


# ---------------------------------------------------------------------------
# Validation against the source version
# ---------------------------------------------------------------------------


def validate_spec(spec: CalibrationSpec, source: CalSource) -> None:
    """Cross-validate the measurements against the source pins. Pydantic
    already guarantees measurement-point shapes, units and reed consistency;
    here we check coverage: every pin must be measurable and playable."""
    unknown = sorted(set(spec.pin_heights_mm) - {p.note_id for p in source.pins})
    if unknown:
        raise CalibrationError(
            f"pin_heights_mm reference unknown note ids: {unknown}"
        )
    datums = spec.runout.datum_points_mm
    for z in datums:
        if z > source.length_mm + EPS:
            raise CalibrationError(
                f"runout datum point {z}mm exceeds cylinder length "
                f"{source.length_mm}mm"
            )
    for r in spec.reeds:
        if r.axial_mm > source.length_mm + EPS:
            raise CalibrationError(
                f"reed pitch {r.pitch} axial_mm {r.axial_mm} exceeds cylinder "
                f"length {source.length_mm}mm"
            )
    if source.pins:
        lo = min(p.axial_mm for p in source.pins)
        hi = max(p.axial_mm for p in source.pins)
        if datums[0] > lo + EPS or datums[-1] < hi - EPS:
            raise CalibrationError(
                f"runout datum points [{datums[0]}, {datums[-1]}]mm do not "
                f"cover the pin axial range [{r6(lo)}, {r6(hi)}]mm"
            )
    used = {p.pitch for p in source.pins}
    missing = sorted(used - {r.pitch for r in spec.reeds})
    if missing:
        raise CalibrationError(f"no measured reed for pitches: {missing}")


# ---------------------------------------------------------------------------
# Axis fitting (least squares, closed form, deterministic)
# ---------------------------------------------------------------------------


def _det3(m: list[list[float]]) -> float:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _solve3(g: list[list[float]], rhs: list[float]) -> tuple[float, float, float]:
    """Solve a 3x3 system by Cramer's rule. With >= 3 distinct measurement
    angles the runout normal equations are non-singular (a sinusoid has at
    most two roots per period), so the guard is defensive."""
    det = _det3(g)
    if abs(det) < 1e-12:
        raise CalibrationError(
            "runout angles are degenerate: cannot fit an eccentricity"
        )
    out = []
    for col in range(3):
        m = [row[:] for row in g]
        for row in range(3):
            m[row][col] = rhs[row]
        out.append(_det3(m) / det)
    return out[0], out[1], out[2]


def _fit_station(angles_deg: list[float], values: list[float]) -> tuple[float, float, float]:
    """Least-squares fit of r(theta) = a + b*cos(theta) + c*sin(theta) at one
    datum station. (b, c) is the shell eccentricity vector there; a is the
    mean radius deviation."""
    cos = [math.cos(math.radians(a)) for a in angles_deg]
    sin = [math.sin(math.radians(a)) for a in angles_deg]
    n = len(values)
    sc, ss = sum(cos), sum(sin)
    scc = sum(c * c for c in cos)
    scs = sum(c * s for c, s in zip(cos, sin))
    sss = sum(s * s for s in sin)
    g = [[float(n), sc, ss], [sc, scc, scs], [ss, scs, sss]]
    rhs = [
        sum(values),
        sum(v * c for v, c in zip(values, cos)),
        sum(v * s for v, s in zip(values, sin)),
    ]
    return _solve3(g, rhs)


def _line(zs: list[float], vs: list[float]) -> tuple[float, float]:
    """Least-squares straight line v = intercept + slope*z (exact for two
    datum stations)."""
    n = len(zs)
    zm = sum(zs) / n
    vm = sum(vs) / n
    den = sum((z - zm) ** 2 for z in zs)
    slope = sum((z - zm) * (v - vm) for z, v in zip(zs, vs)) / den
    return vm - slope * zm, slope


@dataclass
class AxisModel:
    """Fitted axis in unrounded form: runout(z, theta) =
    a(z) + bx(z)*cos(theta) + by(z)*sin(theta), each linear in z."""

    a0: float
    a1: float
    bx0: float
    bx1: float
    by0: float
    by1: float
    stations: list[tuple[float, float, float]]  # per datum (a, b, c)
    max_residual_mm: float

    def runout(self, z_mm: float, angle_deg: float) -> float:
        t = math.radians(angle_deg)
        return (
            self.a0
            + self.a1 * z_mm
            + (self.bx0 + self.bx1 * z_mm) * math.cos(t)
            + (self.by0 + self.by1 * z_mm) * math.sin(t)
        )

    def eccentricity(self, z_mm: float) -> tuple[float, float]:
        """Eccentricity vector (x toward the comb, y lateral) at z."""
        return self.bx0 + self.bx1 * z_mm, self.by0 + self.by1 * z_mm


def fit_axis_model(spec: CalibrationSpec) -> AxisModel:
    grid = spec.runout
    zs = grid.datum_points_mm
    stations = [_fit_station(grid.angles_deg, row) for row in grid.values_mm]
    a0, a1 = _line(zs, [s[0] for s in stations])
    bx0, bx1 = _line(zs, [s[1] for s in stations])
    by0, by1 = _line(zs, [s[2] for s in stations])
    model = AxisModel(a0, a1, bx0, bx1, by0, by1, stations, 0.0)
    residual = max(
        abs(v - model.runout(z, theta))
        for z, row in zip(zs, grid.values_mm)
        for theta, v in zip(grid.angles_deg, row)
    )
    model.max_residual_mm = residual
    return model


def _angle_deg(bx: float, by: float) -> float:
    if math.hypot(bx, by) <= EPS:
        return 0.0
    return r6(math.degrees(math.atan2(by, bx)) % 360.0)


def axis_fit_out(model: AxisModel, spec: CalibrationSpec) -> AxisFit:
    def ecc_at(z: float) -> tuple[float, float]:
        bx, by = model.eccentricity(z)
        return r6(math.hypot(bx, by)), _angle_deg(bx, by)

    ecc_a, ang_a = ecc_at(spec.bearing_a_mm)
    ecc_b, ang_b = ecc_at(spec.bearing_b_mm)
    return AxisFit(
        radius_intercept_mm=r6(model.a0),
        radius_slope=r6(model.a1),
        ecc_x_intercept_mm=r6(model.bx0),
        ecc_x_slope=r6(model.bx1),
        ecc_y_intercept_mm=r6(model.by0),
        ecc_y_slope=r6(model.by1),
        ecc_at_bearing_a_mm=ecc_a,
        ecc_at_bearing_a_angle_deg=ang_a,
        ecc_at_bearing_b_mm=ecc_b,
        ecc_at_bearing_b_angle_deg=ang_b,
        max_residual_mm=r6(model.max_residual_mm),
        datums=[
            DatumFit(
                axial_mm=r6(z),
                mean_deviation_mm=r6(a),
                eccentricity_mm=r6(math.hypot(b, c)),
                eccentricity_angle_deg=_angle_deg(b, c),
            )
            for z, (a, b, c) in zip(spec.runout.datum_points_mm, model.stations)
        ],
    )


# ---------------------------------------------------------------------------
# Per-pin resolution against the measured comb
# ---------------------------------------------------------------------------


@dataclass
class CalAnalysis:
    pins: list[CalibrationPin]
    diagnostics: CalDiagnostics
    summary: CalSummary


@dataclass
class _Prepared:
    """Pin data that does not depend on the adjustment (cached for search)."""

    pin: CalSourcePin
    pin_height_mm: float
    runout_mm: float


def prepare(spec: CalibrationSpec, source: CalSource, model: AxisModel) -> list[_Prepared]:
    return [
        _Prepared(
            pin=p,
            pin_height_mm=spec.pin_heights_mm.get(p.note_id, spec.default_pin_height_mm),
            runout_mm=model.runout(p.axial_mm, p.angle_deg),
        )
        for p in source.pins
    ]


def _build_diagnostics(issues: list[CalIssue]) -> CalDiagnostics:
    counts: Counter[str] = Counter(i.kind for i in issues)
    first: dict[str, CalIssue | None] = {k: None for k in VIOLATION_KINDS}
    for issue in issues:  # already sorted by (pin_index, kind order)
        if first[issue.kind] is None:
            first[issue.kind] = issue
    return CalDiagnostics(
        ok=not issues,
        issue_counts={k: counts.get(k, 0) for k in VIOLATION_KINDS},
        first_by_kind=first,
        issues=issues,
    )


def analyze(
    spec: CalibrationSpec,
    source: CalSource,
    model: AxisModel,
    adj: Adjustment,
    prepared: list[_Prepared] | None = None,
) -> CalAnalysis:
    """Resolve every pin against the measured comb under one adjustment.

    Pin tip height = rotation-axis height at the pin's axial position
    (bearing heights interpolated linearly, shims included) + shell radius +
    fitted runout + pin height. Pluck depth = tip height - reed tip height.
    Axial contact is decided by the overlap of the pin footprint with each
    reed tip (comb shift included)."""
    if prepared is None:
        prepared = prepare(spec, source, model)
    reeds = spec.reeds
    intended = {r.pitch: i for i, r in enumerate(reeds)}
    h_a = spec.bearing_a_height_mm + adj.shim_a_mm
    h_b = spec.bearing_b_height_mm + adj.shim_b_mm
    z_a, z_b = spec.bearing_a_mm, spec.bearing_b_mm
    axis_slope = (h_b - h_a) / (z_b - z_a)
    half_pin = source.pin_diameter_mm / 2.0
    reed_lo = [r.axial_mm + adj.comb_shift_mm - r.width_mm / 2.0 for r in reeds]
    reed_hi = [r.axial_mm + adj.comb_shift_mm + r.width_mm / 2.0 for r in reeds]
    reed_h = [r.height_mm + adj.comb_height_mm for r in reeds]

    pins_out: list[CalibrationPin] = []
    issues: list[CalIssue] = []
    min_margin: float | None = None
    pins_with_violations = 0

    for idx, prep in enumerate(prepared):
        p = prep.pin
        z = p.axial_mm
        y_tip = h_a + axis_slope * (z - z_a) + source.radius_mm + prep.runout_mm + prep.pin_height_mm
        i_int = intended[p.pitch]
        r_int = reeds[i_int]
        depth = y_tip - reed_h[i_int]
        deviation = z - (r_int.axial_mm + adj.comb_shift_mm)
        p_lo, p_hi = z - half_pin, z + half_pin
        overlaps = [
            min(p_hi, reed_hi[i]) - max(p_lo, reed_lo[i]) for i in range(len(reeds))
        ]
        contacted = [i for i, ov in enumerate(overlaps) if ov > EPS]
        int_contacted = i_int in contacted

        kinds: list[str] = []
        if not int_contacted:
            kinds.append("missed")
            issues.append(
                CalIssue(
                    kind="missed",
                    pin_index=idx,
                    note_id=p.note_id,
                    pitch=p.pitch,
                    angle_deg=r6(p.angle_deg),
                    axial_mm=r6(z),
                    measured=r6(overlaps[i_int]),
                    required=0.0,
                    unit="mm",
                    message=(
                        f"pin does not overlap reed {p.pitch} axially "
                        f"(overlap {r6(overlaps[i_int])}mm)"
                    ),
                )
            )
            if contacted:
                kinds.append("wrong_reed")
                issues.append(
                    CalIssue(
                        kind="wrong_reed",
                        pin_index=idx,
                        note_id=p.note_id,
                        pitch=p.pitch,
                        angle_deg=r6(p.angle_deg),
                        axial_mm=r6(z),
                        measured=r6(max(overlaps[i] for i in contacted)),
                        required=0.0,
                        unit="mm",
                        message=(
                            f"pin contacts reed(s) "
                            f"{[reeds[i].pitch for i in contacted]} instead of "
                            f"reed {p.pitch}"
                        ),
                    )
                )
        else:
            if depth < spec.min_engagement_mm - EPS:
                kinds.append("missed")
                issues.append(
                    CalIssue(
                        kind="missed",
                        pin_index=idx,
                        note_id=p.note_id,
                        pitch=p.pitch,
                        angle_deg=r6(p.angle_deg),
                        axial_mm=r6(z),
                        measured=r6(depth),
                        required=spec.min_engagement_mm,
                        unit="mm",
                        message=(
                            f"pluck depth {r6(depth)}mm below minimum engagement "
                            f"{spec.min_engagement_mm}mm"
                        ),
                    )
                )
            if depth > r_int.max_pluck_depth_mm + EPS:
                kinds.append("over_depth")
                issues.append(
                    CalIssue(
                        kind="over_depth",
                        pin_index=idx,
                        note_id=p.note_id,
                        pitch=p.pitch,
                        angle_deg=r6(p.angle_deg),
                        axial_mm=r6(z),
                        measured=r6(depth),
                        required=r_int.max_pluck_depth_mm,
                        unit="mm",
                        message=(
                            f"pluck depth {r6(depth)}mm exceeds allowed "
                            f"{r_int.max_pluck_depth_mm}mm"
                        ),
                    )
                )
        if len(contacted) >= 2:
            kinds.append("double_contact")
            graze = max(
                overlaps[i] for i in contacted if i != i_int
            ) if int_contacted else max(overlaps[i] for i in contacted)
            issues.append(
                CalIssue(
                    kind="double_contact",
                    pin_index=idx,
                    note_id=p.note_id,
                    pitch=p.pitch,
                    angle_deg=r6(p.angle_deg),
                    axial_mm=r6(z),
                    measured=r6(graze),
                    required=0.0,
                    unit="mm",
                    message=(
                        f"pin contacts {len(contacted)} reeds simultaneously "
                        f"({[reeds[i].pitch for i in contacted]})"
                    ),
                )
            )

        ov_int = overlaps[i_int]
        others = [overlaps[i] for i in range(len(reeds)) if i != i_int]
        gap_other = -max(others) if others else math.inf
        axial_margin = min(ov_int, gap_other)
        depth_margin = min(
            depth - spec.min_engagement_mm, r_int.max_pluck_depth_mm - depth
        )
        margin = min(axial_margin, depth_margin)
        min_margin = margin if min_margin is None else min(min_margin, margin)
        if kinds:
            pins_with_violations += 1

        pins_out.append(
            CalibrationPin(
                pin_index=idx,
                note_id=p.note_id,
                pitch=p.pitch,
                angle_deg=r6(p.angle_deg),
                axial_mm=r6(z),
                pin_height_mm=r6(prep.pin_height_mm),
                runout_mm=r6(prep.runout_mm),
                tip_height_mm=r6(y_tip),
                intended_reed_pitch=r_int.pitch,
                contacted_pitches=[reeds[i].pitch for i in contacted],
                axial_deviation_mm=r6(deviation),
                pluck_depth_mm=r6(depth),
                axial_margin_mm=r6(axial_margin),
                depth_margin_mm=r6(depth_margin),
                margin_mm=r6(margin),
                violations=kinds,  # type: ignore[arg-type]
            )
        )

    issues.sort(key=lambda i: (i.pin_index, _KIND_ORDER[i.kind]))
    diagnostics = _build_diagnostics(issues)
    summary = CalSummary(
        pin_count=len(pins_out),
        pins_with_violations=pins_with_violations,
        violation_count=len(issues),
        min_margin_mm=r6(min_margin) if min_margin is not None else 0.0,
        ok=not issues,
    )
    return CalAnalysis(pins=pins_out, diagnostics=diagnostics, summary=summary)


# ---------------------------------------------------------------------------
# Shim stacks and the adjustment search
# ---------------------------------------------------------------------------


def shim_options(
    thicknesses: list[float], max_pieces: int
) -> list[tuple[float, tuple[float, ...]]]:
    """Every achievable shim-stack total with its canonical decomposition:
    fewest varieties, then fewest pieces. Physics depends only on the total,
    so each total appears once."""
    best: dict[float, tuple[tuple, tuple[float, ...]]] = {}
    for r in range(0, max_pieces + 1):
        for combo in combinations_with_replacement(sorted(thicknesses), r):
            total = round(sum(combo), 9)
            key = (len(set(combo)), len(combo), combo)
            if total not in best or key < best[total][0]:
                best[total] = (key, combo)
    return sorted((total, combo) for total, (key, combo) in best.items())


def decompose_shim(
    total_mm: float, thicknesses: list[float], max_pieces: int
) -> tuple[float, ...] | None:
    """Canonical decomposition of a requested shim total, or None when the
    total is not achievable with the stocked sizes and piece limit."""
    best: tuple[float, tuple[float, ...]] | None = None
    for total, pieces in shim_options(thicknesses, max_pieces):
        diff = abs(total - total_mm)
        if diff <= TOL and (best is None or diff < best[0]):
            best = (diff, pieces)
    return best[1] if best else None


def resolve_adjustment(
    limits: CalibrationSearchLimits, adj: Adjustment
) -> ResolvedAdjustment:
    """Validate an adjustment against the search envelope (locks, ranges,
    achievable shim totals) and resolve the shim stacks to stocked sizes."""
    if limits.lock_bearing_a and abs(adj.shim_a_mm) > TOL:
        raise CalibrationError("bearing A is locked: shim_a_mm must be 0")
    if limits.lock_bearing_b and abs(adj.shim_b_mm) > TOL:
        raise CalibrationError("bearing B is locked: shim_b_mm must be 0")
    if limits.lock_comb_shift and abs(adj.comb_shift_mm) > TOL:
        raise CalibrationError("comb shift is locked: comb_shift_mm must be 0")
    if limits.lock_comb_height and abs(adj.comb_height_mm) > TOL:
        raise CalibrationError("comb height is locked: comb_height_mm must be 0")
    if abs(adj.comb_shift_mm) > limits.comb_shift_range_mm + TOL:
        raise CalibrationError(
            f"comb shift {adj.comb_shift_mm}mm exceeds the allowed range "
            f"+/-{limits.comb_shift_range_mm}mm"
        )
    if abs(adj.comb_height_mm) > limits.comb_height_range_mm + TOL:
        raise CalibrationError(
            f"comb height adjustment {adj.comb_height_mm}mm exceeds the "
            f"allowed range +/-{limits.comb_height_range_mm}mm"
        )
    pieces_a = decompose_shim(adj.shim_a_mm, limits.shim_thicknesses_mm, limits.max_shims_per_end)
    if pieces_a is None:
        raise CalibrationError(
            f"shim total {adj.shim_a_mm}mm is not achievable with up to "
            f"{limits.max_shims_per_end} shims of {limits.shim_thicknesses_mm}mm"
        )
    pieces_b = decompose_shim(adj.shim_b_mm, limits.shim_thicknesses_mm, limits.max_shims_per_end)
    if pieces_b is None:
        raise CalibrationError(
            f"shim total {adj.shim_b_mm}mm is not achievable with up to "
            f"{limits.max_shims_per_end} shims of {limits.shim_thicknesses_mm}mm"
        )
    total_a = round(sum(pieces_a), 9)
    total_b = round(sum(pieces_b), 9)
    varieties = len(set(pieces_a) | set(pieces_b))
    return ResolvedAdjustment(
        shim_a_mm=r6(total_a),
        shim_b_mm=r6(total_b),
        comb_shift_mm=r6(adj.comb_shift_mm),
        comb_height_mm=r6(adj.comb_height_mm),
        shim_a_pieces_mm=[r6(v) for v in pieces_a],
        shim_b_pieces_mm=[r6(v) for v in pieces_b],
        shim_varieties=varieties,
        adjustment_total_mm=r6(
            total_a + total_b + abs(adj.comb_shift_mm) + abs(adj.comb_height_mm)
        ),
    )


def snap_adjustment(resolved: ResolvedAdjustment) -> Adjustment:
    """The canonical adjustment actually evaluated: shim totals snapped to
    the achievable stacks, all values rounded for stable output."""
    return Adjustment(
        shim_a_mm=resolved.shim_a_mm,
        shim_b_mm=resolved.shim_b_mm,
        comb_shift_mm=resolved.comb_shift_mm,
        comb_height_mm=resolved.comb_height_mm,
    )


def search(
    spec: CalibrationSpec, source: CalSource, limits: CalibrationSearchLimits
) -> tuple[list[CalibrationCandidate], int]:
    """Enumerate shim stacks x comb shift x comb height within the locks and
    rank by (violation count, minimum margin larger first, adjustment amount,
    shim variety count), then the adjustment tuple for determinism."""
    model = fit_axis_model(spec)
    prepared = prepare(spec, source, model)
    opts_a = (
        [(0.0, ())]
        if limits.lock_bearing_a
        else shim_options(limits.shim_thicknesses_mm, limits.max_shims_per_end)
    )
    opts_b = (
        [(0.0, ())]
        if limits.lock_bearing_b
        else shim_options(limits.shim_thicknesses_mm, limits.max_shims_per_end)
    )
    shifts = (
        [0.0]
        if limits.lock_comb_shift
        else symmetric_offsets_mm(limits.comb_shift_range_mm, limits.comb_shift_step_mm)
    )
    heights = (
        [0.0]
        if limits.lock_comb_height
        else symmetric_offsets_mm(limits.comb_height_range_mm, limits.comb_height_step_mm)
    )

    scored: list[tuple[tuple, CalibrationCandidate]] = []
    evaluated = 0
    for (t_a, pieces_a), (t_b, pieces_b), d_shift, d_height in product(
        opts_a, opts_b, shifts, heights
    ):
        adj = Adjustment(
            shim_a_mm=t_a, shim_b_mm=t_b,
            comb_shift_mm=d_shift, comb_height_mm=d_height,
        )
        ana = analyze(spec, source, model, adj, prepared)
        evaluated += 1
        varieties = len(set(pieces_a) | set(pieces_b))
        total_adj = t_a + t_b + abs(d_shift) + abs(d_height)
        resolved = ResolvedAdjustment(
            shim_a_mm=r6(t_a),
            shim_b_mm=r6(t_b),
            comb_shift_mm=r6(d_shift),
            comb_height_mm=r6(d_height),
            shim_a_pieces_mm=[r6(v) for v in pieces_a],
            shim_b_pieces_mm=[r6(v) for v in pieces_b],
            shim_varieties=varieties,
            adjustment_total_mm=r6(total_adj),
        )
        key = (
            ana.summary.violation_count,
            round(-ana.summary.min_margin_mm, 9),
            round(total_adj, 9),
            varieties,
            r6(t_a),
            r6(t_b),
            r6(d_shift),
            r6(d_height),
        )
        scored.append(
            (
                key,
                CalibrationCandidate(
                    rank=0,
                    adjustment=resolved,
                    violation_count=ana.summary.violation_count,
                    min_margin_mm=ana.summary.min_margin_mm,
                    issue_counts=ana.diagnostics.issue_counts,
                    first_by_kind=ana.diagnostics.first_by_kind,
                ),
            )
        )
    scored.sort(key=lambda item: item[0])
    top = [c for _, c in scored[: limits.max_candidates]]
    for rank, c in enumerate(top, start=1):
        c.rank = rank
    return top, evaluated
