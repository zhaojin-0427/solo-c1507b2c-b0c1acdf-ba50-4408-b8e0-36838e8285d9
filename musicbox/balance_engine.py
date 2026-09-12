"""Deterministic two-plane dynamic-balancing engine.

Every pin of the source version becomes a rotating mass vector (its mass at
its centre-of-mass radius, angle and axial position). The engine sums those
vectors into the static imbalance and the couple about the mid-plane, derives
bearing reactions, computes the ideal two-plane correction and searches the
weight inventory for concrete placements.

Everything here is a pure function of its inputs: no clocks, no randomness,
no ambient state, so recomputing a frozen plan yields identical output.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .balance_models import (
    BalanceCandidate,
    BalanceDerived,
    BalanceSearchLimits,
    BalanceSpec,
    Imbalance,
    IdealCorrection,
    PinContribution,
    PlaneCorrection,
    Residual,
    WeightPlacement,
    WeightRef,
)
from .engine import r6

EPS = 1e-9
BEAM_WIDTH = 32  # plans kept per weight-count level during the search


class BalancePlanError(ValueError):
    """Raised when a balance request contradicts its source version or the
    physical limits (unknown pin ids, planes/weights beyond the cylinder,
    unknown or exhausted weights, overlapping weights)."""


# ---------------------------------------------------------------------------
# Source snapshot (from a frozen pin-arrangement version)
# ---------------------------------------------------------------------------


@dataclass
class SourcePin:
    note_id: str
    angle_deg: float
    axial_mm: float
    locked: bool


@dataclass
class SourceInfo:
    """Everything a balance plan needs from the frozen pin version."""

    version_id: int
    content_hash: str
    diameter_mm: float
    length_mm: float
    pin_diameter_mm: float
    seam_zone_mm: float
    pins: list[SourcePin]

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0

    @property
    def circumference_mm(self) -> float:
        return math.pi * self.diameter_mm

    def snapshot_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "content_hash": self.content_hash,
            "cylinder": {
                "diameter_mm": self.diameter_mm,
                "effective_length_mm": self.length_mm,
                "pin_diameter_mm": self.pin_diameter_mm,
                "seam_zone_mm": self.seam_zone_mm,
            },
            "pins": [
                {
                    "note_id": p.note_id,
                    "angle_deg": p.angle_deg,
                    "axial_mm": p.axial_mm,
                    "locked": p.locked,
                }
                for p in self.pins
            ],
        }

    @staticmethod
    def from_snapshot_dict(d: dict) -> "SourceInfo":
        c = d["cylinder"]
        return SourceInfo(
            version_id=d["version_id"],
            content_hash=d["content_hash"],
            diameter_mm=c["diameter_mm"],
            length_mm=c["effective_length_mm"],
            pin_diameter_mm=c["pin_diameter_mm"],
            seam_zone_mm=c["seam_zone_mm"],
            pins=[
                SourcePin(
                    note_id=p["note_id"],
                    angle_deg=p["angle_deg"],
                    axial_mm=p["axial_mm"],
                    locked=p.get("locked", False),
                )
                for p in d["pins"]
            ],
        )


def source_from_version_row(row) -> SourceInfo:
    """Build the source snapshot from a stored pin-arrangement version row."""
    import json

    request = json.loads(row["request_json"])
    result = json.loads(row["result_json"])
    arr = request["arrangement"]
    return SourceInfo(
        version_id=row["id"],
        content_hash=row["content_hash"],
        diameter_mm=arr["cylinder"]["diameter_mm"],
        length_mm=arr["cylinder"]["effective_length_mm"],
        pin_diameter_mm=arr["constraints"]["pin_diameter_mm"],
        seam_zone_mm=arr["constraints"]["seam_zone_mm"],
        pins=[
            SourcePin(
                note_id=p["note_id"],
                angle_deg=p["angle_deg"],
                axial_mm=p["axial_mm"],
                locked=p["locked"],
            )
            for p in result["pins"]
        ],
    )


# ---------------------------------------------------------------------------
# Rotating mass points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MassPoint:
    """A point mass rotating with the cylinder."""

    mass_g: float
    radius_mm: float
    angle_deg: float
    axial_mm: float
    kind: str  # "pin" | "locked" | "weight"
    ref: str  # note_id or weight id

    def vector(self) -> complex:
        """Static imbalance vector m*r at the mounting angle (g*mm)."""
        return self.mass_g * self.radius_mm * complex(
            math.cos(math.radians(self.angle_deg)),
            math.sin(math.radians(self.angle_deg)),
        )


def pin_points(spec: BalanceSpec, source: SourceInfo) -> list[MassPoint]:
    """One rotating mass per pin; centre of mass at half the pin height."""
    r = source.radius_mm + spec.pin_height_mm / 2.0
    return [
        MassPoint(
            mass_g=spec.pin_masses.get(p.note_id, spec.default_pin_mass_g),
            radius_mm=r,
            angle_deg=p.angle_deg,
            axial_mm=p.axial_mm,
            kind="pin",
            ref=p.note_id,
        )
        for p in source.pins
    ]


def weight_points(
    spec: BalanceSpec, source: SourceInfo, weights: list[WeightRef], kind: str
) -> list[MassPoint]:
    """Weights sit on the shell surface on their correction plane."""
    planes = {1: spec.plane_1_mm, 2: spec.plane_2_mm}
    by_id = {t.id: t for t in spec.inventory}
    return [
        MassPoint(
            mass_g=by_id[w.weight_id].mass_g,
            radius_mm=source.radius_mm,
            angle_deg=w.angle_deg,
            axial_mm=planes[w.plane],
            kind=kind,
            ref=w.weight_id,
        )
        for w in weights
    ]


def sum_vectors(points: list[MassPoint], mid_plane_mm: float) -> tuple[complex, complex]:
    """(static imbalance U in g*mm, couple M about the mid-plane in g*mm^2)."""
    u = sum((p.vector() for p in points), 0j)
    m = sum((p.vector() * (p.axial_mm - mid_plane_mm) for p in points), 0j)
    return u, m


# ---------------------------------------------------------------------------
# Validation against the source version and physical limits
# ---------------------------------------------------------------------------


def _weight_fits_length(axial_mm: float, diameter_mm: float, length_mm: float) -> bool:
    return axial_mm - diameter_mm / 2.0 >= -EPS and axial_mm + diameter_mm / 2.0 <= length_mm + EPS


def validate_spec(spec: BalanceSpec, source: SourceInfo) -> None:
    unknown = sorted(set(spec.pin_masses) - {p.note_id for p in source.pins})
    if unknown:
        raise BalancePlanError(
            f"pin_masses reference unknown note ids: {unknown}"
        )
    for name, z in (("plane_1_mm", spec.plane_1_mm), ("plane_2_mm", spec.plane_2_mm)):
        if z > source.length_mm + EPS:
            raise BalancePlanError(
                f"correction plane {name}={z} exceeds cylinder length "
                f"{source.length_mm}"
            )


def validate_weights(
    spec: BalanceSpec,
    source: SourceInfo,
    locked: list[WeightRef],
    chosen: list[WeightRef],
) -> None:
    by_id = {t.id: t for t in spec.inventory}
    planes = {1: spec.plane_1_mm, 2: spec.plane_2_mm}
    all_weights = list(locked) + list(chosen)
    for w in all_weights:
        if w.weight_id not in by_id:
            raise BalancePlanError(f"unknown weight id: '{w.weight_id}'")
        z = planes[w.plane]
        d = by_id[w.weight_id].diameter_mm
        if not _weight_fits_length(z, d, source.length_mm):
            raise BalancePlanError(
                f"weight '{w.weight_id}' on plane {w.plane} (axial {z} mm, "
                f"diameter {d} mm) extends beyond the cylinder length "
                f"{source.length_mm}"
            )
    counts = Counter(w.weight_id for w in all_weights)
    for wid, n in sorted(counts.items()):
        if n > by_id[wid].quantity:
            raise BalancePlanError(
                f"weight '{wid}' used {n} times but only "
                f"{by_id[wid].quantity} in stock"
            )
    # no two mounted weights may physically overlap
    pts = weight_points(spec, source, all_weights, "check")
    dias = [by_id[w.weight_id].diameter_mm for w in all_weights]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            edge = _surface_distance(source, pts[i], pts[j]) - (
                dias[i] + dias[j]
            ) / 2.0
            if edge < -EPS:
                raise BalancePlanError(
                    f"weights '{pts[i].ref}' and '{pts[j].ref}' overlap "
                    f"(plane angles {pts[i].angle_deg}/{pts[j].angle_deg} deg)"
                )


def _surface_distance(source: SourceInfo, a: MassPoint, b: MassPoint) -> float:
    """Centre-to-centre distance of two surface points on the unrolled shell."""
    d_ang = abs(a.angle_deg - b.angle_deg) % 360.0
    d_ang = min(d_ang, 360.0 - d_ang)
    dy = source.circumference_mm * d_ang / 360.0
    return math.hypot(abs(a.axial_mm - b.axial_mm), dy)


# ---------------------------------------------------------------------------
# Imbalance computation
# ---------------------------------------------------------------------------


def _angle_of(v: complex) -> float:
    return r6(math.degrees(math.atan2(v.imag, v.real)) % 360.0)


def compute_imbalance(
    spec: BalanceSpec, source: SourceInfo, points: list[MassPoint]
) -> Imbalance:
    mid = source.length_mm / 2.0
    u, m = sum_vectors(points, mid)
    total = spec.shell_mass_g + sum(p.mass_g for p in points)
    omega = 2.0 * math.pi * spec.working_rpm / 60.0
    # rotating bearing reactions (complex, g*mm); force = reaction * omega^2
    za, zb = spec.bearing_a_mm, spec.bearing_b_mm
    moment_about_a = sum(
        (p.vector() * (p.axial_mm - za) for p in points), 0j
    )
    reaction_b = moment_about_a / (zb - za)
    reaction_a = u - reaction_b
    to_mn = omega * omega * 1e-3  # g*mm/s^2 -> mN
    return Imbalance(
        total_mass_g=r6(total),
        com_offset_mm=r6(abs(u) / total),
        static_gmm=r6(abs(u)),
        static_angle_deg=_angle_of(u),
        couple_gmm2=r6(abs(m)),
        couple_angle_deg=_angle_of(m),
        bearing_a_load_mn=r6(abs(reaction_a) * to_mn),
        bearing_b_load_mn=r6(abs(reaction_b) * to_mn),
        within_limit=abs(u) <= spec.residual_limit_gmm + EPS,
    )


def ideal_correction(
    spec: BalanceSpec, source: SourceInfo, points: list[MassPoint]
) -> IdealCorrection:
    """Exact two-plane correction vectors: weights W1, W2 (g*mm at shell
    radius) with W1 + W2 = -U and W1*a1 + W2*a2 = -M."""
    mid = source.length_mm / 2.0
    u, m = sum_vectors(points, mid)
    a1 = spec.plane_1_mm - mid
    a2 = spec.plane_2_mm - mid
    w2 = (u * a1 - m) / (a2 - a1)
    w1 = -u - w2
    r = source.radius_mm

    def plane(w: complex) -> PlaneCorrection:
        return PlaneCorrection(mass_g=r6(abs(w) / r), angle_deg=_angle_of(w))

    return IdealCorrection(plane_1=plane(w1), plane_2=plane(w2))


def pin_contributions(
    spec: BalanceSpec, source: SourceInfo, points: list[MassPoint]
) -> list[PinContribution]:
    mid = source.length_mm / 2.0
    out = []
    for p in points:
        if p.kind != "pin":
            continue
        v = p.vector()
        c = v * (p.axial_mm - mid)
        out.append(
            PinContribution(
                note_id=p.ref,
                mass_g=r6(p.mass_g),
                radius_mm=r6(p.radius_mm),
                angle_deg=r6(p.angle_deg),
                axial_mm=r6(p.axial_mm),
                static_x_gmm=r6(v.real),
                static_y_gmm=r6(v.imag),
                couple_x_gmm2=r6(c.real),
                couple_y_gmm2=r6(c.imag),
            )
        )
    return out


def derived_info(spec: BalanceSpec, source: SourceInfo) -> BalanceDerived:
    return BalanceDerived(
        circumference_mm=r6(source.circumference_mm),
        pin_radius_mm=r6(source.radius_mm + spec.pin_height_mm / 2.0),
        weight_radius_mm=r6(source.radius_mm),
        mid_plane_mm=r6(source.length_mm / 2.0),
        omega_rad_s=r6(2.0 * math.pi * spec.working_rpm / 60.0),
        pin_count=len(source.pins),
    )


# ---------------------------------------------------------------------------
# Weight placement search
# ---------------------------------------------------------------------------


def angle_grid(step_deg: float) -> list[float]:
    """Placement angles 0, step, 2*step, ... below 360 (deterministic)."""
    out = []
    k = 0
    while k * step_deg < 360.0 - EPS:
        out.append(r6(k * step_deg))
        k += 1
    return out


@dataclass(frozen=True)
class _Placement:
    plane: int
    weight_id: str
    angle_deg: float
    mass_g: float
    diameter_mm: float
    axial_mm: float

    def point(self, radius_mm: float) -> MassPoint:
        return MassPoint(
            mass_g=self.mass_g,
            radius_mm=radius_mm,
            angle_deg=self.angle_deg,
            axial_mm=self.axial_mm,
            kind="weight",
            ref=self.weight_id,
        )


def _allowed_placements(
    spec: BalanceSpec,
    source: SourceInfo,
    limits: BalanceSearchLimits,
    locked: list[tuple[MassPoint, float]],
    available: dict[str, int],
) -> list[_Placement]:
    """Every (plane, weight type, angle) placement that fits the cylinder
    length, keeps the safety distances to pins and the seam, and does not
    overlap an already mounted (locked) weight."""
    circ = source.circumference_mm
    pin_half = source.pin_diameter_mm / 2.0
    pin_points_ = [
        MassPoint(0.0, 0.0, sp.angle_deg, sp.axial_mm, "pin", sp.note_id)
        for sp in source.pins
    ]
    out: list[_Placement] = []
    for plane, z in ((1, spec.plane_1_mm), (2, spec.plane_2_mm)):
        for t in spec.inventory:
            if available[t.id] <= 0:
                continue
            if not _weight_fits_length(z, t.diameter_mm, source.length_mm):
                continue  # would stick out beyond the cylinder length
            half = t.diameter_mm / 2.0
            for angle in angle_grid(limits.angle_step_deg):
                # seam safety distance (edge of the weight to the seam line)
                seam_arc = min(angle, 360.0 - angle) / 360.0 * circ - half
                if seam_arc < limits.seam_clearance_mm - EPS:
                    continue
                cand = MassPoint(t.mass_g, source.radius_mm, angle, z, "weight", t.id)
                # pin safety distance (edge to edge)
                if any(
                    _surface_distance(source, cand, p) - half - pin_half
                    < limits.pin_clearance_mm - EPS
                    for p in pin_points_
                ):
                    continue
                # no overlap with locked weights
                if any(
                    _surface_distance(source, cand, lw) - half - lw_half < -EPS
                    for lw, lw_half in locked
                ):
                    continue
                out.append(_Placement(plane, t.id, angle, t.mass_g, t.diameter_mm, z))
    out.sort(key=lambda p: (p.plane, p.angle_deg, p.weight_id))
    return out


def _placement_key(p: _Placement) -> tuple:
    return (p.plane, p.angle_deg, p.weight_id)


def _plan_key(plan: tuple[_Placement, ...]) -> tuple:
    return tuple(_placement_key(p) for p in plan)


def search_weights(
    spec: BalanceSpec,
    source: SourceInfo,
    pins: list[MassPoint],
    locked_points: list[MassPoint],
    limits: BalanceSearchLimits,
) -> tuple[list[BalanceCandidate], int]:
    """Search inventory placements on the two correction planes.

    Plans with one or two weights (the standard two-plane correction) are
    enumerated exhaustively; deeper plans up to ``max_weights`` are explored
    by beam search from the best shallower plans. Candidates are ranked
    lexicographically by (residual static imbalance, residual couple, added
    mass, weight count). Fully deterministic."""
    mid = source.length_mm / 2.0
    radius = source.radius_mm
    base = pins + locked_points
    u0, m0 = sum_vectors(base, mid)

    locked_count = Counter(p.ref for p in locked_points)
    available = {
        t.id: t.quantity - locked_count.get(t.id, 0) for t in spec.inventory
    }
    dia_by_id = {t.id: t.diameter_mm for t in spec.inventory}
    locked_with_dia = [(p, dia_by_id[p.ref] / 2.0) for p in locked_points]
    placements = _allowed_placements(
        spec, source, limits, locked_with_dia, available
    )

    def evaluate(plan: tuple[_Placement, ...]) -> tuple:
        u, m = u0, m0
        added = 0.0
        for pl in plan:
            v = pl.mass_g * radius * complex(
                math.cos(math.radians(pl.angle_deg)),
                math.sin(math.radians(pl.angle_deg)),
            )
            u += v
            m += v * (pl.axial_mm - mid)
            added += pl.mass_g
        # round at 1e-9 so mathematically equal residuals (float noise ~1e-16)
        # tie and the ranking falls through to the next key
        return (
            round(abs(u), 9),
            round(abs(m), 9),
            round(added, 9),
            len(plan),
            _plan_key(plan),
        )

    def fits(plan: tuple[_Placement, ...], pl: _Placement) -> bool:
        used = Counter(p.weight_id for p in plan)
        if used[pl.weight_id] >= available[pl.weight_id]:
            return False
        return not any(
            _surface_distance(source, pl.point(radius), q.point(radius))
            - (pl.diameter_mm + q.diameter_mm) / 2.0
            < -EPS
            for q in plan
        )

    # evaluated plans: plan -> objective (levels 1-2 exhaustive, then beamed)
    scored: dict[tuple[_Placement, ...], tuple] = {(): evaluate(())}
    evaluated = 0

    def add(plan: tuple[_Placement, ...]) -> None:
        nonlocal evaluated
        if plan not in scored:
            scored[plan] = evaluate(plan)
            evaluated += 1

    singles = [(pl,) for pl in placements]
    if limits.max_weights >= 1:
        for plan in singles:
            add(plan)

    if limits.max_weights >= 2:
        pairs: list[tuple[_Placement, ...]] = []
        for i, p in enumerate(placements):
            for j in range(i, len(placements)):
                q = placements[j]
                if fits((p,), q):
                    pairs.append(tuple(sorted((p, q), key=_placement_key)))
        for plan in pairs:
            add(plan)
        beam = sorted(pairs, key=lambda p: scored[p])[:BEAM_WIDTH]
    else:
        beam = sorted(singles, key=lambda p: scored[p])[:BEAM_WIDTH]

    for _level in range(3, limits.max_weights + 1):
        nxt: list[tuple[_Placement, ...]] = []
        for plan in beam:
            for pl in placements:
                if fits(plan, pl):
                    new = tuple(sorted(plan + (pl,), key=_placement_key))
                    if new not in scored:
                        nxt.append(new)
                        add(new)
        if not nxt:
            break
        beam = sorted(nxt, key=lambda p: scored[p])[:BEAM_WIDTH]

    plans = sorted(scored, key=lambda p: scored[p])

    candidates: list[BalanceCandidate] = []
    for rank, plan in enumerate(plans[: limits.max_candidates], start=1):
        points = base + [pl.point(radius) for pl in plan]
        imb = compute_imbalance(spec, source, points)
        u, m = sum_vectors(points, mid)
        added = sum(pl.mass_g for pl in plan)
        candidates.append(
            BalanceCandidate(
                rank=rank,
                weights=[
                    WeightPlacement(
                        plane=pl.plane,
                        weight_id=pl.weight_id,
                        angle_deg=r6(pl.angle_deg),
                        axial_mm=r6(pl.axial_mm),
                        mass_g=r6(pl.mass_g),
                        diameter_mm=r6(pl.diameter_mm),
                    )
                    for pl in plan
                ],
                residual=Residual(
                    static_gmm=r6(abs(u)),
                    couple_gmm2=r6(abs(m)),
                    added_mass_g=r6(added),
                    weight_count=len(plan),
                    within_limit=abs(u) <= spec.residual_limit_gmm + EPS,
                ),
                imbalance=imb,
            )
        )
    return candidates, evaluated
