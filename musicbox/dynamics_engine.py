"""Deterministic powertrain dynamics engine.

The trial snapshot provides pin design phases from a frozen pin-arrangement
version; the engine integrates the cylinder angular velocity with a fixed time
step. Drive torque is the mainspring curve (interpolated at the released-turns
state, reflected through the gear ratio and efficiency), drag is the sampled
governor curve scaled by a coefficient, and every pin crossing subtracts the
reed pluck energy as an impulsive kinetic-energy loss.

SI internally: torque N*m, energy J, inertia kg*m^2; inputs are converted from
mN*m, uJ and g*cm^2. Everything here is a pure function of its inputs: no
clocks, no randomness, no ambient state, so recomputing a frozen plan yields
identical output.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import product

from .dynamics_models import (
    DynamicsPin,
    DynamicsSpec,
    PluckEventOut,
    SampledCurve,
    ScenarioParams,
    ScenarioResolved,
    SimCurve,
    SimResult,
    SimSummary,
    ViolationLoc,
)

# unit conversion to SI
TORQUE_MNM_TO_NM = 1e-3
ENERGY_UJ_TO_J = 1e-6
INERTIA_GCM2_TO_KGM2 = 1e-7

EPS = 1e-9
MAX_CURVE_POINTS = 1200
MAX_COMBINATIONS = 5000
# run length guard: stop (as a stall) past this multiple of the song duration
TIME_BUDGET_MULTIPLE = 10.0


class DynamicsError(ValueError):
    """Raised when a dynamics request contradicts its source version, its
    curves are unusable, or the mainspring cannot cover the tune."""


def r6(x: float) -> float:
    v = round(float(x), 6)
    return 0.0 if v == 0 else v


# ---------------------------------------------------------------------------
# Source snapshot (from a frozen pin-arrangement version)
# ---------------------------------------------------------------------------


@dataclass
class PluckEvent:
    """One pin phase crossing (several pins share an event at the same time)."""

    index: int
    note_ids: list[str]
    pitches: list[int]
    design_time_s: float
    phase_rev: float
    energy_J: float = 0.0


@dataclass
class DynamicsSource:
    version_id: int
    content_hash: str
    design_rpm: float
    pins: list[DynamicsPin]
    events: list[PluckEvent] = field(default_factory=list)

    @property
    def required_cylinder_revolutions(self) -> float:
        return self.events[-1].phase_rev if self.events else 0.0

    def snapshot_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "content_hash": self.content_hash,
            "design_rpm": self.design_rpm,
            "pins": [p.model_dump(mode="json") for p in self.pins],
        }

    @staticmethod
    def from_snapshot_dict(d: dict) -> "DynamicsSource":
        pins = [DynamicsPin(**p) for p in d["pins"]]
        return build_source(d["version_id"], d["content_hash"], d["design_rpm"], pins)


def source_from_version_row(row) -> DynamicsSource:
    request = json.loads(row["request_json"])
    result = json.loads(row["result_json"])
    pins = [
        DynamicsPin(
            note_id=p["note_id"],
            pitch=p["pitch"],
            design_time_s=p["time_seconds"],
            angle_deg=p["angle_deg"],
            phase_rev=p["time_seconds"] * request["arrangement"]["cylinder"]["rpm"] / 60.0,
        )
        for p in result["pins"]
    ]
    return build_source(
        row["id"],
        row["content_hash"],
        request["arrangement"]["cylinder"]["rpm"],
        pins,
    )


def build_source(
    version_id: int, content_hash: str, design_rpm: float, pins: list[DynamicsPin]
) -> DynamicsSource:
    """Order pins by design time and group coincident phases into events."""
    pins = sorted(pins, key=lambda p: (p.design_time_s, p.note_id))
    events: list[PluckEvent] = []
    for p in pins:
        if events and abs(p.phase_rev - events[-1].phase_rev) <= EPS:
            events[-1].note_ids.append(p.note_id)
            events[-1].pitches.append(p.pitch)
            events[-1].design_time_s = p.design_time_s
        else:
            events.append(
                PluckEvent(
                    index=len(events),
                    note_ids=[p.note_id],
                    pitches=[p.pitch],
                    design_time_s=p.design_time_s,
                    phase_rev=p.phase_rev,
                )
            )
    for i, e in enumerate(events):
        e.index = i
    return DynamicsSource(
        version_id=version_id,
        content_hash=content_hash,
        design_rpm=design_rpm,
        pins=pins,
        events=events,
    )


# ---------------------------------------------------------------------------
# Spec validation / derived data
# ---------------------------------------------------------------------------


def total_ratio(spec: DynamicsSpec) -> float:
    return math.prod(g.ratio for g in spec.gear_train)


def total_efficiency(spec: DynamicsSpec) -> float:
    return math.prod(g.efficiency for g in spec.gear_train)


def spring_span_turns(spec: DynamicsSpec) -> float:
    return spec.spring_torque.x[-1] - spec.spring_torque.x[0]


def validate_spec(spec: DynamicsSpec, source: DynamicsSource) -> None:
    """Cross-validate the spec against the source pins. Pydantic already
    guarantees strictly-increasing curve abscissae, positive parameters and
    declared units; here we check pin coverage and spring travel."""
    used = {p.pitch for p in source.pins}
    missing = sorted(used - set(spec.pluck_energy_uJ))
    if missing:
        raise DynamicsError(
            f"pluck_energy_uJ missing reeds for pitches: {missing}"
        )
    ratio = total_ratio(spec)
    required_barrel = source.required_cylinder_revolutions / ratio
    span = spring_span_turns(spec)
    if required_barrel > span + EPS:
        raise DynamicsError(
            f"usable mainspring travel {r6(span)} turns is insufficient: "
            f"the tune needs {r6(required_barrel)} barrel turns "
            f"({r6(source.required_cylinder_revolutions)} cylinder revolutions "
            f"at ratio {r6(ratio)})"
        )


def derived_info(spec: DynamicsSpec, source: DynamicsSource) -> dict:
    ratio = total_ratio(spec)
    return {
        "design_rpm": r6(source.design_rpm),
        "total_ratio": r6(ratio),
        "total_efficiency": r6(total_efficiency(spec)),
        "inertia_kg_m2": r6(spec.base_inertia_g_cm2 * INERTIA_GCM2_TO_KGM2),
        "required_cylinder_revolutions": r6(source.required_cylinder_revolutions),
        "required_barrel_turns": r6(source.required_cylinder_revolutions / ratio),
        "available_spring_turns": r6(spring_span_turns(spec)),
        "pluck_event_count": len(source.events),
        "pin_count": len(source.pins),
    }


def event_energies(spec: DynamicsSpec, events: list[PluckEvent]) -> None:
    for e in events:
        e.energy_J = (
            sum(spec.pluck_energy_uJ[p] for p in e.pitches) * ENERGY_UJ_TO_J
        )


# ---------------------------------------------------------------------------
# Curve lookup
# ---------------------------------------------------------------------------


def interp(curve: SampledCurve, x: float) -> float:
    """Piecewise-linear interpolation; ordinates must be non-negative."""
    xs, ys = curve.x, curve.y
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    f = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + f * (ys[hi] - ys[lo])


# ---------------------------------------------------------------------------
# Fixed-step integration
# ---------------------------------------------------------------------------


def _rpm(omega: float) -> float:
    return omega * 60.0 / (2.0 * math.pi)


def simulate(
    spec: DynamicsSpec,
    source: DynamicsSource,
    scenario: ScenarioParams,
    keep_curves: bool = True,
) -> SimResult:
    """Integrate cylinder angular velocity at the fixed step ``spec.dt_s``.

    Each step advances the phase; pin phases crossed inside the step receive
    their pluck impulse (kinetic-energy subtraction) at the linearly
    interpolated crossing instant. The first stall, overspeed and beat-drift
    occurrence is located by pin index and time."""
    ratio = scenario.gear_ratio if scenario.gear_ratio is not None else total_ratio(spec)
    eta = total_efficiency(spec)
    J = (spec.base_inertia_g_cm2 + scenario.flywheel_inertia_g_cm2) * INERTIA_GCM2_TO_KGM2
    dt = spec.dt_s
    design_omega = 2.0 * math.pi * source.design_rpm / 60.0
    stall_omega = spec.stall_rpm_ratio * design_omega
    over_omega = spec.overspeed_rpm_ratio * design_omega

    event_energies(spec, source.events)

    prewind = scenario.prewind_turns
    span = spring_span_turns(spec)
    if prewind > span + EPS:
        raise DynamicsError(
            f"prewind {r6(prewind)} turns exceeds usable mainspring travel "
            f"{r6(span)} turns"
        )
    x0 = spec.spring_torque.x[0]

    t = 0.0
    theta = 0.0  # cylinder phase in radians (unwrapped)
    omega = design_omega
    used_barrel = 0.0

    ts: list[float] = []
    omegas: list[float] = []
    margins: list[float] = []

    out_events: list[PluckEventOut] = []
    first: dict[str, ViolationLoc | None] = {
        "stall": None,
        "overspeed": None,
        "beat_drift": None,
    }
    stall = overspeed = False
    drift_count = 0
    min_omega = omega
    min_margin = math.inf
    interval_min_omega: list[float] = []  # running minima, closed at each event
    segment_min = omega  # min after the last pluck, until the next

    def spring_drive(barrel_turns_used: float) -> float:
        reserve = prewind - barrel_turns_used
        if reserve <= -EPS:
            return 0.0  # mainspring run down
        coord = x0 + min(max(reserve, 0.0), span)
        t_spring = interp(spec.spring_torque, coord) * TORQUE_MNM_TO_NM
        # power conservation: T_cyl*omega_cyl = eta*T_barrel*omega_barrel
        return t_spring / ratio * eta

    def margin_at(w: float, used: float) -> float:
        drive = spring_drive(used)
        drag = scenario.governor_coefficient * interp(
            spec.governor_drag, _rpm(w)
        ) * TORQUE_MNM_TO_NM
        return drive - drag

    def sample(w: float, used: float) -> None:
        if not keep_curves:
            return
        ts.append(r6(t))
        omegas.append(r6(_rpm(w)))
        margins.append(r6(margin_at(w, used) / TORQUE_MNM_TO_NM))

    def mark_stall(idx: int | None, w: float, detail: str) -> None:
        nonlocal stall
        stall = True
        if first["stall"] is None:
            ids = source.events[idx].note_ids if idx is not None else []
            first["stall"] = ViolationLoc(
                kind="stall",
                pin_index=idx,
                note_ids=ids,
                time_s=r6(t),
                rpm=r6(_rpm(w)),
                detail=detail,
            )

    sample(omega, used_barrel)

    design_song_s = (
        source.events[-1].design_time_s if source.events else 0.0
    )
    max_steps = int(design_song_s * TIME_BUDGET_MULTIPLE / dt) + 16

    steps = 0
    fired_count = 0
    n_events = len(source.events)
    while steps < max_steps:
        steps += 1
        # Integrate the fixed dt step as constant-acceleration kinematics,
        # splitting it into sub-segments at every pin phase crossing. The
        # post-pluck velocity is the initial velocity of the next sub-segment,
        # so a pluck's energy loss carries into later steps (no snap-back).
        cur_t = t
        cur_theta = theta
        cur_omega = omega
        cur_used = used_barrel
        h_remaining = dt
        stall_in_step = False

        while True:
            margin = margin_at(cur_omega, cur_used)
            min_margin = min(min_margin, margin)
            alpha = margin / J

            # deceleration to a full stop inside this sub-segment
            if alpha < 0.0 and cur_omega + alpha * h_remaining <= 0.0:
                h_stop = cur_omega / -alpha
                cur_t += h_stop
                cur_theta += cur_omega * h_stop + 0.5 * alpha * h_stop * h_stop
                cur_omega = 0.0
                cur_used = cur_theta / (2.0 * math.pi) / ratio
                t, theta, omega, used_barrel = cur_t, cur_theta, cur_omega, cur_used
                min_omega = min(min_omega, cur_omega)
                mark_stall(
                    source.events[fired_count - 1].index if fired_count else None,
                    cur_omega,
                    f"speed dropped to 0 rpm (stall threshold "
                    f"{r6(_rpm(stall_omega))} rpm)",
                )
                stall_in_step = True
                break

            end_omega = max(0.0, cur_omega + alpha * h_remaining)
            end_theta = (
                cur_theta
                + cur_omega * h_remaining
                + 0.5 * alpha * h_remaining * h_remaining
            )

            next_event = (
                source.events[fired_count] if fired_count < n_events else None
            )
            crosses = (
                next_event is not None
                and next_event.phase_rev * 2.0 * math.pi <= end_theta + EPS
            )
            if not crosses:
                cur_t += h_remaining
                cur_theta = end_theta
                cur_omega = end_omega
                cur_used = cur_theta / (2.0 * math.pi) / ratio
                break

            e = next_event
            cross_theta = e.phase_rev * 2.0 * math.pi
            dtheta = cross_theta - cur_theta
            if abs(alpha) <= EPS:
                tau = dtheta / max(cur_omega, EPS)
            else:
                # 0.5*alpha*tau^2 + omega*tau = dtheta  (positive root)
                disc = cur_omega * cur_omega + 2.0 * alpha * dtheta
                tau = (-cur_omega + math.sqrt(max(0.0, disc))) / alpha
            tau = min(max(tau, 0.0), h_remaining)
            t_cross = cur_t + tau
            w_before = max(0.0, cur_omega + alpha * tau)
            w2 = w_before * w_before - 2.0 * e.energy_J / J
            w_after = math.sqrt(max(0.0, w2))

            # close the previous interval with the minimum seen up to crossing
            segment_min = min(segment_min, w_before)
            if out_events:
                interval_min_omega.append(segment_min)
            segment_min = w_after

            # beat drift of this interval vs the designed interval; the first
            # interval is measured from run start (t = 0 / design time 0)
            if out_events:
                prev = out_events[-1]
                design_gap = e.design_time_s - prev.design_time_s
                actual_gap = t_cross - prev.time_s
            else:
                design_gap = e.design_time_s
                actual_gap = t_cross
            drift = (
                abs(actual_gap - design_gap) / design_gap
                if design_gap > EPS
                else abs(actual_gap - design_gap)
            )
            drift = max(0.0, drift)
            if drift > spec.beat_drift_limit + EPS:
                drift_count += 1
                if first["beat_drift"] is None:
                    first["beat_drift"] = ViolationLoc(
                        kind="beat_drift",
                        pin_index=e.index,
                        note_ids=list(e.note_ids),
                        time_s=r6(t_cross),
                        rpm=r6(_rpm(w_after)),
                        detail=(
                            f"interval beat drift {r6(drift)} exceeds limit "
                            f"{spec.beat_drift_limit}"
                        ),
                    )

            out_events.append(
                PluckEventOut(
                    index=e.index,
                    note_ids=list(e.note_ids),
                    pitches=list(e.pitches),
                    phase_rev=r6(e.phase_rev),
                    design_time_s=r6(e.design_time_s),
                    time_s=r6(t_cross),
                    rpm_before=r6(_rpm(w_before)),
                    rpm_after=r6(_rpm(w_after)),
                    min_rpm_until_next=0.0,  # filled when the next event closes
                    beat_drift_ratio=r6(drift),
                    energy_uJ=r6(e.energy_J / ENERGY_UJ_TO_J),
                )
            )
            fired_count += 1
            cur_t = t_cross
            cur_theta = cross_theta
            cur_omega = w_after
            cur_used = cur_theta / (2.0 * math.pi) / ratio
            h_remaining -= tau
            min_omega = min(min_omega, w_before, w_after)

            if w_after <= stall_omega + EPS:
                t, theta, omega, used_barrel = cur_t, cur_theta, cur_omega, cur_used
                mark_stall(
                    e.index,
                    w_after,
                    f"post-pluck speed {r6(_rpm(w_after))} rpm at/below stall "
                    f"threshold {r6(_rpm(stall_omega))} rpm",
                )
                stall_in_step = True
                break

        # commit the integrated step state
        t, theta, omega, used_barrel = cur_t, cur_theta, cur_omega, cur_used
        segment_min = min(segment_min, omega)
        min_omega = min(min_omega, omega)

        if stall_in_step:
            sample(omega, used_barrel)
            break

        if omega <= stall_omega + EPS:
            mark_stall(
                source.events[fired_count - 1].index if fired_count else None,
                omega,
                f"speed {r6(_rpm(omega))} rpm at/below stall threshold "
                f"{r6(_rpm(stall_omega))} rpm",
            )
            sample(omega, used_barrel)
            break
        if omega >= over_omega - EPS and first["overspeed"] is None:
            overspeed = True
            first["overspeed"] = ViolationLoc(
                kind="overspeed",
                pin_index=(
                    source.events[fired_count - 1].index if fired_count else None
                ),
                note_ids=(
                    list(source.events[fired_count - 1].note_ids)
                    if fired_count
                    else []
                ),
                time_s=r6(t),
                rpm=r6(_rpm(omega)),
                detail=(
                    f"speed {r6(_rpm(omega))} rpm reaches overspeed threshold "
                    f"{r6(_rpm(over_omega))} rpm"
                ),
            )

        sample(omega, used_barrel)
        if fired_count == n_events:
            break
    else:
        # step budget exhausted without reaching all pins: treat as a stall
        mark_stall(
            source.events[fired_count - 1].index if fired_count else None,
            omega,
            "integration step budget exhausted before the tune finished",
        )

    # close the trailing interval
    if out_events:
        interval_min_omega.append(segment_min)
        for ev, lo_w in zip(out_events, interval_min_omega):
            ev.min_rpm_until_next = r6(_rpm(lo_w))

    completed = len(out_events) == len(source.events) and not stall
    margin_min = 0.0 if min_margin is math.inf else min_margin / TORQUE_MNM_TO_NM
    max_drift = max((e.beat_drift_ratio for e in out_events), default=0.0)
    violation_count = int(stall) + int(overspeed) + drift_count

    curves = SimCurve(t_s=ts, rpm=omegas, torque_margin_mNm=margins)
    if keep_curves and len(ts) > MAX_CURVE_POINTS:
        stride = -(-len(ts) // MAX_CURVE_POINTS)  # ceil division
        curves = SimCurve(
            t_s=ts[::stride],
            rpm=omegas[::stride],
            torque_margin_mNm=margins[::stride],
        )

    summary = SimSummary(
        completed=completed,
        total_time_s=r6(t),
        pins_plucked=len(out_events),
        pin_count=len(source.events),
        used_spring_turns=r6(used_barrel),
        remaining_spring_turns=r6(max(0.0, prewind - used_barrel)),
        min_rpm=r6(_rpm(min_omega)),
        min_torque_margin_mNm=r6(margin_min),
        max_beat_drift_ratio=r6(max_drift),
        stall=stall,
        overspeed=overspeed,
        drift_violations=drift_count,
        violation_count=violation_count,
    )
    return SimResult(
        design_rpm=r6(source.design_rpm),
        total_efficiency=r6(eta),
        total_inertia_g_cm2=r6(
            spec.base_inertia_g_cm2 + scenario.flywheel_inertia_g_cm2
        ),
        dt_s=dt,
        scenario=ScenarioResolved(
            prewind_turns=r6(prewind),
            governor_coefficient=r6(scenario.governor_coefficient),
            flywheel_inertia_g_cm2=r6(scenario.flywheel_inertia_g_cm2),
            gear_ratio=r6(ratio),
        ),
        summary=summary,
        curves=curves,
        pluck_events=out_events,
        first_violations=first,
    )


# ---------------------------------------------------------------------------
# Scenario grid search
# ---------------------------------------------------------------------------


def search_scenarios(
    spec: DynamicsSpec,
    source: DynamicsSource,
    ratios: list[float],
    prewinds: list[float],
    coefficients: list[float],
    inertias: list[float],
    max_candidates: int,
) -> tuple[list[dict], int]:
    """Enumerate the parameter grid, simulate every feasible point and rank by
    (violation count, max beat drift, larger torque margin, larger remaining
    spring travel). Deterministic tie-breaking by the scenario tuple."""
    n_combos = len(ratios) * len(prewinds) * len(coefficients) * len(inertias)
    if n_combos > MAX_COMBINATIONS:
        raise DynamicsError(
            f"scenario space too large ({n_combos} combinations, max "
            f"{MAX_COMBINATIONS}); narrow the candidate grids"
        )
    span = spring_span_turns(spec)
    evaluated = 0
    scored: list[tuple[tuple, dict]] = []
    for ratio, prewind, kgov, finertia in product(
        sorted(set(ratios)), sorted(set(prewinds)),
        sorted(set(coefficients)), sorted(set(inertias)),
    ):
        if prewind > span + EPS:
            continue  # wind reserve beyond the usable travel: not a valid point
        scenario = ScenarioParams(
            prewind_turns=prewind,
            governor_coefficient=kgov,
            flywheel_inertia_g_cm2=finertia,
            gear_ratio=ratio,
        )
        result = simulate(spec, source, scenario, keep_curves=False)
        evaluated += 1
        s = result.summary
        rank_key = (
            s.violation_count,
            round(s.max_beat_drift_ratio, 9),
            round(-s.min_torque_margin_mNm, 9),
            round(-s.remaining_spring_turns, 9),
            r6(ratio),
            r6(prewind),
            r6(kgov),
            r6(finertia),
        )
        scored.append(
            (
                rank_key,
                {
                    "scenario": result.scenario,
                    "summary": s,
                    "first_violations": result.first_violations,
                    "pluck_min_rpm": [e.min_rpm_until_next for e in result.pluck_events],
                },
            )
        )
    scored.sort(key=lambda item: item[0])
    return [c for _, c in scored[:max_candidates]], evaluated
