"""Deterministic placement engine.

Pipeline: score notes -> (transpose / quantize / tempo / delete) -> cylinder
coordinates -> conflict diagnostics -> (optionally) search over solutions.

Everything here is a pure function of its inputs: no clocks, no randomness,
no ambient state, so recomputing a frozen version yields identical output.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .models import (
    ArrangementRequest,
    Candidate,
    DerivedInfo,
    Diagnostics,
    Issue,
    Metrics,
    MissingNote,
    NoteIn,
    PinMargins,
    PinOut,
    SearchLimits,
    SolutionSpec,
)

EPS = 1e-9
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ISSUE_KINDS = (
    "missing_pitch",
    "chord_collision",
    "seam_crossing",
    "rebound_too_dense",
    "pin_clearance",
)
_KIND_ORDER = {k: i for i, k in enumerate(ISSUE_KINDS)}


def pitch_name(midi: int) -> str:
    return f"{PITCH_NAMES[midi % 12]}{midi // 12 - 1}"


def r6(x: float) -> float:
    """Round for stable, tidy output; normalise -0.0 to 0.0."""
    v = round(float(x), 6)
    return 0.0 if v == 0 else v


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class TxNote:
    """A note after applying a SolutionSpec, placed on the cylinder."""

    id: str
    beat: float  # after quantize
    pitch: int  # after transpose
    duration: float
    locked: bool
    orig_beat: float
    orig_pitch: int
    time_s: float
    angle_deg: float
    x_mm: float
    axial_mm: float | None  # None => pitch has no reed on the comb


def circumference(req: ArrangementRequest) -> float:
    return math.pi * req.cylinder.diameter_mm


def seconds_per_beat(req: ArrangementRequest, tempo_factor: float = 1.0) -> float:
    return 60.0 / (req.bpm * tempo_factor)


def transform_note(req: ArrangementRequest, sol: SolutionSpec, n: NoteIn) -> TxNote:
    """Apply transposition/quantization (unlocked notes only) and project the
    note onto the unrolled cylinder: angle around the circumference and axial
    position of its reed. The tempo factor scales the global clock only."""
    beat, pitch = n.beat, n.pitch
    if not n.locked:
        pitch += sol.transpose_semitones
        if sol.quantize_grid_beats:
            g = sol.quantize_grid_beats
            beat = math.floor(beat / g + 0.5) * g  # round half up, deterministic
    t = beat * seconds_per_beat(req, sol.tempo_factor)
    angle = (t * 6.0 * req.cylinder.rpm) % 360.0  # 6 deg/s per rpm
    axial = next((r.axial_mm for r in req.comb if r.pitch == pitch), None)
    return TxNote(
        id=n.id,
        beat=beat,
        pitch=pitch,
        duration=n.duration,
        locked=n.locked,
        orig_beat=n.beat,
        orig_pitch=n.pitch,
        time_s=t,
        angle_deg=angle,
        x_mm=angle / 360.0 * circumference(req),
        axial_mm=axial,
    )


def place(
    req: ArrangementRequest, sol: SolutionSpec
) -> tuple[list[TxNote], list[TxNote], list[TxNote]]:
    """Place every kept note. Returns (placed, missing, kept) where ``missing``
    notes have no reed for their (possibly transposed) pitch."""
    deleted = set(sol.deleted_note_ids)
    placed: list[TxNote] = []
    missing: list[TxNote] = []
    for n in req.notes:
        if n.id in deleted:
            continue
        tx = transform_note(req, sol, n)
        (placed if tx.axial_mm is not None else missing).append(tx)
    placed.sort(key=lambda p: (p.time_s, p.id))
    missing.sort(key=lambda p: (p.time_s, p.id))
    return placed, missing, placed + missing


def deleted_markers(req: ArrangementRequest, sol: SolutionSpec) -> list[TxNote]:
    """Transformed positions of deleted notes (for drawing red X marks)."""
    deleted = set(sol.deleted_note_ids)
    marks = [
        transform_note(req, sol, n)
        for n in req.notes
        if n.id in deleted and not n.locked
    ]
    return [m for m in marks if m.axial_mm is not None]


def derived_info(req: ArrangementRequest, tempo_factor: float = 1.0) -> DerivedInfo:
    spb = seconds_per_beat(req, tempo_factor)
    dps = 6.0 * req.cylinder.rpm
    end_beat = max(n.beat + n.duration for n in req.notes)
    song_s = end_beat * spb
    return DerivedInfo(
        circumference_mm=r6(circumference(req)),
        seconds_per_beat=r6(spb),
        degrees_per_second=r6(dps),
        degrees_per_beat=r6(dps * spb),
        song_seconds=r6(song_s),
        revolutions=r6(song_s * req.cylinder.rpm / 60.0),
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _seam_limit_deg(req: ArrangementRequest) -> float:
    """Angular distance from the seam line within which no pin centre may fall."""
    cons = req.constraints
    half_zone = cons.seam_zone_mm / 2.0 + cons.pin_diameter_mm / 2.0
    return half_zone / circumference(req) * 360.0


def _seam_distance_deg(angle: float) -> float:
    return min(angle, 360.0 - angle)


def _surface_net_mm(req: ArrangementRequest, a: TxNote, b: TxNote) -> float:
    """Edge-to-edge distance between two pins on the unrolled surface."""
    dx = abs(a.axial_mm - b.axial_mm)  # type: ignore[operator]  # placed notes only
    d_ang = abs(a.angle_deg - b.angle_deg) % 360.0
    d_ang = min(d_ang, 360.0 - d_ang)
    dy = circumference(req) * d_ang / 360.0
    return math.hypot(dx, dy) - req.constraints.pin_diameter_mm


def diagnose(
    req: ArrangementRequest, placed: list[TxNote], missing: list[TxNote]
) -> list[Issue]:
    cons = req.constraints
    circ = circumference(req)
    issues: list[Issue] = []

    for m in missing:
        issues.append(
            Issue(
                kind="missing_pitch",
                beat=r6(m.beat),
                note_ids=[m.id],
                message=f"pitch {m.pitch} ({pitch_name(m.pitch)}) has no reed on the comb",
                angle_deg=r6(m.angle_deg),
            )
        )

    seam_limit = _seam_limit_deg(req)
    for p in placed:
        d = _seam_distance_deg(p.angle_deg)
        if d < seam_limit - EPS:
            issues.append(
                Issue(
                    kind="seam_crossing",
                    beat=r6(p.beat),
                    note_ids=[p.id],
                    message=(
                        f"pin at {r6(p.angle_deg)} deg lies inside the seam forbidden zone"
                    ),
                    angle_deg=r6(p.angle_deg),
                    axial_mm=r6(p.axial_mm),  # type: ignore[arg-type]
                    measured=r6(d / 360.0 * circ),
                    required=r6(seam_limit / 360.0 * circ),
                    unit="mm",
                )
            )

    by_pitch: dict[int, list[TxNote]] = defaultdict(list)
    for p in placed:
        by_pitch[p.pitch].append(p)
    for pitch, group in by_pitch.items():
        group.sort(key=lambda p: (p.time_s, p.id))
        for a, b in zip(group, group[1:]):
            gap = b.time_s - a.time_s
            if gap < cons.min_rebound_seconds - EPS:
                issues.append(
                    Issue(
                        kind="rebound_too_dense",
                        beat=r6(a.beat),
                        note_ids=[a.id, b.id],
                        message=(
                            f"reed {pitch_name(pitch)} retriggered after {r6(gap)}s "
                            f"(< {cons.min_rebound_seconds}s)"
                        ),
                        angle_deg=r6(b.angle_deg),
                        axial_mm=r6(b.axial_mm),  # type: ignore[arg-type]
                        measured=r6(gap),
                        required=cons.min_rebound_seconds,
                        unit="s",
                    )
                )

    need_center = cons.pin_diameter_mm + cons.min_clearance_mm
    for i in range(len(placed)):
        a = placed[i]
        for j in range(i + 1, len(placed)):
            b = placed[j]
            if abs(a.axial_mm - b.axial_mm) >= need_center:  # type: ignore[operator]
                continue  # axial separation alone satisfies clearance
            net = _surface_net_mm(req, a, b)
            if net < cons.min_clearance_mm - EPS:
                chord = abs(a.time_s - b.time_s) < 1e-6
                issues.append(
                    Issue(
                        kind="chord_collision" if chord else "pin_clearance",
                        beat=r6(a.beat),
                        note_ids=[a.id, b.id],
                        message=(
                            f"{'simultaneous pins' if chord else 'pins'} only "
                            f"{r6(net)}mm apart (< {cons.min_clearance_mm}mm)"
                        ),
                        angle_deg=r6(b.angle_deg),
                        axial_mm=r6(b.axial_mm),  # type: ignore[arg-type]
                        measured=r6(net),
                        required=cons.min_clearance_mm,
                        unit="mm",
                    )
                )

    issues.sort(key=lambda i: (i.beat, _KIND_ORDER[i.kind], i.note_ids))
    return issues


def build_diagnostics(issues: list[Issue]) -> Diagnostics:
    counts: Counter[str] = Counter(i.kind for i in issues)
    first: dict[str, Issue | None] = {k: None for k in ISSUE_KINDS}
    for issue in issues:  # already sorted by beat
        if first[issue.kind] is None:
            first[issue.kind] = issue
    return Diagnostics(
        ok=not issues,
        issue_counts={k: counts.get(k, 0) for k in ISSUE_KINDS},
        first_by_kind=first,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Pins, margins, metrics
# ---------------------------------------------------------------------------


def build_pins(req: ArrangementRequest, placed: list[TxNote]) -> list[PinOut]:
    cons = req.constraints
    circ = circumference(req)
    seam_limit = _seam_limit_deg(req)

    clearance_margin: dict[str, float] = {}
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a, b = placed[i], placed[j]
            margin = _surface_net_mm(req, a, b) - cons.min_clearance_mm
            for p in (a, b):
                if p.id not in clearance_margin or margin < clearance_margin[p.id]:
                    clearance_margin[p.id] = margin

    rebound_margin: dict[str, float] = {}
    by_pitch: dict[int, list[TxNote]] = defaultdict(list)
    for p in placed:
        by_pitch[p.pitch].append(p)
    for group in by_pitch.values():
        group.sort(key=lambda p: (p.time_s, p.id))
        for k, p in enumerate(group):
            neighbours = []
            if k:
                neighbours.append(group[k - 1])
            if k + 1 < len(group):
                neighbours.append(group[k + 1])
            slacks = [abs(p.time_s - o.time_s) - cons.min_rebound_seconds for o in neighbours]
            if slacks:
                rebound_margin[p.id] = min(slacks)

    pins: list[PinOut] = []
    for p in placed:
        seam_margin = (_seam_distance_deg(p.angle_deg) - seam_limit) / 360.0 * circ
        cm = clearance_margin.get(p.id)
        rm = rebound_margin.get(p.id)
        ok = seam_margin >= -EPS and (cm is None or cm >= -EPS) and (rm is None or rm >= -EPS)
        pins.append(
            PinOut(
                note_id=p.id,
                pitch=p.pitch,
                pitch_name=pitch_name(p.pitch),
                locked=p.locked,
                beat=r6(p.beat),
                duration=r6(p.duration),
                time_seconds=r6(p.time_s),
                angle_deg=r6(p.angle_deg),
                x_mm=r6(p.x_mm),
                axial_mm=r6(p.axial_mm),  # type: ignore[arg-type]
                margins=PinMargins(
                    clearance_mm=r6(cm) if cm is not None else None,
                    rebound_s=r6(rm) if rm is not None else None,
                    seam_mm=r6(seam_margin),
                    ok=ok,
                ),
            )
        )
    return pins


def min_clearance(req: ArrangementRequest, placed: list[TxNote]) -> float | None:
    best = None
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            net = _surface_net_mm(req, placed[i], placed[j])
            best = net if best is None else min(best, net)
    return r6(best) if best is not None else None


def compute_metrics(
    req: ArrangementRequest, sol: SolutionSpec, kept: list[TxNote], placed: list[TxNote]
) -> Metrics:
    spb_orig = seconds_per_beat(req, 1.0)
    spb_new = seconds_per_beat(req, sol.tempo_factor)
    pitch_dev = (
        sum(abs(t.pitch - t.orig_pitch) for t in kept) / len(kept) if kept else 0.0
    )
    rhythm_err = (
        max(abs(t.beat * spb_new - t.orig_beat * spb_orig) for t in kept) if kept else 0.0
    )
    return Metrics(
        deleted_count=len(sol.deleted_note_ids),
        pitch_deviation_semitones=r6(pitch_dev),
        rhythm_error_seconds=r6(rhythm_err),
        min_clearance_mm=min_clearance(req, placed),
    )


def missing_notes_out(missing: list[TxNote]) -> list[MissingNote]:
    return [
        MissingNote(
            note_id=m.id,
            pitch=m.pitch,
            pitch_name=pitch_name(m.pitch),
            beat=r6(m.beat),
            angle_deg=r6(m.angle_deg),
            reason="no reed on the comb for this pitch",
        )
        for m in missing
    ]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _tempo_factors(limits: SearchLimits) -> list[float]:
    f, s = limits.tempo_float_percent, limits.tempo_step_percent
    steps = int(round(2 * f / s))
    return [round(1.0 + (-f + k * s) / 100.0, 6) for k in range(steps + 1)]


def _quantize_feasible(kept: list[TxNote], limits: SearchLimits, grid: float | None) -> bool:
    if grid is None:
        return True
    return all(
        t.locked or abs(t.beat - t.orig_beat) <= limits.max_quantize_error_beats + EPS
        for t in kept
    )


def _greedy_deletions(
    req: ArrangementRequest, sol: SolutionSpec, limits: SearchLimits
) -> tuple[SolutionSpec | None, list[TxNote], list[TxNote]]:
    """Delete conflicting unlocked notes (most-conflicting first) until the
    layout is manufacturable. Locked notes are never deleted; if only locked
    notes remain in conflict the combination is infeasible."""
    locked_by_id = {n.id: n.locked for n in req.notes}
    beat_by_id = {n.id: n.beat for n in req.notes}
    deleted: list[str] = []
    while True:
        placed, missing, _ = place(req, sol)
        issues = diagnose(req, placed, missing)
        if not issues:
            return sol, placed, missing
        counts: Counter[str] = Counter()
        for issue in issues:
            for nid in issue.note_ids:
                if not locked_by_id[nid]:
                    counts[nid] += 1
        if not counts:  # only locked notes in conflict -> unfixable
            return None, placed, missing
        victim = min(counts, key=lambda nid: (-counts[nid], beat_by_id[nid], nid))
        deleted.append(victim)
        if len(deleted) > limits.max_deletions:
            return None, placed, missing
        sol = sol.model_copy(update={"deleted_note_ids": sorted(deleted)})


def _candidate_key(c: Candidate) -> tuple:
    m, s = c.metrics, c.solution
    return (
        m.deleted_count,
        m.pitch_deviation_semitones,
        m.rhythm_error_seconds,
        -(m.min_clearance_mm if m.min_clearance_mm is not None else math.inf),
        abs(s.tempo_factor - 1.0),
        abs(s.transpose_semitones),
        s.quantize_grid_beats or 0.0,
    )


def search(req: ArrangementRequest, limits: SearchLimits) -> tuple[list[Candidate], int]:
    """Enumerate transpose x tempo x quantize combinations, repair each with
    greedy deletions, and rank by (deletions, pitch deviation, rhythm error,
    min clearance). Fully deterministic."""
    transposes = range(-limits.max_transpose_semitones, limits.max_transpose_semitones + 1)
    factors = _tempo_factors(limits)
    grids: list[float | None] = [None] + sorted(set(limits.quantize_grids_beats))

    candidates: list[Candidate] = []
    evaluated = 0
    for t in transposes:
        for f in factors:
            for g in grids:
                evaluated += 1
                sol = SolutionSpec(
                    transpose_semitones=t, tempo_factor=f, quantize_grid_beats=g
                )
                _, _, kept = place(req, sol)
                if not _quantize_feasible(kept, limits, g):
                    continue
                final, placed, _ = _greedy_deletions(req, sol, limits)
                if final is None:
                    continue
                candidates.append(
                    Candidate(
                        rank=0,
                        solution=final,
                        metrics=compute_metrics(req, final, kept, placed),
                        pins=build_pins(req, placed),
                        diagnostics=build_diagnostics([]),
                    )
                )

    candidates.sort(key=_candidate_key)
    top = candidates[: limits.max_candidates]
    for rank, c in enumerate(top, start=1):
        c.rank = rank
    return top, evaluated
