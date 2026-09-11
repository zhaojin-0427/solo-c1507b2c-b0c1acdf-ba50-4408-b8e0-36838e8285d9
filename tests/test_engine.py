"""Unit tests for the deterministic placement engine."""

from __future__ import annotations

import math

from musicbox import engine
from musicbox.models import (
    ArrangementRequest,
    SearchLimits,
    SolutionSpec,
)

from conftest import make_arrangement, note

CIRC = 60.0 * math.pi  # 188.4955592...


def req(notes) -> ArrangementRequest:
    return ArrangementRequest.model_validate(make_arrangement(notes))


def test_geometry_beat_to_angle_and_axial():
    # 120 bpm -> 0.5 s/beat; 10 rpm -> 60 deg/s -> 30 deg per beat
    r = req([note("a", 2.0, 60)])
    placed, missing, _ = engine.place(r, SolutionSpec())
    assert missing == []
    (p,) = placed
    assert p.time_s == 1.0
    assert p.angle_deg == 60.0
    assert p.x_mm == 60.0 / 360.0 * CIRC  # angle/360 * circumference
    assert p.axial_mm == 10.0


def test_geometry_wraps_full_revolutions():
    r = req([note("a", 12.0, 60)])  # 12 beats = 6 s = exactly one revolution
    (p,), _, _ = engine.place(r, SolutionSpec())
    assert p.angle_deg == 0.0


def test_tempo_factor_scales_time_and_angle():
    r = req([note("a", 2.0, 60)])
    (p,), _, _ = engine.place(r, SolutionSpec(tempo_factor=2.0))
    assert p.time_s == 0.5
    assert p.angle_deg == 30.0


def test_quantize_rounds_half_up_deterministically():
    r = req([note("a", 1.1, 60), note("b", 1.13, 62), note("c", 1.125, 64)])
    placed, _, _ = engine.place(r, SolutionSpec(quantize_grid_beats=0.25))
    beats = {p.id: p.beat for p in placed}
    assert beats == {"a": 1.0, "b": 1.25, "c": 1.25}


def test_locked_notes_are_never_adjusted():
    r = req([note("lk", 1.1, 60, locked=True), note("free", 1.1, 60)])
    placed, _, _ = engine.place(
        r, SolutionSpec(transpose_semitones=2, quantize_grid_beats=0.25)
    )
    by_id = {p.id: p for p in placed}
    assert by_id["lk"].pitch == 60 and by_id["lk"].beat == 1.1
    assert by_id["free"].pitch == 62 and by_id["free"].beat == 1.0


def test_pitch_names():
    assert engine.pitch_name(60) == "C4"
    assert engine.pitch_name(69) == "A4"
    assert engine.pitch_name(61) == "C#4"


def test_missing_pitch_detected():
    r = req([note("m", 1.0, 66)])
    placed, missing, _ = engine.place(r, SolutionSpec())
    assert placed == [] and [m.id for m in missing] == ["m"]
    issues = engine.diagnose(r, placed, missing)
    assert [i.kind for i in issues] == ["missing_pitch"]
    assert issues[0].beat == 1.0


def test_seam_crossing_detected_at_zero_angle():
    r = req([note("s", 0.0, 62)])  # beat 0 -> angle 0 -> on the seam
    placed, missing, _ = engine.place(r, SolutionSpec())
    issues = engine.diagnose(r, placed, missing)
    assert [i.kind for i in issues] == ["seam_crossing"]
    assert issues[0].measured == 0.0
    # seam limit: (10/2 + 2/2) mm of arc
    assert issues[0].required == engine.r6(6.0)


def test_seam_ok_just_outside_zone():
    # seam forbidden half-width = 6 mm of arc -> 6/C*360 = 11.459156 deg
    # beat 0.4 -> angle 12 deg -> outside
    r = req([note("s", 0.4, 62)])
    placed, missing, _ = engine.place(r, SolutionSpec())
    assert engine.diagnose(r, placed, missing) == []


def test_rebound_too_dense_detected():
    r = req([note("r1", 6.0, 64), note("r2", 6.5, 64)])  # gap 0.25 s < 0.3 s
    placed, missing, _ = engine.place(r, SolutionSpec())
    issues = engine.diagnose(r, placed, missing)
    assert [i.kind for i in issues] == ["rebound_too_dense"]
    assert issues[0].beat == 6.0
    assert issues[0].note_ids == ["r1", "r2"]
    assert issues[0].measured == 0.25


def test_chord_collision_on_neighbouring_reeds():
    r = req([note("c1", 4.0, 60), note("c2", 4.0, 61)])  # axial gap 2 mm < 3 mm
    placed, missing, _ = engine.place(r, SolutionSpec())
    issues = engine.diagnose(r, placed, missing)
    assert [i.kind for i in issues] == ["chord_collision"]
    assert issues[0].measured == 0.0  # 2 mm centre distance - 2 mm pin diameter


def test_pin_clearance_non_simultaneous():
    # beat 10 -> 300 deg; beat 10.05 -> 301.5 deg; dy = 1.5/360*C = 0.7854 mm
    r = req([note("q1", 10.0, 60), note("q2", 10.05, 61)])
    placed, missing, _ = engine.place(r, SolutionSpec())
    issues = engine.diagnose(r, placed, missing)
    assert [i.kind for i in issues] == ["pin_clearance"]
    assert issues[0].measured == engine.r6(math.hypot(2.0, CIRC * 1.5 / 360.0) - 2.0)


def test_clearance_uses_wraparound_angle():
    # angles 1 deg and 359 deg are 2 deg apart across the seam line
    r = req([note("a", 0.0333, 60), note("b", 11.9667, 61)])
    placed, missing, _ = engine.place(r, SolutionSpec())
    kinds = {i.kind for i in engine.diagnose(r, placed, missing)}
    assert "pin_clearance" in kinds  # and both also trip the seam zone


def test_first_issue_per_kind_is_earliest():
    r = req(
        [
            note("m", 1.0, 66),  # missing
            note("s", 0.0, 62),  # seam
            note("c1", 4.0, 60),
            note("c2", 4.0, 61),  # chord
            note("r1", 6.0, 64),
            note("r2", 6.5, 64),  # rebound
            note("q1", 10.0, 60),
            note("q2", 10.05, 61),  # clearance
        ]
    )
    placed, missing, _ = engine.place(r, SolutionSpec())
    diag = engine.build_diagnostics(engine.diagnose(r, placed, missing))
    assert not diag.ok
    first = diag.first_by_kind
    assert first["missing_pitch"].beat == 1.0
    assert first["seam_crossing"].beat == 0.0
    assert first["chord_collision"].beat == 4.0
    assert first["rebound_too_dense"].beat == 6.0
    assert first["pin_clearance"].beat == 10.0
    assert diag.issue_counts == {
        "missing_pitch": 1,
        "chord_collision": 1,
        "seam_crossing": 1,
        "rebound_too_dense": 1,
        "pin_clearance": 1,
    }


def test_pin_margins_on_clean_layout():
    r = req([note("a", 2.0, 60), note("b", 4.0, 62), note("c", 6.0, 64)])
    placed, _, _ = engine.place(r, SolutionSpec())
    pins = engine.build_pins(r, placed)
    assert len(pins) == 3
    for p in pins:
        assert p.margins.ok
        assert p.margins.seam_mm > 0
        assert p.margins.clearance_mm is not None and p.margins.clearance_mm > 0
        assert p.margins.rebound_s is None  # no same-reed neighbours


def test_search_finds_transposition_with_zero_deletions():
    # score sits one octave below the comb; +12 semitones makes it fit
    r = req([note("n1", 2.0, 48), note("n2", 4.0, 50), note("n3", 6.0, 52)])
    limits = SearchLimits(
        max_transpose_semitones=12,
        tempo_float_percent=2.0,
        tempo_step_percent=1.0,
        quantize_grids_beats=[0.25],
    )
    candidates, evaluated = engine.search(r, limits)
    assert evaluated == 25 * 5 * 2  # transposes x tempo factors x (none + 1 grid)
    assert candidates
    best = candidates[0]
    assert best.rank == 1
    assert best.metrics.deleted_count == 0
    assert best.solution.transpose_semitones == 12
    assert best.solution.tempo_factor == 1.0
    assert best.solution.quantize_grid_beats is None
    assert best.metrics.pitch_deviation_semitones == 12.0
    assert best.diagnostics.ok


def test_search_deletes_unlocked_but_keeps_locked():
    # locked note and free note 0.25 s apart on the same reed: only the free
    # note may be sacrificed
    r = req(
        [
            note("keep", 6.0, 64, locked=True),
            note("drop", 6.5, 64),
        ]
    )
    limits = SearchLimits(
        max_transpose_semitones=0,
        tempo_float_percent=5.0,
        tempo_step_percent=1.0,
        quantize_grids_beats=[],
    )
    candidates, _ = engine.search(r, limits)
    assert candidates
    best = candidates[0]
    assert best.solution.deleted_note_ids == ["drop"]
    pin_ids = {p.note_id for p in best.pins}
    assert pin_ids == {"keep"}
    keep = best.pins[0]
    assert keep.pitch == 64 and keep.beat == 6.0  # untouched
    for c in candidates:
        assert "keep" not in c.solution.deleted_note_ids


def test_search_infeasible_when_only_locked_notes_conflict():
    r = req(
        [
            note("l1", 6.0, 64, locked=True),
            note("l2", 6.5, 64, locked=True),
        ]
    )
    limits = SearchLimits(
        max_transpose_semitones=2,
        tempo_float_percent=2.0,
        tempo_step_percent=1.0,
        quantize_grids_beats=[],
    )
    candidates, _ = engine.search(r, limits)
    assert candidates == []


def test_search_ranking_prefers_fewer_deletions_then_less_deviation():
    # one missing pitch among good notes: deleting it (1 deletion, deviation 0)
    # must outrank transposing everything into a fitting key with 0 deletions
    # only if... here deletion wins because deletions sort first
    r = req([note("bad", 2.0, 66), note("g1", 4.0, 60), note("g2", 6.0, 62)])
    limits = SearchLimits(
        max_transpose_semitones=3,
        tempo_float_percent=0.0,
        tempo_step_percent=1.0,
        quantize_grids_beats=[],
    )
    candidates, _ = engine.search(r, limits)
    assert candidates[0].metrics.deleted_count == 1
    assert candidates[0].solution.deleted_note_ids == ["bad"]
    deletions = [c.metrics.deleted_count for c in candidates]
    assert deletions == sorted(deletions)


def test_determinism_same_input_same_output():
    r = req([note("a", 2.0, 60), note("b", 4.0, 61), note("c", 6.0, 64)])
    sol = SolutionSpec(transpose_semitones=1, tempo_factor=0.98, quantize_grid_beats=0.25)
    pins1 = engine.build_pins(r, engine.place(r, sol)[0])
    pins2 = engine.build_pins(r, engine.place(r, sol)[0])
    assert pins1 == pins2
    limits = SearchLimits(
        max_transpose_semitones=2, tempo_float_percent=2.0, tempo_step_percent=2.0
    )
    c1, _ = engine.search(r, limits)
    c2, _ = engine.search(r, limits)
    assert c1 == c2
