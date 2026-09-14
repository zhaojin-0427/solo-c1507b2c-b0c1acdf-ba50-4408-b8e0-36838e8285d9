"""Tests for the assembly-calibration engine, freezing and API.

Test rig A (fit + analysis, hand-checkable):
  diameter 60 mm -> shell radius R = 30 mm; length 80 mm; pin diameter 2 mm
  datum stations z = 10/70 mm, angles 0/120/240 deg
  station z=10: runout = 0.02 + 0.01*cos - 0.03*sin  (ecc 0.0316228 at 288.435 deg)
  station z=70: runout = 0.00 + 0.02*cos + 0.01*sin  (ecc 0.0223607 at 26.565 deg)
  bearings at z = 0/80 mm, heights 0 / 0.08 mm -> axis slope 0.001 mm/mm
  pin "p1": angle 90 deg, axial 10 mm, height 4 mm
    -> runout -0.01, axis height 0.01, tip height 34.0 mm exactly

Test rig B (violations/search): zero runout, level bearings -> every 4 mm
pin tip sits at exactly 34.0 mm; reeds at height 33.6 mm give depth 0.4 mm.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from musicbox import calibration_engine as ce
from musicbox.calibration_models import (
    Adjustment,
    CalibrationSearchLimits,
    CalibrationSpec,
    symmetric_offsets_mm,
)
from musicbox.main import create_app

from conftest import make_arrangement, note

SQRT3_2 = math.sqrt(3.0) / 2.0

# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------


def source(pins: list[tuple[str, int, float, float]]) -> ce.CalSource:
    return ce.CalSource(
        version_id=1,
        content_hash="src",
        diameter_mm=60.0,
        length_mm=80.0,
        pin_diameter_mm=2.0,
        pins=[ce.CalSourcePin(nid, p, a, z) for nid, p, a, z in pins],
    )


def reed(pitch: int, axial: float, height: float = 33.6, width: float = 1.8,
         max_depth: float = 1.0) -> dict:
    return {
        "pitch": pitch,
        "axial_mm": axial,
        "height_mm": height,
        "width_mm": width,
        "max_pluck_depth_mm": max_depth,
    }


def make_spec(reeds: list[dict], **overrides) -> dict:
    spec = {
        "length_unit": "mm",
        "angle_unit": "deg",
        "runout": {
            "datum_points_mm": [10.0, 70.0],
            "angles_deg": [0.0, 120.0, 240.0],
            "values_mm": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        "bearing_a_mm": 0.0,
        "bearing_b_mm": 80.0,
        "bearing_a_height_mm": 0.0,
        "bearing_b_height_mm": 0.0,
        "default_pin_height_mm": 4.0,
        "pin_heights_mm": {},
        "reeds": reeds,
        "min_engagement_mm": 0.2,
    }
    spec.update(overrides)
    return spec


def spec(reeds: list[dict], **overrides) -> CalibrationSpec:
    return CalibrationSpec.model_validate(make_spec(reeds, **overrides))


def spec_a() -> CalibrationSpec:
    """Rig A: sinusoid runout at two stations, tilted axis."""
    return spec(
        [reed(60, 10.0), reed(61, 12.0), reed(64, 30.0), reed(65, 40.0)],
        runout={
            "datum_points_mm": [10.0, 70.0],
            "angles_deg": [0.0, 120.0, 240.0],
            "values_mm": [
                [0.03, 0.015 - 0.03 * SQRT3_2, 0.015 + 0.03 * SQRT3_2],
                [0.02, -0.01 + 0.01 * SQRT3_2, -0.01 - 0.01 * SQRT3_2],
            ],
        },
        bearing_b_height_mm=0.08,
    )


def limits(**overrides) -> CalibrationSearchLimits:
    base = {
        "shim_thicknesses_mm": [0.1, 0.3],
        "max_shims_per_end": 2,
        "comb_shift_range_mm": 0.0,
        "comb_shift_step_mm": 0.1,
        "comb_height_range_mm": 0.0,
        "comb_height_step_mm": 0.05,
    }
    base.update(overrides)
    return CalibrationSearchLimits(**base)


ZERO = Adjustment()


# ---------------------------------------------------------------------------
# axis fitting
# ---------------------------------------------------------------------------


def test_fit_recovers_station_sinusoids():
    model = ce.fit_axis_model(spec_a())
    (a1, b1, c1), (a2, b2, c2) = model.stations
    assert (a1, b1, c1) == pytest.approx((0.02, 0.01, -0.03), abs=1e-12)
    assert (a2, b2, c2) == pytest.approx((0.0, 0.02, 0.01), abs=1e-12)
    assert model.max_residual_mm == pytest.approx(0.0, abs=1e-12)


def test_axis_lines_interpolate_between_stations():
    model = ce.fit_axis_model(spec_a())
    # a: 0.02 -> 0.0 over z 10..70
    assert model.a1 == pytest.approx(-0.02 / 60.0, abs=1e-12)
    assert model.a0 == pytest.approx(0.02 + 10 * 0.02 / 60.0, abs=1e-12)
    # bx: 0.01 -> 0.02; by: -0.03 -> 0.01
    assert model.bx1 == pytest.approx(0.01 / 60.0, abs=1e-12)
    assert model.bx0 == pytest.approx(0.01 - 10 * 0.01 / 60.0, abs=1e-12)
    assert model.by1 == pytest.approx(0.04 / 60.0, abs=1e-12)
    assert model.by0 == pytest.approx(-0.03 - 10 * 0.04 / 60.0, abs=1e-12)


def test_axis_fit_output_eccentricity_at_bearings():
    fit = ce.axis_fit_out(ce.fit_axis_model(spec_a()), spec_a())
    # at z=0: (bx, by) = (0.00833333, -0.03666667)
    assert fit.ecc_at_bearing_a_mm == pytest.approx(0.037602, abs=1e-6)
    assert fit.ecc_at_bearing_a_angle_deg == pytest.approx(282.804266, abs=1e-6)
    # at z=80: (0.02166667, 0.01666667)
    assert fit.ecc_at_bearing_b_mm == pytest.approx(0.027335, abs=1e-6)
    assert fit.ecc_at_bearing_b_angle_deg == pytest.approx(37.568593, abs=1e-6)
    d1, d2 = fit.datums
    assert d1.axial_mm == 10.0
    assert d1.mean_deviation_mm == 0.02
    assert d1.eccentricity_mm == pytest.approx(math.hypot(0.01, 0.03), abs=1e-6)
    assert d1.eccentricity_angle_deg == pytest.approx(288.434949, abs=1e-6)
    assert d2.eccentricity_mm == pytest.approx(math.hypot(0.02, 0.01), abs=1e-6)
    assert d2.eccentricity_angle_deg == pytest.approx(26.565051, abs=1e-6)
    assert fit.max_residual_mm == 0.0


def test_fit_with_three_stations_least_squares_residual():
    # mean deviations 0.02 / 0.05 / 0.0 are not collinear -> residual > 0
    sp = spec(
        [reed(60, 10.0)],
        runout={
            "datum_points_mm": [10.0, 40.0, 70.0],
            "angles_deg": [0.0, 120.0, 240.0],
            "values_mm": [
                [0.02, 0.02, 0.02],
                [0.05, 0.05, 0.05],
                [0.0, 0.0, 0.0],
            ],
        },
    )
    model = ce.fit_axis_model(sp)
    # least-squares line through (10, .02), (40, .05), (70, 0): slope -1/3000
    assert model.a1 == pytest.approx(-0.0003333333, abs=1e-9)
    assert model.a0 == pytest.approx(0.0366666667, abs=1e-9)
    residual = max(
        abs(v - model.runout(z, th))
        for z, row in zip((10.0, 40.0, 70.0), sp.runout.values_mm)
        for th, v in zip(sp.runout.angles_deg, row)
    )
    assert model.max_residual_mm == pytest.approx(residual, abs=1e-12)
    assert residual > 0.0
    # deterministic: refitting yields identical coefficients
    again = ce.fit_axis_model(sp)
    assert (again.a0, again.a1) == (model.a0, model.a1)


# ---------------------------------------------------------------------------
# per-pin analysis
# ---------------------------------------------------------------------------


def test_pin_tip_height_depth_and_margins_hand_computed():
    sp = spec_a()
    src = source([("p1", 60, 90.0, 10.0)])
    ana = ce.analyze(sp, src, ce.fit_axis_model(sp), ZERO)
    (p,) = ana.pins
    assert p.runout_mm == pytest.approx(-0.01, abs=1e-6)
    assert p.tip_height_mm == pytest.approx(34.0, abs=1e-6)
    assert p.pluck_depth_mm == pytest.approx(0.4, abs=1e-6)
    assert p.axial_deviation_mm == 0.0
    assert p.contacted_pitches == [60]
    assert p.depth_margin_mm == pytest.approx(0.2, abs=1e-6)
    # reed 61 at 12 mm: edge gap 11.1 - 11 = 0.1 binds the axial margin
    assert p.axial_margin_mm == pytest.approx(0.1, abs=1e-6)
    assert p.margin_mm == pytest.approx(0.1, abs=1e-6)
    assert p.violations == []
    assert ana.summary.ok is True
    assert ana.summary.min_margin_mm == pytest.approx(0.1, abs=1e-6)


def test_second_pin_uses_local_runout_and_axis_height():
    sp = spec_a()
    src = source([("p2", 64, 200.0, 30.0)])
    ana = ce.analyze(sp, src, ce.fit_axis_model(sp), ZERO)
    (p,) = ana.pins
    # a(30)=0.0133333, bx(30)=0.0133333, by(30)=-0.0166667; cos200/-sin200 below
    run = (
        0.0133333333
        + 0.0133333333 * math.cos(math.radians(200.0))
        - 0.0166666667 * math.sin(math.radians(200.0))
    )
    assert p.runout_mm == pytest.approx(run, abs=1e-6)
    assert p.tip_height_mm == pytest.approx(0.03 + 30.0 + run + 4.0, abs=1e-6)
    assert p.pluck_depth_mm == pytest.approx(0.43 + run, abs=1e-6)


def test_over_depth_flagged_with_measured_and_required():
    sp = spec([reed(60, 10.0, height=32.9)])  # depth 1.1 > allowed 1.0
    src = source([("p1", 60, 90.0, 10.0)])
    ana = ce.analyze(sp, src, ce.fit_axis_model(sp), ZERO)
    (p,) = ana.pins
    assert p.violations == ["over_depth"]
    issue = ana.diagnostics.first_by_kind["over_depth"]
    assert issue is not None and issue.pin_index == 0
    assert issue.measured == pytest.approx(1.1, abs=1e-6)
    assert issue.required == 1.0
    assert p.depth_margin_mm == pytest.approx(-0.1, abs=1e-6)


def test_missed_when_depth_below_minimum_engagement():
    sp = spec([reed(60, 10.0, height=33.85)])  # depth 0.15 < 0.2
    src = source([("p1", 60, 90.0, 10.0)])
    ana = ce.analyze(sp, src, ce.fit_axis_model(sp), ZERO)
    (p,) = ana.pins
    assert p.violations == ["missed"]
    issue = ana.diagnostics.first_by_kind["missed"]
    assert issue.measured == pytest.approx(0.15, abs=1e-6)
    assert issue.required == 0.2


def test_missed_axially_without_any_contact():
    sp = spec([reed(60, 13.5), reed(61, 16.0)])
    src = source([("p1", 60, 90.0, 10.0)])
    ana = ce.analyze(sp, src, ce.fit_axis_model(sp), ZERO)
    (p,) = ana.pins
    assert p.contacted_pitches == []
    assert p.violations == ["missed"]
    assert p.axial_margin_mm < 0.0


def test_wrong_reed_when_only_a_neighbour_is_contacted():
    sp = spec([reed(60, 13.5), reed(61, 11.5)])
    src = source([("p1", 60, 90.0, 10.0)])
    ana = ce.analyze(sp, src, ce.fit_axis_model(sp), ZERO)
    (p,) = ana.pins
    assert p.contacted_pitches == [61]
    assert p.violations == ["missed", "wrong_reed"]
    issue = ana.diagnostics.first_by_kind["wrong_reed"]
    assert issue is not None and "61" in issue.message


def test_double_contact_when_straddling_two_reeds():
    sp = spec([reed(60, 10.0, width=0.6), reed(61, 10.7, width=0.6)])
    src = source([("p1", 60, 90.0, 10.0)])
    ana = ce.analyze(sp, src, ce.fit_axis_model(sp), ZERO)
    (p,) = ana.pins
    assert p.contacted_pitches == [60, 61]
    assert p.violations == ["double_contact"]
    assert p.axial_margin_mm == pytest.approx(-0.6, abs=1e-6)


def test_first_by_kind_locates_earliest_pin():
    sp = spec(
        [reed(60, 10.0), reed(64, 30.0, height=33.85), reed(65, 40.0, height=32.9)]
    )
    src = source(
        [("p1", 60, 30.0, 10.0), ("p2", 64, 90.0, 30.0), ("p3", 65, 150.0, 40.0)]
    )
    ana = ce.analyze(sp, src, ce.fit_axis_model(sp), ZERO)
    first = ana.diagnostics.first_by_kind
    assert first["missed"].pin_index == 1
    assert first["over_depth"].pin_index == 2
    assert first["wrong_reed"] is None
    assert first["double_contact"] is None
    assert ana.diagnostics.issue_counts == {
        "missed": 1,
        "wrong_reed": 0,
        "double_contact": 0,
        "over_depth": 1,
    }
    assert ana.summary.pins_with_violations == 2


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_validate_rejects_unknown_pin_height_id():
    sp = spec([reed(60, 10.0)], pin_heights_mm={"ghost": 4.1})
    with pytest.raises(ce.CalibrationError, match="unknown note ids"):
        ce.validate_spec(sp, source([("p1", 60, 0.0, 10.0)]))


def test_validate_rejects_uncovered_pin_range():
    sp = spec(
        [reed(60, 10.0), reed(64, 40.0)],
        runout={
            "datum_points_mm": [20.0, 70.0],
            "angles_deg": [0.0, 120.0, 240.0],
            "values_mm": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
    )
    src = source([("p1", 60, 0.0, 10.0), ("p2", 64, 0.0, 40.0)])
    with pytest.raises(ce.CalibrationError, match="do not cover"):
        ce.validate_spec(sp, src)


def test_validate_rejects_missing_reed_pitch():
    sp = spec([reed(60, 10.0)])
    with pytest.raises(ce.CalibrationError, match="no measured reed"):
        ce.validate_spec(sp, source([("p1", 64, 0.0, 10.0)]))


def test_validate_rejects_datum_or_reed_outside_length():
    sp = spec(
        [reed(60, 10.0)],
        runout={
            "datum_points_mm": [10.0, 90.0],
            "angles_deg": [0.0, 120.0, 240.0],
            "values_mm": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
    )
    with pytest.raises(ce.CalibrationError, match="exceeds cylinder length"):
        ce.validate_spec(sp, source([("p1", 60, 0.0, 10.0)]))
    sp2 = spec([reed(60, 90.0)])
    with pytest.raises(ce.CalibrationError, match="exceeds cylinder length"):
        ce.validate_spec(sp2, source([("p1", 60, 0.0, 90.0)]))


def test_spec_model_rejects_bad_units_grid_and_reeds():
    with pytest.raises(Exception, match="length_unit"):
        spec([reed(60, 10.0)], length_unit="cm")
    with pytest.raises(Exception, match="one reading per angle"):
        spec(
            [reed(60, 10.0)],
            runout={
                "datum_points_mm": [10.0, 70.0],
                "angles_deg": [0.0, 120.0, 240.0],
                "values_mm": [[0.0, 0.0], [0.0, 0.0, 0.0]],
            },
        )
    with pytest.raises(Exception, match="strictly increasing"):
        spec(
            [reed(60, 10.0)],
            runout={
                "datum_points_mm": [70.0, 10.0],
                "angles_deg": [0.0, 120.0, 240.0],
                "values_mm": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            },
        )
    with pytest.raises(Exception, match="at least 3"):
        spec(
            [reed(60, 10.0)],
            runout={
                "datum_points_mm": [10.0, 70.0],
                "angles_deg": [0.0, 180.0],
                "values_mm": [[0.0, 0.0], [0.0, 0.0]],
            },
        )
    with pytest.raises(Exception, match="overlap axially"):
        spec([reed(60, 10.0), reed(61, 10.5)])
    with pytest.raises(Exception, match="smaller than the allowed pluck depth"):
        spec([reed(60, 10.0, max_depth=0.1)])
    with pytest.raises(Exception, match="unique"):
        spec([reed(60, 10.0), reed(60, 20.0)])


# ---------------------------------------------------------------------------
# shims and adjustment resolution
# ---------------------------------------------------------------------------


def test_shim_decomposition_prefers_fewest_varieties():
    # 0.3 = 0.1+0.1+0.1 (1 variety) beats 0.1+0.2 (2 varieties)
    assert ce.decompose_shim(0.3, [0.1, 0.2], 3) == (0.1, 0.1, 0.1)
    assert ce.decompose_shim(0.3, [0.1, 0.2], 2) == (0.1, 0.2)
    assert ce.decompose_shim(0.35, [0.1, 0.2], 2) is None
    assert ce.decompose_shim(0.0, [0.1, 0.2], 2) == ()


def test_resolve_adjustment_enforces_locks_ranges_and_shims():
    adj = Adjustment(shim_a_mm=0.1)
    with pytest.raises(ce.CalibrationError, match="bearing A is locked"):
        ce.resolve_adjustment(limits(lock_bearing_a=True), adj)
    with pytest.raises(ce.CalibrationError, match="comb shift is locked"):
        ce.resolve_adjustment(limits(lock_comb_shift=True), Adjustment(comb_shift_mm=0.1))
    with pytest.raises(ce.CalibrationError, match="not achievable"):
        ce.resolve_adjustment(limits(), Adjustment(shim_b_mm=0.35))
    with pytest.raises(ce.CalibrationError, match="exceeds the allowed range"):
        ce.resolve_adjustment(
            limits(comb_shift_range_mm=0.2), Adjustment(comb_shift_mm=0.5)
        )
    resolved = ce.resolve_adjustment(limits(), Adjustment(shim_b_mm=0.3))
    assert resolved.shim_b_pieces_mm == [0.3]
    assert resolved.shim_varieties == 1
    assert resolved.adjustment_total_mm == pytest.approx(0.3, abs=1e-6)


def test_search_space_validator_caps_combinations():
    with pytest.raises(Exception, match="search space too large"):
        limits(
            shim_thicknesses_mm=[0.05, 0.1, 0.2],
            max_shims_per_end=4,
            comb_shift_range_mm=20.0,
            comb_shift_step_mm=0.01,
            comb_height_range_mm=20.0,
            comb_height_step_mm=0.01,
        )


def test_symmetric_offsets_include_zero_and_endpoints():
    assert symmetric_offsets_mm(0.6, 0.1) == pytest.approx(
        [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    assert symmetric_offsets_mm(0.0, 0.1) == [0.0]


# ---------------------------------------------------------------------------
# adjustment search
# ---------------------------------------------------------------------------


def shift_rig() -> tuple[CalibrationSpec, ce.CalSource]:
    """Reeds sit 2.0 mm above the pins; depth margin 0.5 exceeds the axial
    margins so the ranking is decided by the axial geometry."""
    sp = spec(
        [reed(60, 12.0, height=33.4, max_depth=1.1)],
        min_engagement_mm=0.1,
    )
    return sp, source([("p1", 60, 30.0, 10.0)])


def test_search_ranks_margin_before_adjustment():
    sp, src = shift_rig()
    lim = limits(
        shim_thicknesses_mm=[0.1], max_shims_per_end=0,
        comb_shift_range_mm=2.0, comb_shift_step_mm=0.5,
    )
    candidates, evaluated = ce.search(sp, src, lim)
    assert evaluated == 9  # 1x1 shim options x 9 shifts x 1 height
    zero = [c for c in candidates if c.violation_count == 0]
    # shifts -1.0/-1.5/-2.0 all center well (margin 0.5); -0.5 grazes (0.4)
    assert zero[0].adjustment.comb_shift_mm == -1.0
    assert zero[0].min_margin_mm == pytest.approx(0.5, abs=1e-6)
    assert zero[0].adjustment.adjustment_total_mm == pytest.approx(1.0, abs=1e-6)
    shifts = [c.adjustment.comb_shift_mm for c in zero]
    assert shifts.index(-0.5) > shifts.index(-1.5)  # worse margin ranks later


def test_search_fixes_bearing_tilt_with_shims():
    sp = spec(
        [reed(60, 10.0), reed(64, 70.0)],
        bearing_b_height_mm=-0.3,
    )
    src = source([("p1", 60, 30.0, 10.0), ("p2", 64, 90.0, 70.0)])
    lim = limits()  # shims [0.1, 0.3] x2 per end, no comb moves
    candidates, evaluated = ce.search(sp, src, lim)
    assert evaluated == 36  # 6 shim totals per end
    best = candidates[0]
    assert best.violation_count == 0
    assert best.adjustment.shim_a_mm == pytest.approx(0.1, abs=1e-6)
    assert best.adjustment.shim_b_mm == pytest.approx(0.6, abs=1e-6)
    assert best.adjustment.shim_b_pieces_mm == [0.3, 0.3]
    assert best.adjustment.shim_varieties == 2
    assert best.min_margin_mm == pytest.approx(0.325, abs=1e-6)


def test_search_respects_locks():
    sp = spec(
        [reed(60, 10.0), reed(64, 70.0)],
        bearing_b_height_mm=-0.3,
    )
    src = source([("p1", 60, 30.0, 10.0), ("p2", 64, 90.0, 70.0)])
    lim = limits(lock_bearing_b=True)
    candidates, _ = ce.search(sp, src, lim)
    assert candidates
    assert all(c.adjustment.shim_b_mm == 0.0 for c in candidates)


# ---------------------------------------------------------------------------
# API: batch lifecycle, search, confirm, recompute
# ---------------------------------------------------------------------------


def freeze_source_version(client: TestClient) -> int:
    arr = make_arrangement(
        [
            note("n1", 1.0, 60),
            note("n2", 3.0, 64),
            note("n3", 5.0, 65),
        ]
    )
    r = client.post(
        "/api/versions",
        json={"arrangement": arr, "solution": {}},
    )
    assert r.status_code == 201
    return r.json()["id"]


def api_spec(**overrides) -> dict:
    comb = overrides.pop("reeds", [
        reed(60, 10.0), reed(61, 12.0), reed(62, 20.0), reed(64, 30.0),
        reed(65, 40.0), reed(67, 50.0), reed(69, 60.0), reed(71, 70.0),
    ])
    return make_spec(comb, **overrides)


def create_batch(client: TestClient, version_id: int, **spec_overrides) -> dict:
    r = client.post(
        "/api/calibration/batches",
        json={"source_version_id": version_id, "spec": api_spec(**spec_overrides)},
    )
    assert r.status_code == 201, r.json()
    return r.json()


def test_batch_create_get_list_and_idempotency(client: TestClient):
    vid = freeze_source_version(client)
    batch = create_batch(client, vid)
    assert batch["status"] == "collecting"
    assert batch["confirmed_plan_id"] is None
    assert batch["source_version_id"] == vid
    assert batch["summary"]["pin_count"] == 3
    assert batch["summary"]["violation_count"] == 0
    assert batch["fit"]["max_residual_mm"] == 0.0
    depths = {p["note_id"]: p["pluck_depth_mm"] for p in batch["pins"]}
    assert depths == {"n1": 0.4, "n2": 0.4, "n3": 0.4}

    again = client.post(
        "/api/calibration/batches",
        json={"source_version_id": vid, "spec": api_spec()},
    )
    assert again.status_code == 200
    assert again.json()["id"] == batch["id"]
    assert again.json()["content_hash"] == batch["content_hash"]

    got = client.get(f"/api/calibration/batches/{batch['id']}")
    assert got.status_code == 200
    assert got.json()["content_hash"] == batch["content_hash"]

    listed = client.get("/api/calibration/batches").json()
    assert [(b["id"], b["status"]) for b in listed] == [(batch["id"], "collecting")]


def test_batch_404_and_validation_422(client: TestClient):
    vid = freeze_source_version(client)
    assert client.get("/api/calibration/batches/999").status_code == 404
    r = client.post("/api/calibration/batches", json={"source_version_id": 999, "spec": api_spec()})
    assert r.status_code == 404
    # uncovered pin range: datum starts at 20 but pin n1 sits at 10
    r = client.post(
        "/api/calibration/batches",
        json={
            "source_version_id": vid,
            "spec": api_spec(
                runout={
                    "datum_points_mm": [20.0, 70.0],
                    "angles_deg": [0.0, 120.0, 240.0],
                    "values_mm": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                }
            ),
        },
    )
    assert r.status_code == 422
    assert "do not cover" in r.json()["detail"]
    # missing reed for pitch 65
    incomplete = [r_ for r_ in api_spec()["reeds"] if r_["pitch"] != 65]
    r = client.post(
        "/api/calibration/batches",
        json={"source_version_id": vid, "spec": api_spec(reeds=incomplete)},
    )
    assert r.status_code == 422
    assert "no measured reed" in r.json()["detail"]
    # wrong unit string
    r = client.post(
        "/api/calibration/batches",
        json={"source_version_id": vid, "spec": api_spec(length_unit="cm")},
    )
    assert r.status_code == 422


def test_batch_reports_first_violation_locations(client: TestClient):
    vid = freeze_source_version(client)
    reeds = api_spec()["reeds"]
    for r_ in reeds:
        if r_["pitch"] == 64:
            r_["height_mm"] = 33.85  # depth 0.15 -> missed (too shallow)
        if r_["pitch"] == 65:
            r_["height_mm"] = 32.9  # depth 1.1 -> over_depth
    batch = create_batch(client, vid, reeds=reeds)
    first = batch["diagnostics"]["first_by_kind"]
    assert first["missed"]["note_id"] == "n2"
    assert first["over_depth"]["note_id"] == "n3"
    assert first["wrong_reed"] is None
    assert batch["diagnostics"]["ok"] is False
    by_id = {p["note_id"]: p for p in batch["pins"]}
    assert by_id["n2"]["violations"] == ["missed"]
    assert by_id["n3"]["violations"] == ["over_depth"]


def test_search_endpoint_returns_ranked_candidates(client: TestClient):
    vid = freeze_source_version(client)
    batch = create_batch(client, vid)
    assert batch["summary"]["min_margin_mm"] == pytest.approx(0.1, abs=1e-6)
    r = client.post(
        f"/api/calibration/batches/{batch['id']}/search",
        json={"limits": {"comb_shift_range_mm": 0.2, "comb_height_range_mm": 0.1}},
    )
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["feasible"] is True
    assert body["combinations_evaluated"] == 1600  # 8x8 shim totals x 5 x 5
    best = body["candidates"][0]
    assert best["rank"] == 1
    assert best["violation_count"] == 0
    # shift +0.2 widens the tightest reed gap to 0.3; height -0.1 centres depth
    assert best["adjustment"]["comb_shift_mm"] == pytest.approx(0.2, abs=1e-6)
    assert best["adjustment"]["comb_height_mm"] == pytest.approx(-0.1, abs=1e-6)
    assert best["adjustment"]["adjustment_total_mm"] == pytest.approx(0.3, abs=1e-6)
    assert best["min_margin_mm"] == pytest.approx(0.3, abs=1e-6)
    assert client.post("/api/calibration/batches/999/search", json={}).status_code == 404


def test_confirm_freezes_plan_and_marks_batch_confirmed(client: TestClient):
    vid = freeze_source_version(client)
    batch = create_batch(client, vid)
    adj = {"shim_a_mm": 0.0, "shim_b_mm": 0.1, "comb_shift_mm": 0.0, "comb_height_mm": 0.0}
    r = client.post(
        f"/api/calibration/batches/{batch['id']}/confirm",
        json={"adjustment": adj, "limits": {"shim_thicknesses_mm": [0.1, 0.3]}},
    )
    assert r.status_code == 201, r.json()
    plan = r.json()
    assert plan["adjustment"]["shim_b_mm"] == 0.1
    assert plan["adjustment"]["shim_b_pieces_mm"] == [0.1]
    assert plan["adjustment"]["shim_varieties"] == 1
    assert plan["batch_content_hash"] == batch["content_hash"]
    assert plan["summary"]["violation_count"] == 0

    # batch flips to confirmed
    got = client.get(f"/api/calibration/batches/{batch['id']}").json()
    assert got["status"] == "confirmed"
    assert got["confirmed_plan_id"] == plan["id"]
    listed = client.get("/api/calibration/batches").json()
    assert listed[0]["status"] == "confirmed"

    # idempotent confirm
    again = client.post(
        f"/api/calibration/batches/{batch['id']}/confirm",
        json={"adjustment": adj, "limits": {"shim_thicknesses_mm": [0.1, 0.3]}},
    )
    assert again.status_code == 200
    assert again.json()["id"] == plan["id"]

    # plan endpoints
    got_plan = client.get(f"/api/calibration/plans/{plan['id']}")
    assert got_plan.status_code == 200
    assert got_plan.json()["content_hash"] == plan["content_hash"]
    plans = client.get("/api/calibration/plans").json()
    assert [p["id"] for p in plans] == [plan["id"]]

    rec = client.post(f"/api/calibration/plans/{plan['id']}/recompute")
    assert rec.status_code == 200
    assert rec.json()["match"] is True
    assert rec.json()["stored_hash"] == rec.json()["recomputed_hash"]
    assert client.get("/api/calibration/plans/999").status_code == 404
    assert client.post("/api/calibration/plans/999/recompute").status_code == 404


def test_confirm_rejects_adjustment_outside_envelope(client: TestClient):
    vid = freeze_source_version(client)
    batch = create_batch(client, vid)
    # locked comb may not be shifted
    r = client.post(
        f"/api/calibration/batches/{batch['id']}/confirm",
        json={
            "adjustment": {"comb_shift_mm": 0.1},
            "limits": {"lock_comb_shift": True},
        },
    )
    assert r.status_code == 422
    assert "locked" in r.json()["detail"]
    # shim total not achievable with the stocked sizes
    r = client.post(
        f"/api/calibration/batches/{batch['id']}/confirm",
        json={
            "adjustment": {"shim_b_mm": 0.35},
            "limits": {"shim_thicknesses_mm": [0.1, 0.2], "max_shims_per_end": 2},
        },
    )
    assert r.status_code == 422
    assert "not achievable" in r.json()["detail"]
    assert client.post("/api/calibration/batches/999/confirm", json={}).status_code == 404


def test_plan_hash_stable_across_instances(tmp_path):
    hashes = []
    for name in ("a.db", "b.db"):
        app = create_app(str(tmp_path / name))
        with TestClient(app) as c:
            vid = freeze_source_version(c)
            batch = create_batch(c, vid)
            r = c.post(
                f"/api/calibration/batches/{batch['id']}/confirm",
                json={"adjustment": {"shim_b_mm": 0.1}},
            )
            assert r.status_code == 201
            hashes.append((batch["content_hash"], r.json()["content_hash"]))
    assert hashes[0] == hashes[1]
