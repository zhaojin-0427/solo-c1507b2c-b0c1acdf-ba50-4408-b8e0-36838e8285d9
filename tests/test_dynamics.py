"""Tests for the powertrain dynamics engine, freezing and API.

Test rig (hand-checkable):
  design cylinder speed 10 rpm; last of three pins at 3 s -> 0.5 cylinder rev
  mainspring 4 mNm, one stage ratio 10 / efficiency 0.9
    -> drive at the cylinder = 4/10*0.9 = 0.36 mNm
  inertia 100 g*cm^2 = 1e-5 kg*m^2; kinetic energy at 10 rpm ~= 5.5 uJ
  governor slope 0.036 mNm/rpm -> equilibrium ~= 10 rpm
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from musicbox import dynamics_engine as de
from musicbox.dynamics_models import DynamicsPin, DynamicsSpec, ScenarioParams
from musicbox.main import create_app

from conftest import make_arrangement, note

PITCHES = [60, 62, 64]


def make_spec(**overrides) -> dict:
    spec = {
        "spring_torque": {
            "x": [0.0, 12.0],
            "y": [4.0, 3.0],
            "x_unit": "turns",
            "y_unit": "mN*m",
        },
        "gear_train": [{"ratio": 10.0, "efficiency": 0.9}],
        "base_inertia_g_cm2": 100.0,
        "governor_drag": {
            "x": [0.0, 30.0],
            "y": [0.0, 1.08],
            "x_unit": "rpm",
            "y_unit": "mN*m",
        },
        "pluck_energy_uJ": {"60": 0.5, "62": 0.5, "64": 0.5},
        "dt_s": 0.001,
    }
    spec.update(overrides)
    return spec


def spec(**overrides) -> DynamicsSpec:
    return DynamicsSpec.model_validate(make_spec(**overrides))


def make_source(pins: list[tuple[str, int, float]] | None = None) -> de.DynamicsSource:
    if pins is None:
        # beats 2/4/6 at 120 bpm -> 1/2/3 s; design 10 rpm -> 1/6..1/2 rev
        pins = [("a", 60, 1.0), ("b", 62, 2.0), ("c", 64, 3.0)]
    return de.build_source(
        1,
        "src-hash",
        10.0,
        [
            DynamicsPin(
                note_id=nid,
                pitch=p,
                design_time_s=t,
                angle_deg=(t * 60.0) % 360.0,
                phase_rev=t / 6.0,
            )
            for nid, p, t in pins
        ],
    )


def scenario(**overrides) -> ScenarioParams:
    base = {
        "prewind_turns": 2.0,
        "governor_coefficient": 1.0,
        "flywheel_inertia_g_cm2": 0.0,
    }
    base.update(overrides)
    return ScenarioParams.model_validate(base)


# ---------------------------------------------------------------------------
# engine: validation and derived data
# ---------------------------------------------------------------------------


def test_total_ratio_and_efficiency_products():
    s = spec(gear_train=[{"ratio": 2.0, "efficiency": 0.9},
                          {"ratio": 5.0, "efficiency": 0.8}])
    assert de.total_ratio(s) == pytest.approx(10.0)
    assert de.total_efficiency(s) == pytest.approx(0.72)


def test_validate_rejects_missing_reed_energy():
    s = spec(pluck_energy_uJ={"60": 1.0})
    with pytest.raises(de.DynamicsError, match="missing reeds"):
        de.validate_spec(s, make_source())


def test_validate_rejects_insufficient_spring_travel():
    # tune needs 0.5 cylinder rev / ratio 10 = 0.05 barrel turns; give 0.01
    short = {
        "x": [0.0, 0.01],
        "y": [4.0, 4.0],
        "x_unit": "turns",
        "y_unit": "mN*m",
    }
    with pytest.raises(de.DynamicsError, match="mainspring travel"):
        de.validate_spec(spec(spring_torque=short), make_source())


def test_coincident_phases_group_into_one_event():
    src = make_source(
        [("a", 60, 1.0), ("a2", 62, 1.0), ("b", 64, 2.0)]
    )
    assert len(src.events) == 2
    assert src.events[0].note_ids == ["a", "a2"]
    assert src.events[0].pitches == [60, 62]


# ---------------------------------------------------------------------------
# engine: integration
# ---------------------------------------------------------------------------


def test_clean_run_completes_with_pluck_dips_and_drift():
    s = spec()
    res = de.simulate(s, make_source(), scenario())
    assert res.summary.completed is True
    assert res.summary.pins_plucked == 3
    assert res.summary.stall is False
    assert res.summary.overspeed is False
    assert res.summary.violation_count == 0
    # every pluck takes energy: speed dips, then recovers before the next one
    for e in res.pluck_events:
        assert e.rpm_after < e.rpm_before
        assert e.min_rpm_until_next >= e.rpm_after - 1e-9
    # accumulated running time near the designed 3 s at the ~10 rpm equilibrium
    assert res.summary.total_time_s == pytest.approx(3.0, abs=0.25)
    assert res.summary.remaining_spring_turns < 2.0


def test_stall_is_located_at_first_pin():
    s = spec(pluck_energy_uJ={"60": 50.0, "62": 0.5, "64": 0.5})  # > KE @10rpm
    res = de.simulate(s, make_source(), scenario())
    assert res.summary.stall is True
    assert res.summary.completed is False
    loc = res.first_violations["stall"]
    assert loc is not None
    assert loc.kind == "stall"
    assert loc.pin_index == 0
    assert loc.note_ids == ["a"]
    assert res.summary.pins_plucked == 1


def test_overspeed_is_detected_with_weak_governor():
    # 0.005 mNm/rpm -> equilibrium ~72 rpm, far past the 12.5 rpm threshold
    gov = {"x": [0.0, 100.0], "y": [0.0, 0.5], "x_unit": "rpm", "y_unit": "mN*m"}
    res = de.simulate(spec(governor_drag=gov), make_source(), scenario())
    assert res.summary.overspeed is True
    loc = res.first_violations["overspeed"]
    assert loc is not None
    # detection happens on the first step boundary at/above 12.5 rpm
    assert loc.rpm >= 12.5
    assert loc.rpm < 14.0


def test_beat_drift_is_located_by_pin_when_running_slow():
    # 0.05 mNm/rpm slope -> equilibrium ~7.2 rpm: intervals ~45% too long
    gov = {"x": [0.0, 30.0], "y": [0.0, 1.5], "x_unit": "rpm", "y_unit": "mN*m"}
    small_plucks = {"60": 0.05, "62": 0.05, "64": 0.05}
    res = de.simulate(
        spec(governor_drag=gov, pluck_energy_uJ=small_plucks),
        make_source(),
        scenario(),
    )
    assert res.summary.stall is False
    assert res.summary.drift_violations >= 1
    loc = res.first_violations["beat_drift"]
    assert loc is not None and loc.pin_index == 0
    assert loc.note_ids == ["a"]
    assert res.summary.max_beat_drift_ratio == pytest.approx(0.45, abs=0.05)


def test_step_size_changes_result_but_each_is_deterministic():
    s1 = spec(dt_s=0.02)
    s2 = spec(dt_s=0.0005)
    a = de.simulate(s1, make_source(), scenario())
    a2 = de.simulate(s1, make_source(), scenario())
    b = de.simulate(s2, make_source(), scenario())
    assert a.model_dump() == a2.model_dump()
    assert a.summary.total_time_s != b.summary.total_time_s
    # both coarse and fine grids converge toward the same answer
    assert abs(a.summary.total_time_s - b.summary.total_time_s) < 0.05


def test_prewind_beyond_travel_is_rejected():
    with pytest.raises(de.DynamicsError, match="prewind"):
        de.simulate(spec(), make_source(), scenario(prewind_turns=100.0))


def test_post_pluck_speed_carries_into_next_step_with_zero_drive():
    # zero drive and zero drag: without torque the only speed changes are the
    # pluck impulses; the post-pluck velocity must persist into the next step
    # instead of snapping back to the design speed
    s = spec(
        spring_torque={"x": [0.0, 12.0], "y": [0.0, 0.0],
                        "x_unit": "turns", "y_unit": "mN*m"},
        governor_drag={"x": [0.0, 100.0], "y": [0.0, 0.0],
                       "x_unit": "rpm", "y_unit": "mN*m"},
        pluck_energy_uJ={"60": 50.0, "62": 50.0, "64": 50.0},
    )
    pins = [
        DynamicsPin(note_id=nid, pitch=p, design_time_s=t,
                    angle_deg=(t * 360.0) % 360.0, phase_rev=t / 60.0)
        for nid, p, t in [("a", 60, 1.0), ("b", 62, 2.0), ("c", 64, 3.0)]
    ]
    src = de.build_source(1, "h", 60.0, pins)
    res = de.simulate(s, src, scenario(prewind_turns=10.0))

    e0, e1, e2 = res.pluck_events
    assert e0.rpm_before == pytest.approx(60.0, abs=1e-4)
    assert e0.rpm_after == pytest.approx(51.84698, abs=1e-4)
    # the next pin must fire from the slowed speed, not from a restored 60 rpm
    assert e1.rpm_before == pytest.approx(e0.rpm_after, abs=1e-4)
    assert e1.rpm_after == pytest.approx(42.14521, abs=1e-4)
    assert e2.rpm_before == pytest.approx(e1.rpm_after, abs=1e-4)
    # the speed curve between the plucks stays flat at the slowed value
    between = [
        r for tt, r in zip(res.curves.t_s, res.curves.rpm)
        if e0.time_s + res.dt_s <= tt < e1.time_s
    ]
    assert between and all(abs(r - 51.84698) < 1e-4 for r in between)
    # with no drive the third pluck drops below the stall threshold
    assert res.summary.stall is True
    assert res.first_violations["stall"].pin_index == 2


def test_gear_ratio_override_reflects_drive_and_travel():
    res = de.simulate(spec(), make_source(), scenario(gear_ratio=12.0))
    assert res.scenario.gear_ratio == 12.0
    # more ratio -> less barrel travel consumed
    assert res.summary.used_spring_turns == pytest.approx(0.5 / 12.0, abs=1e-3)


# ---------------------------------------------------------------------------
# engine: scenario search ranking
# ---------------------------------------------------------------------------


def test_search_ranks_violations_then_drift_then_margin_then_travel():
    s = spec()
    # 0.05 slope -> drift violations; 0.036 slope -> clean
    candidates, evaluated = de.search_scenarios(
        s,
        make_source(),
        ratios=[10.0],
        prewinds=[1.0, 2.0],
        coefficients=[0.72, 1.0, 1.39],  # 0.026 / 0.036 / 0.05 mNm/rpm slopes
        inertias=[0.0],
        max_candidates=6,
    )
    assert evaluated == 6
    counts = [c["summary"].violation_count for c in candidates]
    assert counts == sorted(counts)
    assert counts[0] == 0
    # within the clean tier, smaller max drift comes first
    clean = [c for c in candidates if c["summary"].violation_count == 0]
    drifts = [c["summary"].max_beat_drift_ratio for c in clean]
    assert drifts == sorted(drifts)


def test_search_skips_prewind_beyond_travel():
    candidates, evaluated = de.search_scenarios(
        spec(), make_source(), [10.0], [100.0], [1.0], [0.0], 5
    )
    assert candidates == []
    assert evaluated == 0


def test_search_space_is_capped():
    with pytest.raises(de.DynamicsError, match="too large"):
        de.search_scenarios(
            spec(),
            make_source(),
            [1.0] * 20,
            [1.0] * 20,
            [1.0] * 20,
            [1.0] * 20,
            5,
        )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def freeze_version(client: TestClient) -> int:
    arr = make_arrangement(
        [note("a", 2.0, 60), note("b", 4.0, 62), note("c", 6.0, 64)]
    )
    r = client.post("/api/versions", json={"arrangement": arr, "solution": {}})
    assert r.status_code == 201
    return r.json()["id"]


def create_trial(client: TestClient, vid: int, **spec_overrides):
    r = client.post(
        "/api/dynamics/trials",
        json={"source_version_id": vid, "spec": make_spec(**spec_overrides)},
    )
    return r


def test_create_trial_stores_curves_gear_inertia_and_pluck_energies(client: TestClient):
    vid = freeze_version(client)
    r = create_trial(client, vid)
    assert r.status_code == 201
    body = r.json()
    assert body["source_version_id"] == vid
    assert body["source_content_hash"]
    d = body["derived"]
    assert d["design_rpm"] == 10.0
    assert d["total_ratio"] == 10.0
    assert d["total_efficiency"] == 0.9
    assert d["inertia_kg_m2"] == 1e-5
    assert d["required_barrel_turns"] == 0.05
    assert d["available_spring_turns"] == 12.0
    assert d["pluck_event_count"] == 3
    assert body["spec"]["spring_torque"]["x_unit"] == "turns"
    assert len(body["pins"]) == 3


def test_trial_is_idempotent_and_listed(client: TestClient):
    vid = freeze_version(client)
    r1 = create_trial(client, vid)
    r2 = create_trial(client, vid)
    assert r1.status_code == 201 and r2.status_code == 200
    assert r1.json()["id"] == r2.json()["id"]
    assert r1.json()["content_hash"] == r2.json()["content_hash"]
    listed = client.get("/api/dynamics/trials").json()
    assert [t["id"] for t in listed] == [r1.json()["id"]]
    got = client.get(f"/api/dynamics/trials/{r1.json()['id']}").json()
    assert got["content_hash"] == r1.json()["content_hash"]


def test_trial_rejects_missing_source_bad_curves_and_units(client: TestClient):
    assert create_trial(client, 999).status_code == 404

    vid = freeze_version(client)
    bad = make_spec()
    bad["governor_drag"]["x"] = [0.0, 1.0, 1.0]  # non-increasing
    assert create_trial(client, vid, governor_drag=bad["governor_drag"]).status_code == 422

    bad = make_spec()
    bad["spring_torque"]["x_unit"] = "rpm"  # wrong unit
    assert create_trial(client, vid, spring_torque=bad["spring_torque"]).status_code == 422

    bad = make_spec()
    bad["spring_torque"]["y"][0] = -1.0  # negative torque
    assert create_trial(client, vid, spring_torque=bad["spring_torque"]).status_code == 422

    bad = make_spec()
    bad["spring_torque"]["x"].append(5.0)  # unequal x/y length
    assert create_trial(client, vid, spring_torque=bad["spring_torque"]).status_code == 422

    bad = make_spec(pluck_energy_uJ={"60": 1.0})  # missing reed
    assert client.post(
        "/api/dynamics/trials", json={"source_version_id": vid, "spec": bad}
    ).status_code == 422

    short = {"x": [0.0, 0.01], "y": [4.0, 4.0], "x_unit": "turns", "y_unit": "mN*m"}
    assert create_trial(client, vid, spring_torque=short).status_code == 422


def test_simulate_endpoint_returns_curves_events_and_locations(client: TestClient):
    tid = create_trial(client, freeze_version(client)).json()["id"]
    r = client.post(
        f"/api/dynamics/trials/{tid}/simulate",
        json={"prewind_turns": 2.0, "governor_coefficient": 1.0,
              "flywheel_inertia_g_cm2": 0.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["trial_id"] == tid
    assert body["summary"]["completed"] is True
    assert len(body["curves"]["t_s"]) == len(body["curves"]["rpm"])
    assert len(body["curves"]["t_s"]) == len(body["curves"]["torque_margin_mNm"])
    assert [e["note_ids"] for e in body["pluck_events"]] == [["a"], ["b"], ["c"]]
    assert body["first_violations"] == {
        "stall": None,
        "overspeed": None,
        "beat_drift": None,
    }


def test_simulate_rejects_unknown_trial_and_excessive_prewind(client: TestClient):
    assert client.post(
        "/api/dynamics/trials/999/simulate", json={"prewind_turns": 1.0}
    ).status_code == 404
    tid = create_trial(client, freeze_version(client)).json()["id"]
    r = client.post(
        f"/api/dynamics/trials/{tid}/simulate", json={"prewind_turns": 99.0}
    )
    assert r.status_code == 422


def test_search_endpoint_with_locked_ratio(client: TestClient):
    tid = create_trial(client, freeze_version(client)).json()["id"]
    r = client.post(
        f"/api/dynamics/trials/{tid}/search",
        json={
            "gear_ratio_lock": 12.0,
            "prewind_turns": [1.0, 2.0],
            "governor_coefficients": [0.8, 0.9],
            "flywheel_inertia_g_cm2": [0.0, 200.0],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["combinations_evaluated"] == 8
    assert all(c["scenario"]["gear_ratio"] == 12.0 for c in body["candidates"])
    assert body["feasible"] is True
    counts = [c["metrics"]["violation_count"] for c in body["candidates"]]
    assert counts == sorted(counts)


def test_search_all_prewinds_invalid_is_422(client: TestClient):
    tid = create_trial(client, freeze_version(client)).json()["id"]
    r = client.post(
        f"/api/dynamics/trials/{tid}/search",
        json={"prewind_turns": [99.0]},
    )
    assert r.status_code == 422


def test_search_rejects_non_positive_ratio_candidate(client: TestClient):
    tid = create_trial(client, freeze_version(client)).json()["id"]
    r = client.post(
        f"/api/dynamics/trials/{tid}/search",
        json={"gear_ratio_candidates": [0.0], "prewind_turns": [2.0]},
    )
    assert r.status_code == 422


def test_search_rejects_candidates_outside_single_scenario_bounds(client: TestClient):
    # out-of-range grid values must be 422 at the request, never 500 from the
    # internal ScenarioParams construction
    tid = create_trial(client, freeze_version(client)).json()["id"]
    for grid, value in (
        ("governor_coefficients", 101.0),
        ("flywheel_inertia_g_cm2", 2.0e9),
        ("prewind_turns", 200000.0),
        ("gear_ratio_candidates", 20000.0),
    ):
        body = {"prewind_turns": [2.0], grid: [value]}
        r = client.post(f"/api/dynamics/trials/{tid}/search", json=body)
        assert r.status_code == 422, (grid, r.status_code)


# ---------------------------------------------------------------------------
# frozen plans
# ---------------------------------------------------------------------------


def _freeze_plan(client: TestClient, tid: int, **scenario_overrides):
    body = {"prewind_turns": 2.0, "governor_coefficient": 1.0,
            "flywheel_inertia_g_cm2": 0.0}
    body.update(scenario_overrides)
    return client.post(
        "/api/dynamics/plans", json={"trial_id": tid, "scenario": body}
    )


def test_freeze_plan_snapshot_curves_dt_and_recomputes(client: TestClient):
    tid = create_trial(client, freeze_version(client)).json()["id"]
    r = _freeze_plan(client, tid)
    assert r.status_code == 201
    plan = r.json()
    assert plan["dt_s"] == 0.001
    assert plan["scenario"]["prewind_turns"] == 2.0
    assert plan["result"]["summary"]["completed"] is True
    assert plan["result"]["curves"]["t_s"]

    rec = client.post(f"/api/dynamics/plans/{plan['id']}/recompute").json()
    assert rec["match"] is True
    assert rec["stored_hash"] == rec["recomputed_hash"] == plan["content_hash"]


def test_plan_idempotent_listed_and_unknown_404(client: TestClient):
    tid = create_trial(client, freeze_version(client)).json()["id"]
    r1 = _freeze_plan(client, tid)
    r2 = _freeze_plan(client, tid)
    assert r1.status_code == 201 and r2.status_code == 200
    assert r1.json()["id"] == r2.json()["id"]
    listed = client.get("/api/dynamics/plans").json()
    assert [p["id"] for p in listed] == [r1.json()["id"]]
    got = client.get(f"/api/dynamics/plans/{r1.json()['id']}")
    assert got.status_code == 200
    assert client.get("/api/dynamics/plans/999").status_code == 404
    assert client.post("/api/dynamics/plans/999/recompute").status_code == 404


def test_plan_hash_stable_across_instances(tmp_path):
    def build(db: str):
        app = create_app(db)
        with TestClient(app) as c:
            vid = freeze_version(c)
            tid = create_trial(c, vid).json()["id"]
            plan = _freeze_plan(c, tid).json()
        return tid, plan

    tid1, p1 = build(str(tmp_path / "a.db"))
    tid2, p2 = build(str(tmp_path / "b.db"))
    assert tid1 == tid2
    assert p1["content_hash"] == p2["content_hash"]
    assert p1["result"] == p2["result"]


def test_plan_recompute_is_self_contained_from_snapshot(client: TestClient):
    from musicbox import dynamics_freeze

    tid = create_trial(client, freeze_version(client)).json()["id"]
    plan = _freeze_plan(client, tid).json()
    # recompute straight from the stored request snapshot (source pins embedded)
    import json
    from musicbox.store import Store

    store: Store = client.app.state.store
    row = store.get_dynamics_plan(plan["id"])
    request_dict = json.loads(row["request_json"])
    h, result = dynamics_freeze.recompute_plan(request_dict)
    assert h == plan["content_hash"]
    assert result["summary"]["completed"] is True
