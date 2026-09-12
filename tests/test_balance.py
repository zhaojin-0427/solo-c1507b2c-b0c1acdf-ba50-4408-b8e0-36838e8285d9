"""Tests for the dynamic-balancing engine, freezing and API.

Geometry of the test rig (chosen for hand-checkable numbers):
  diameter 60 mm -> shell radius R = 30 mm, circumference C = 60*pi
  length 80 mm   -> mid-plane at 40 mm
  pin height 4 mm -> pin centre-of-mass radius 32 mm
  pin "a": angle 60 deg, axial 10 mm, mass 0.045 g
    -> static vector U = 0.045*32 = 1.44 g*mm at 60 deg
    -> couple about mid-plane = 1.44 * (10-40) = -43.2 g*mm^2
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from musicbox import balance_engine as be
from musicbox.balance_models import (
    BalanceSearchLimits,
    BalanceSpec,
    WeightRef,
    WeightType,
)
from musicbox.main import create_app

from conftest import make_arrangement, note

# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------


def make_spec(**overrides) -> dict:
    spec = {
        "shell_mass_g": 200.0,
        "pin_height_mm": 4.0,
        "default_pin_mass_g": 0.045,
        "pin_masses": {},
        "working_rpm": 10.0,
        "bearing_a_mm": 0.0,
        "bearing_b_mm": 80.0,
        "plane_1_mm": 20.0,
        "plane_2_mm": 60.0,
        "residual_limit_gmm": 0.01,
        "inventory": [
            {"id": "A", "mass_g": 0.06, "diameter_mm": 4.0, "quantity": 5},
            {"id": "B", "mass_g": 0.012, "diameter_mm": 2.0, "quantity": 5},
        ],
    }
    spec.update(overrides)
    return spec


def spec(**overrides) -> BalanceSpec:
    return BalanceSpec.model_validate(make_spec(**overrides))


def source(pins: list[be.SourcePin]) -> be.SourceInfo:
    return be.SourceInfo(
        version_id=1,
        content_hash="src",
        diameter_mm=60.0,
        length_mm=80.0,
        pin_diameter_mm=2.0,
        seam_zone_mm=10.0,
        pins=pins,
    )


PIN_A = be.SourcePin("a", 60.0, 10.0, False)  # U = 1.44 g*mm at 60 deg


def limits(**overrides) -> BalanceSearchLimits:
    base = {
        "angle_step_deg": 15.0,
        "pin_clearance_mm": 1.0,
        "seam_clearance_mm": 2.0,
        "max_weights": 4,
        "max_candidates": 5,
    }
    base.update(overrides)
    return BalanceSearchLimits(**base)


OMEGA2 = (2.0 * math.pi * 10.0 / 60.0) ** 2  # omega^2 at 10 rpm


# ---------------------------------------------------------------------------
# engine: imbalance computation
# ---------------------------------------------------------------------------


def test_single_pin_imbalance_hand_computed():
    sp = spec()
    src = source([PIN_A])
    pins = be.pin_points(sp, src)
    (p,) = pins
    assert p.mass_g == 0.045
    assert p.radius_mm == 32.0  # 30 + 4/2

    imb = be.compute_imbalance(sp, src, pins)
    assert imb.static_gmm == 1.44
    assert imb.static_angle_deg == 60.0
    assert imb.couple_gmm2 == 43.2  # 1.44 * (40-10)
    assert imb.couple_angle_deg == 240.0
    assert imb.total_mass_g == 200.045
    assert imb.com_offset_mm == be.r6(1.44 / 200.045)
    # bearings at 0 and 80 mm: reactions 70/80 and 10/80 of U
    assert imb.bearing_a_load_mn == be.r6(1.44 * 0.875 * OMEGA2 * 1e-3)
    assert imb.bearing_b_load_mn == be.r6(1.44 * 0.125 * OMEGA2 * 1e-3)
    assert imb.within_limit is False  # 1.44 > 0.01 limit


def test_per_pin_contributions_sum_to_static_vector():
    sp = spec()
    src = source([PIN_A, be.SourcePin("b", 200.0, 50.0, False)])
    pins = be.pin_points(sp, src)
    contribs = be.pin_contributions(sp, src, pins)
    assert [c.note_id for c in contribs] == ["a", "b"]
    sx = sum(c.static_x_gmm for c in contribs)
    sy = sum(c.static_y_gmm for c in contribs)
    imb = be.compute_imbalance(sp, src, pins)
    assert math.hypot(sx, sy) == pytest.approx(imb.static_gmm, abs=1e-5)
    a = contribs[0]
    assert a.static_x_gmm == 0.72  # 1.44 * cos(60)
    assert a.static_y_gmm == be.r6(1.44 * math.sin(math.radians(60)))
    assert a.couple_x_gmm2 == -21.6  # 0.72 * (10-40)


def test_symmetric_pins_cancel_static_but_keep_couple():
    sp = spec()
    # same mass, opposite angles: static cancels; different axial -> couple
    src = source([PIN_A, be.SourcePin("b", 240.0, 70.0, False)])
    pins = be.pin_points(sp, src)
    imb = be.compute_imbalance(sp, src, pins)
    assert imb.static_gmm == 0.0
    assert imb.com_offset_mm == 0.0
    assert imb.couple_gmm2 == 86.4  # 2 * 1.44 * 30
    # pure couple -> equal and opposite bearing reactions
    assert imb.bearing_a_load_mn == imb.bearing_b_load_mn
    assert imb.bearing_a_load_mn == be.r6(1.44 * 0.75 * OMEGA2 * 1e-3)


def test_static_imbalance_is_independent_of_working_rpm():
    src = source([PIN_A])
    slow = be.compute_imbalance(spec(working_rpm=5.0), src, be.pin_points(spec(), src))
    fast = be.compute_imbalance(spec(working_rpm=50.0), src, be.pin_points(spec(), src))
    assert slow.static_gmm == fast.static_gmm
    # load scales with omega^2 = (rpm)^2
    ratio = fast.bearing_a_load_mn / slow.bearing_a_load_mn
    assert ratio == pytest.approx(100.0, rel=1e-4)


def test_ideal_correction_matches_hand_solution():
    sp = spec()
    src = source([PIN_A])
    ideal = be.ideal_correction(sp, src, be.pin_points(sp, src))
    # W1 = -5U/4 -> 1.8 g*mm at 240 deg; W2 = U/4 -> 0.36 g*mm at 60 deg
    assert ideal.plane_1.mass_g == 0.06  # 1.8 / 30
    assert ideal.plane_1.angle_deg == 240.0
    assert ideal.plane_2.mass_g == 0.012  # 0.36 / 30
    assert ideal.plane_2.angle_deg == 60.0


def test_per_pin_mass_override():
    sp = spec(pin_masses={"a": 0.09})
    src = source([PIN_A])
    (p,) = be.pin_points(sp, src)
    assert p.mass_g == 0.09
    imb = be.compute_imbalance(sp, src, [p])
    assert imb.static_gmm == 2.88  # 0.09 * 32


# ---------------------------------------------------------------------------
# engine: validation
# ---------------------------------------------------------------------------


def test_validate_spec_rejects_unknown_pin_ids():
    with pytest.raises(be.BalancePlanError, match="unknown note ids"):
        be.validate_spec(spec(pin_masses={"ghost": 0.1}), source([PIN_A]))


def test_validate_spec_rejects_plane_beyond_length():
    with pytest.raises(be.BalancePlanError, match="exceeds cylinder length"):
        be.validate_spec(spec(plane_2_mm=80.5), source([PIN_A]))
    be.validate_spec(spec(plane_2_mm=80.0), source([PIN_A]))  # boundary ok


def test_validate_weights_rejects_unknown_weight_id():
    with pytest.raises(be.BalancePlanError, match="unknown weight id"):
        be.validate_weights(
            spec(), source([PIN_A]), [], [WeightRef(weight_id="ZZ", plane=1, angle_deg=0.0)]
        )


def test_validate_weights_rejects_weight_beyond_length():
    # plane 2 at 79 mm + weight A diameter 4 mm -> extends to 81 > 80
    with pytest.raises(be.BalancePlanError, match="beyond the cylinder length"):
        be.validate_weights(
            spec(plane_2_mm=79.0),
            source([PIN_A]),
            [],
            [WeightRef(weight_id="A", plane=2, angle_deg=90.0)],
        )


def test_validate_weights_rejects_exhausted_inventory():
    w = [WeightRef(weight_id="A", plane=1, angle_deg=10.0)]
    chosen = [WeightRef(weight_id="A", plane=2, angle_deg=200.0)]
    with pytest.raises(be.BalancePlanError, match="in stock"):
        be.validate_weights(spec(inventory=[
            WeightType(id="A", mass_g=0.06, diameter_mm=4.0, quantity=1)
        ]), source([PIN_A]), w, chosen)


def test_validate_weights_rejects_overlap():
    chosen = [
        WeightRef(weight_id="A", plane=1, angle_deg=90.0),
        WeightRef(weight_id="A", plane=1, angle_deg=93.0),  # 1.57 mm arc < 4 mm
    ]
    with pytest.raises(be.BalancePlanError, match="overlap"):
        be.validate_weights(spec(), source([PIN_A]), [], chosen)


# ---------------------------------------------------------------------------
# engine: search
# ---------------------------------------------------------------------------


def test_search_finds_ideal_two_weight_plan():
    sp = spec()
    src = source([PIN_A])
    candidates, evaluated = be.search_weights(
        sp, src, be.pin_points(sp, src), [], limits()
    )
    assert evaluated > 0
    best = candidates[0]
    assert best.rank == 1
    assert [(w.plane, w.weight_id, w.angle_deg) for w in best.weights] == [
        (1, "A", 240.0),
        (2, "B", 60.0),
    ]
    assert best.residual.static_gmm == 0.0
    assert best.residual.couple_gmm2 == 0.0
    assert best.residual.weight_count == 2
    assert best.residual.within_limit is True
    assert best.imbalance.static_gmm == 0.0
    # ranking is non-decreasing in the residual sort keys
    keys = [
        (c.residual.static_gmm, c.residual.couple_gmm2) for c in candidates
    ]
    assert keys == sorted(keys)


def test_search_excludes_pin_and_seam_positions():
    # pin sits on plane 1 at 60 deg; weight A (d=4) needs 4 mm centre distance
    # -> grid angle 60 on plane 1 is forbidden; seam clearance forbids angle 0
    sp = spec()
    src = source([be.SourcePin("a", 60.0, 20.0, False)])
    candidates, _ = be.search_weights(
        sp, src, be.pin_points(sp, src), [], limits()
    )
    for c in candidates:
        for w in c.weights:
            assert not (w.plane == 1 and w.weight_id == "A" and w.angle_deg == 60.0)
            assert w.angle_deg != 0.0  # inside the seam safety distance


def test_search_respects_inventory_quantity_and_locked_weights():
    # locking the only A weight consumes the stock: no candidate may use A
    sp = spec(inventory=[
        WeightType(id="A", mass_g=0.06, diameter_mm=4.0, quantity=1),
        WeightType(id="B", mass_g=0.012, diameter_mm=2.0, quantity=5),
    ])
    src = source([PIN_A])
    locked = [WeightRef(weight_id="A", plane=1, angle_deg=240.0)]
    locked_pts = be.weight_points(sp, src, locked, "locked")
    candidates, _ = be.search_weights(
        sp, src, be.pin_points(sp, src), locked_pts, limits()
    )
    assert candidates
    for c in candidates:
        assert all(w.weight_id == "B" for w in c.weights)
    # the locked weight already cancels most of the static imbalance
    baseline = be.compute_imbalance(sp, src, be.pin_points(sp, src) + locked_pts)
    assert baseline.static_gmm == 0.36  # 1.8 - 1.44


def test_search_max_weights_zero_returns_baseline_only():
    sp = spec()
    src = source([PIN_A])
    candidates, evaluated = be.search_weights(
        sp, src, be.pin_points(sp, src), [], limits(max_weights=0)
    )
    assert evaluated == 0
    assert len(candidates) == 1
    assert candidates[0].weights == []
    assert candidates[0].residual.static_gmm == 1.44


def test_search_deterministic():
    sp = spec()
    src = source([PIN_A, be.SourcePin("b", 200.0, 50.0, False)])
    pins = be.pin_points(sp, src)
    c1, e1 = be.search_weights(sp, src, pins, [], limits())
    c2, e2 = be.search_weights(sp, src, pins, [], limits())
    assert e1 == e2
    assert c1 == c2


def test_angle_grid():
    assert be.angle_grid(90.0) == [0.0, 90.0, 180.0, 270.0]
    assert be.angle_grid(15.0)[-1] == 345.0
    assert be.angle_grid(50.0) == [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def freeze_source(client: TestClient, notes: list[dict]) -> int:
    body = {"arrangement": make_arrangement(notes), "solution": {}}
    r = client.post("/api/versions", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.fixture()
def source_id(client: TestClient) -> int:
    # note "a": beat 2 -> angle 60 deg, pitch 60 -> axial 10 mm
    return freeze_source(client, [note("a", 2.0, 60)])


def test_analyze_returns_imbalance_and_contributions(client: TestClient, source_id: int):
    r = client.post(
        "/api/balance/analyze",
        json={"source_version_id": source_id, "spec": make_spec()},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source_version_id"] == source_id
    imb = body["imbalance"]
    assert imb["static_gmm"] == 1.44
    assert imb["static_angle_deg"] == 60.0
    assert imb["couple_gmm2"] == 43.2
    assert imb["within_limit"] is False
    assert body["derived"]["pin_radius_mm"] == 32.0
    assert body["derived"]["mid_plane_mm"] == 40.0
    (p,) = body["pins"]
    assert p["note_id"] == "a"
    assert p["static_x_gmm"] == 0.72
    assert body["ideal_correction"]["plane_1"] == {"mass_g": 0.06, "angle_deg": 240.0}


def test_analyze_includes_locked_weights(client: TestClient, source_id: int):
    r = client.post(
        "/api/balance/analyze",
        json={
            "source_version_id": source_id,
            "spec": make_spec(),
            "locked_weights": [{"weight_id": "A", "plane": 1, "angle_deg": 240.0}],
        },
    )
    assert r.status_code == 200
    assert r.json()["imbalance"]["static_gmm"] == 0.36


def test_analyze_unknown_source_version_404(client: TestClient):
    r = client.post("/api/balance/analyze", json={"source_version_id": 999, "spec": make_spec()})
    assert r.status_code == 404


def test_analyze_rejects_unknown_pin_mass_id(client: TestClient, source_id: int):
    r = client.post(
        "/api/balance/analyze",
        json={"source_version_id": source_id, "spec": make_spec(pin_masses={"ghost": 0.1})},
    )
    assert r.status_code == 422
    assert "unknown note ids" in r.json()["detail"]


def test_analyze_rejects_plane_beyond_length(client: TestClient, source_id: int):
    r = client.post(
        "/api/balance/analyze",
        json={"source_version_id": source_id, "spec": make_spec(plane_2_mm=99.0)},
    )
    assert r.status_code == 422
    assert "exceeds cylinder length" in r.json()["detail"]


def test_analyze_pydantic_validation(client: TestClient, source_id: int):
    def post(spec):
        return client.post(
            "/api/balance/analyze", json={"source_version_id": source_id, "spec": spec}
        )

    assert post(make_spec(shell_mass_g=-1.0)).status_code == 422
    assert post(make_spec(working_rpm=0.0)).status_code == 422
    assert post(make_spec(plane_1_mm=70.0, plane_2_mm=60.0)).status_code == 422
    assert post(make_spec(bearing_a_mm=10.0, bearing_b_mm=10.0)).status_code == 422
    assert post(make_spec(pin_masses={"a": -0.1})).status_code == 422
    r = client.post(
        "/api/balance/analyze",
        json={
            "source_version_id": source_id,
            "spec": make_spec(),
            "locked_weights": [{"weight_id": "A", "plane": 1, "angle_deg": 360.0}],
        },
    )
    assert r.status_code == 422  # angle must be < 360


def test_search_endpoint_ranks_candidates(client: TestClient, source_id: int):
    r = client.post(
        "/api/balance/search",
        json={"source_version_id": source_id, "spec": make_spec(), "limits": limits().model_dump()},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feasible"] is True
    assert body["baseline"]["static_gmm"] == 1.44
    best = body["candidates"][0]
    assert best["rank"] == 1
    assert [(w["plane"], w["weight_id"], w["angle_deg"]) for w in best["weights"]] == [
        (1, "A", 240.0),
        (2, "B", 60.0),
    ]
    assert best["residual"]["within_limit"] is True
    ranks = [c["rank"] for c in body["candidates"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_search_unknown_source_404(client: TestClient):
    r = client.post(
        "/api/balance/search", json={"source_version_id": 999, "spec": make_spec()}
    )
    assert r.status_code == 404


def freeze_plan(client: TestClient, source_id: int, **overrides) -> dict:
    body = {
        "source_version_id": source_id,
        "spec": make_spec(),
        "weights": [
            {"weight_id": "A", "plane": 1, "angle_deg": 240.0},
            {"weight_id": "B", "plane": 2, "angle_deg": 60.0},
        ],
    }
    body.update(overrides)
    return client.post("/api/balance/plans", json=body)


def test_freeze_plan_returns_snapshot_hash_and_svg(client: TestClient, source_id: int):
    r = freeze_plan(client, source_id)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] == 1
    assert len(body["content_hash"]) == 64
    assert body["source_version_id"] == source_id
    assert len(body["source_content_hash"]) == 64
    assert body["baseline"]["static_gmm"] == 1.44
    assert body["imbalance"]["static_gmm"] == 0.0
    assert body["imbalance"]["within_limit"] is True
    assert len(body["weights"]) == 2
    assert len(body["pins"]) == 1
    svg = body["svg"]
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert body["content_hash"][:12] in svg
    assert "#6f42c1" in svg  # correction planes
    assert "#fd7e14" in svg  # new weights
    assert "P1" in svg and "P2" in svg


def test_freeze_plan_is_idempotent(client: TestClient, source_id: int):
    r1 = freeze_plan(client, source_id)
    r2 = freeze_plan(client, source_id)
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r1.json()["id"] == r2.json()["id"]
    assert r1.json()["content_hash"] == r2.json()["content_hash"]
    assert r1.json()["svg"] == r2.json()["svg"]
    assert len(client.get("/api/balance/plans").json()) == 1


def test_freeze_plan_unknown_source_404(client: TestClient):
    r = client.post(
        "/api/balance/plans",
        json={"source_version_id": 999, "spec": make_spec(), "weights": []},
    )
    assert r.status_code == 404


def test_freeze_plan_rejects_invalid_input(client: TestClient, source_id: int):
    # per-pin mass referencing an unknown note id
    r = freeze_plan(client, source_id, spec=make_spec(pin_masses={"ghost": 0.1}))
    assert r.status_code == 422 and "unknown note ids" in r.json()["detail"]
    # correction plane beyond the cylinder length
    r = freeze_plan(client, source_id, spec=make_spec(plane_2_mm=99.0))
    assert r.status_code == 422 and "exceeds cylinder length" in r.json()["detail"]
    # weight sticking out beyond the cylinder length
    r = freeze_plan(
        client,
        source_id,
        spec=make_spec(plane_2_mm=79.0),
        weights=[{"weight_id": "A", "plane": 2, "angle_deg": 90.0}],
    )
    assert r.status_code == 422 and "beyond the cylinder length" in r.json()["detail"]
    # unknown weight id
    r = freeze_plan(
        client, source_id, weights=[{"weight_id": "ZZ", "plane": 1, "angle_deg": 0.0}]
    )
    assert r.status_code == 422 and "unknown weight id" in r.json()["detail"]
    # more weights than the stock holds
    r = freeze_plan(
        client,
        source_id,
        spec=make_spec(inventory=[{"id": "A", "mass_g": 0.06, "diameter_mm": 4.0, "quantity": 1}]),
        locked_weights=[{"weight_id": "A", "plane": 1, "angle_deg": 10.0}],
        weights=[{"weight_id": "A", "plane": 2, "angle_deg": 200.0}],
    )
    assert r.status_code == 422 and "in stock" in r.json()["detail"]
    # nothing was stored
    assert client.get("/api/balance/plans").json() == []


def test_get_plan_svg_list_and_recompute(client: TestClient, source_id: int):
    freeze_plan(client, source_id)
    r = client.get("/api/balance/plans/1")
    assert r.status_code == 200
    assert r.json()["weights"][0]["weight_id"] == "A"
    r = client.get("/api/balance/plans/1/svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    listing = client.get("/api/balance/plans").json()
    assert len(listing) == 1
    assert listing[0]["weight_count"] == 2
    assert listing[0]["static_gmm"] == 0.0
    rec = client.post("/api/balance/plans/1/recompute").json()
    assert rec["match"] is True
    assert rec["stored_hash"] == rec["recomputed_hash"]
    assert client.get("/api/balance/plans/999").status_code == 404
    assert client.get("/api/balance/plans/999/svg").status_code == 404
    assert client.post("/api/balance/plans/999/recompute").status_code == 404


def test_plan_recomputes_identically_across_instances(tmp_path):
    hashes, svgs = [], []
    for i in range(2):
        app = create_app(str(tmp_path / f"inst{i}.db"))
        with TestClient(app) as c:
            vid = freeze_source(c, [note("a", 2.0, 60)])
            r = freeze_plan(c, vid)
            assert r.status_code == 201
            hashes.append(r.json()["content_hash"])
            svgs.append(r.json()["svg"])
    assert hashes[0] == hashes[1]
    assert svgs[0] == svgs[1]


def test_full_balance_workflow(client: TestClient, source_id: int):
    # 1. analyze the pinned cylinder
    a = client.post(
        "/api/balance/analyze", json={"source_version_id": source_id, "spec": make_spec()}
    ).json()
    assert a["imbalance"]["within_limit"] is False

    # 2. search the inventory for correction weights
    s = client.post(
        "/api/balance/search",
        json={"source_version_id": source_id, "spec": make_spec()},
    ).json()
    assert s["feasible"] is True
    chosen = [
        {"weight_id": w["weight_id"], "plane": w["plane"], "angle_deg": w["angle_deg"]}
        for w in s["candidates"][0]["weights"]
    ]

    # 3. freeze the chosen plan
    f = client.post(
        "/api/balance/plans",
        json={"source_version_id": source_id, "spec": make_spec(), "weights": chosen},
    )
    assert f.status_code == 201
    plan = f.json()
    assert plan["imbalance"]["within_limit"] is True

    # 4. recompute must reproduce the frozen plan exactly
    rec = client.post(f"/api/balance/plans/{plan['id']}/recompute").json()
    assert rec["match"] is True
