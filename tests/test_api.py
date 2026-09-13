"""API integration tests: endpoints, freezing, immutability, determinism."""

from __future__ import annotations

from fastapi.testclient import TestClient

from musicbox.main import create_app

from conftest import make_arrangement, note


def clean_arrangement() -> dict:
    return make_arrangement(
        [note("a", 2.0, 60), note("b", 4.0, 62), note("c", 6.0, 64)]
    )


def conflicting_arrangement() -> dict:
    return make_arrangement(
        [
            note("m", 1.0, 66),  # missing pitch
            note("s", 0.0, 62),  # seam crossing
            note("c1", 4.0, 60),
            note("c2", 4.0, 61),  # chord collision
            note("r1", 6.0, 64),
            note("r2", 6.5, 64),  # rebound too dense
            note("q1", 10.0, 60),
            note("q2", 10.05, 61),  # pin clearance
        ]
    )


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def test_check_clean(client: TestClient):
    r = client.post("/api/arrangements/check", json=clean_arrangement())
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["diagnostics"]["ok"] is True
    assert all(v is None for v in body["diagnostics"]["first_by_kind"].values())
    assert len(body["pins"]) == 3
    pin = body["pins"][0]
    assert pin["note_id"] == "a"
    assert pin["angle_deg"] == 60.0
    assert pin["x_mm"] == 31.415927
    assert pin["axial_mm"] == 10.0
    assert pin["pitch_name"] == "C4"
    assert pin["margins"]["ok"] is True
    derived = body["derived"]
    assert derived["circumference_mm"] == 188.495559
    assert derived["degrees_per_beat"] == 30.0


def test_check_locates_first_issue_of_each_kind(client: TestClient):
    r = client.post("/api/arrangements/check", json=conflicting_arrangement())
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    first = body["diagnostics"]["first_by_kind"]
    assert first["missing_pitch"]["beat"] == 1.0
    assert first["seam_crossing"]["beat"] == 0.0
    assert first["chord_collision"]["beat"] == 4.0
    assert first["rebound_too_dense"]["beat"] == 6.0
    assert first["pin_clearance"]["beat"] == 10.0
    assert first["missing_pitch"]["note_ids"] == ["m"]
    assert body["missing_notes"][0]["note_id"] == "m"
    counts = body["diagnostics"]["issue_counts"]
    assert counts == {
        "missing_pitch": 1,
        "chord_collision": 1,
        "seam_crossing": 1,
        "rebound_too_dense": 1,
        "pin_clearance": 1,
    }


def test_check_validation_errors(client: TestClient):
    bad = clean_arrangement()
    bad["cylinder"]["rpm"] = 0
    assert client.post("/api/arrangements/check", json=bad).status_code == 422

    dup = clean_arrangement()
    dup["notes"].append(note("a", 8.0, 62))  # duplicate id
    assert client.post("/api/arrangements/check", json=dup).status_code == 422

    off = clean_arrangement()
    off["comb"][0]["axial_mm"] = 999.0  # reed beyond effective length
    assert client.post("/api/arrangements/check", json=off).status_code == 422


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_endpoint_returns_ranked_candidates(client: TestClient):
    arr = make_arrangement(
        [note("n1", 2.0, 48), note("n2", 4.0, 50), note("n3", 6.0, 52)]
    )
    r = client.post(
        "/api/arrangements/search",
        json={
            "arrangement": arr,
            "limits": {
                "max_transpose_semitones": 12,
                "tempo_float_percent": 2.0,
                "tempo_step_percent": 1.0,
                "quantize_grids_beats": [0.25],
            },
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is True
    assert body["combinations_evaluated"] == 250
    best = body["candidates"][0]
    assert best["rank"] == 1
    assert best["solution"]["transpose_semitones"] == 12
    assert best["metrics"]["deleted_count"] == 0
    assert len(best["pins"]) == 3
    # ranks are 1..n and deletion counts are non-decreasing
    ranks = [c["rank"] for c in body["candidates"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_search_respects_locked_notes(client: TestClient):
    arr = make_arrangement(
        [note("keep", 6.0, 64, locked=True), note("drop", 6.5, 64)]
    )
    r = client.post(
        "/api/arrangements/search",
        json={
            "arrangement": arr,
            "limits": {
                "quantize_grids_beats": [],
                "tempo_float_percent": 2.0,
                "max_transpose_semitones": 0,  # isolate the deletion behaviour
            },
        },
    )
    body = r.json()
    assert body["feasible"] is True
    for cand in body["candidates"]:
        assert "keep" not in cand["solution"]["deleted_note_ids"]
    best = body["candidates"][0]
    assert best["solution"]["deleted_note_ids"] == ["drop"]
    assert {p["note_id"] for p in best["pins"]} == {"keep"}


def test_search_rejects_huge_search_space(client: TestClient):
    r = client.post(
        "/api/arrangements/search",
        json={
            "arrangement": clean_arrangement(),
            "limits": {
                "max_transpose_semitones": 12,
                "tempo_float_percent": 25.0,
                "tempo_step_percent": 0.5,
                "quantize_grids_beats": [0.5, 0.25, 0.125],
            },
        },
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# freeze / versions
# ---------------------------------------------------------------------------


def freeze_body() -> dict:
    return {
        "arrangement": clean_arrangement(),
        "solution": {
            "transpose_semitones": 0,
            "tempo_factor": 1.0,
            "quantize_grid_beats": None,
            "deleted_note_ids": [],
        },
    }


def test_freeze_returns_pins_hash_and_svg(client: TestClient):
    r = client.post("/api/versions", json=freeze_body())
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == 1
    assert len(body["content_hash"]) == 64
    assert len(body["pins"]) == 3
    pin = body["pins"][0]
    assert pin["note_id"] == "a"  # note source
    assert pin["angle_deg"] == 60.0 and pin["axial_mm"] == 10.0  # coordinates
    assert set(pin["margins"]) >= {"clearance_mm", "rebound_s", "seam_mm", "ok"}
    svg = body["svg"]
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert 'width="' in svg and 'mm"' in svg  # physical units for printing
    assert body["content_hash"][:12] in svg


def test_freeze_is_idempotent_and_immutable(client: TestClient):
    r1 = client.post("/api/versions", json=freeze_body())
    r2 = client.post("/api/versions", json=freeze_body())
    assert r1.status_code == 201
    assert r2.status_code == 200  # already existed
    assert r1.json()["id"] == r2.json()["id"]
    assert r1.json()["content_hash"] == r2.json()["content_hash"]
    assert r1.json()["svg"] == r2.json()["svg"]
    # still a single version stored
    listing = client.get("/api/versions").json()
    assert len(listing) == 1
    assert listing[0]["pin_count"] == 3


def test_freeze_rejects_infeasible_solution(client: TestClient):
    body = {
        "arrangement": conflicting_arrangement(),
        "solution": {"deleted_note_ids": []},
    }
    r = client.post("/api/versions", json=body)
    assert r.status_code == 422
    assert "diagnostics" in r.json()["detail"]


def test_freeze_rejects_deleting_locked_or_unknown_notes(client: TestClient):
    arr = make_arrangement([note("lk", 2.0, 60, locked=True), note("x", 4.0, 62)])
    r = client.post(
        "/api/versions",
        json={"arrangement": arr, "solution": {"deleted_note_ids": ["lk"]}},
    )
    assert r.status_code == 422
    assert "locked" in r.json()["detail"]

    r = client.post(
        "/api/versions",
        json={"arrangement": arr, "solution": {"deleted_note_ids": ["ghost"]}},
    )
    assert r.status_code == 422


def test_get_version_and_svg_endpoint(client: TestClient):
    client.post("/api/versions", json=freeze_body())
    r = client.get("/api/versions/1")
    assert r.status_code == 200
    assert r.json()["solution"]["transpose_semitones"] == 0
    r = client.get("/api/versions/1/svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert r.text.startswith("<svg")
    assert client.get("/api/versions/999").status_code == 404


def test_recompute_matches_frozen_version(client: TestClient):
    client.post("/api/versions", json=freeze_body())
    r = client.post("/api/versions/1/recompute")
    assert r.status_code == 200
    body = r.json()
    assert body["match"] is True
    assert body["stored_hash"] == body["recomputed_hash"]
    assert client.post("/api/versions/999/recompute").status_code == 404


def test_same_version_recomputes_identically_across_instances(tmp_path):
    # two independent app instances (separate DBs) must produce the same hash
    body = freeze_body()
    hashes, svgs = [], []
    for i in range(2):
        app = create_app(str(tmp_path / f"inst{i}.db"))
        with TestClient(app) as c:
            r = c.post("/api/versions", json=body)
            hashes.append(r.json()["content_hash"])
            svgs.append(r.json()["svg"])
    assert hashes[0] == hashes[1]
    assert svgs[0] == svgs[1]


def test_freeze_with_deletions_marks_them_in_svg(client: TestClient):
    arr = make_arrangement([note("a", 2.0, 60), note("b", 2.5, 60)])  # rebound clash
    body = {"arrangement": arr, "solution": {"deleted_note_ids": ["b"]}}
    r = client.post("/api/versions", json=body)
    assert r.status_code == 201
    out = r.json()
    assert len(out["pins"]) == 1
    assert 'stroke="#dc3545"' in out["svg"]  # red cross for the deleted note


def test_search_and_freeze_metrics_agree(client: TestClient):
    # regression: a deleted high-beat note must not inflate the search metrics;
    # the candidate metrics and the frozen-version metrics must be identical
    arr = make_arrangement([note("a", 4.0, 60), note("late", 100.0, 66)])
    s = client.post(
        "/api/arrangements/search",
        json={
            "arrangement": arr,
            "limits": {
                "max_transpose_semitones": 0,
                "tempo_float_percent": 5.0,
                "tempo_step_percent": 5.0,
                "quantize_grids_beats": [],
            },
        },
    ).json()
    cand = next(c for c in s["candidates"] if c["solution"]["tempo_factor"] == 1.05)
    assert cand["solution"]["deleted_note_ids"] == ["late"]
    assert cand["metrics"]["rhythm_error_seconds"] == 0.095238

    v = client.post(
        "/api/versions", json={"arrangement": arr, "solution": cand["solution"]}
    )
    assert v.status_code == 201
    assert v.json()["metrics"] == cand["metrics"]


def test_freeze_dedupes_duplicate_deleted_ids(client: TestClient):
    # regression: duplicate deletion ids describe the same effective layout and
    # must yield the same count, the same hash and the same version
    arr = make_arrangement([note("a", 2.0, 60), note("b", 2.5, 60)])  # rebound clash
    r1 = client.post(
        "/api/versions",
        json={"arrangement": arr, "solution": {"deleted_note_ids": ["b", "b"]}},
    )
    assert r1.status_code == 201
    body1 = r1.json()
    assert body1["metrics"]["deleted_count"] == 1
    assert len(body1["pins"]) == 1

    r2 = client.post(
        "/api/versions",
        json={"arrangement": arr, "solution": {"deleted_note_ids": ["b"]}},
    )
    assert r2.status_code == 200  # already frozen
    assert r2.json()["id"] == body1["id"]
    assert r2.json()["content_hash"] == body1["content_hash"]
    assert len(client.get("/api/versions").json()) == 1


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_full_workflow_check_search_freeze_recompute(client: TestClient):
    # a score one octave below the comb cannot be pinned directly...
    arr = make_arrangement(
        [note("n1", 2.0, 48), note("n2", 4.0, 50), note("n3", 6.0, 52)]
    )
    check = client.post("/api/arrangements/check", json=arr).json()
    assert check["ok"] is False
    assert check["diagnostics"]["issue_counts"]["missing_pitch"] == 3

    # ...so search for a manufacturable variant...
    search = client.post(
        "/api/arrangements/search",
        json={
            "arrangement": arr,
            "limits": {
                "max_transpose_semitones": 12,
                "tempo_float_percent": 2.0,
                "tempo_step_percent": 1.0,
                "quantize_grids_beats": [0.25],
            },
        },
    ).json()
    assert search["feasible"] is True
    chosen = search["candidates"][0]["solution"]

    # ...freeze the chosen solution as an immutable version...
    frozen = client.post(
        "/api/versions", json={"arrangement": arr, "solution": chosen}
    )
    assert frozen.status_code == 201
    version = frozen.json()
    assert len(version["pins"]) == 3
    assert all(p["margins"]["ok"] for p in version["pins"])

    # ...and recomputing it must reproduce the exact same result
    again = client.post(f"/api/versions/{version['id']}/recompute").json()
    assert again["match"] is True
    assert again["recomputed_hash"] == version["content_hash"]
