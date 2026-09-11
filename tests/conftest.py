"""Shared fixtures: a small deterministic instrument + score builders.

Geometry of the fixture rig (chosen for hand-checkable numbers):
  diameter 60 mm  -> circumference C = 60*pi ~= 188.495559 mm
  rpm 10          -> 6 s/rev -> 60 deg/s
  bpm 120         -> 0.5 s/beat -> 30 deg/beat
  comb reeds every 10 mm, plus reed 61 only 2 mm from reed 60 (chord tests)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from musicbox.main import create_app


def make_arrangement(notes: list[dict]) -> dict:
    return {
        "bpm": 120.0,
        "notes": notes,
        "cylinder": {"diameter_mm": 60.0, "effective_length_mm": 80.0, "rpm": 10.0},
        "comb": [
            {"pitch": 60, "axial_mm": 10.0},
            {"pitch": 61, "axial_mm": 12.0},
            {"pitch": 62, "axial_mm": 20.0},
            {"pitch": 64, "axial_mm": 30.0},
            {"pitch": 65, "axial_mm": 40.0},
            {"pitch": 67, "axial_mm": 50.0},
            {"pitch": 69, "axial_mm": 60.0},
            {"pitch": 71, "axial_mm": 70.0},
        ],
        "constraints": {
            "pin_diameter_mm": 2.0,
            "seam_zone_mm": 10.0,
            "min_rebound_seconds": 0.3,
            "min_clearance_mm": 1.0,
        },
    }


def note(
    nid: str,
    beat: float,
    pitch: int,
    duration: float = 0.5,
    locked: bool = False,
) -> dict:
    return {
        "id": nid,
        "beat": beat,
        "duration": duration,
        "pitch": pitch,
        "locked": locked,
    }


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c
