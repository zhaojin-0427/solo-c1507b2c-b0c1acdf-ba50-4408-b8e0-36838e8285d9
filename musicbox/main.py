"""HTTP API for the music-box cylinder pin-arrangement service.

Run locally with:  uvicorn musicbox.main:app --reload
The service is fully self-contained: no external services, state lives in a
local SQLite file (env MUSICBOX_DB, default ./musicbox.db).
"""

from __future__ import annotations

import json
import os

from fastapi import Depends, FastAPI, HTTPException, Response

from . import __version__, engine, freeze
from .models import (
    ArrangementRequest,
    CheckResponse,
    FreezeRequest,
    RecomputeResponse,
    SearchRequest,
    SearchResponse,
    SolutionSpec,
    VersionResponse,
    VersionSummary,
)
from .store import Store

IDENTITY = SolutionSpec()


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="Music-Box Cylinder Pin Arranger",
        version=__version__,
        description=(
            "Lay out score notes as pins on a music-box cylinder: geometry "
            "conversion, conflict diagnostics, manufacturability search and "
            "immutable frozen versions with printable SVG unrolls."
        ),
    )
    app.state.store = Store(db_path or os.environ.get("MUSICBOX_DB", "musicbox.db"))

    def get_store() -> Store:
        return app.state.store

    # -- diagnostics ------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/api/arrangements/check", response_model=CheckResponse)
    def check(arr: ArrangementRequest) -> CheckResponse:
        """Convert notes to cylinder coordinates and locate the first
        occurrence of each problem kind (missing pitch, chord collision,
        seam crossing, too-dense repeats, pin clearance)."""
        placed, missing, _ = engine.place(arr, IDENTITY)
        issues = engine.diagnose(arr, placed, missing)
        diagnostics = engine.build_diagnostics(issues)
        return CheckResponse(
            ok=diagnostics.ok,
            derived=engine.derived_info(arr),
            pins=engine.build_pins(arr, placed),
            missing_notes=engine.missing_notes_out(missing),
            diagnostics=diagnostics,
        )

    @app.post("/api/arrangements/search", response_model=SearchResponse)
    def search_route(body: SearchRequest) -> SearchResponse:
        """Search transposition / tempo-float / quantization combinations for
        manufacturable layouts, ranked by deletions, pitch deviation, rhythm
        error and minimum clearance. Locked notes are never adjusted."""
        candidates, evaluated = engine.search(body.arrangement, body.limits)
        return SearchResponse(
            feasible=bool(candidates),
            combinations_evaluated=evaluated,
            candidates=candidates,
        )

    # -- frozen versions ----------------------------------------------------

    def _version_response(row) -> VersionResponse:
        request = json.loads(row["request_json"])
        result = json.loads(row["result_json"])
        return VersionResponse(
            id=row["id"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            arrangement=request["arrangement"],
            solution=request["solution"],
            pins=result["pins"],
            metrics=result["metrics"],
            svg=result["svg"],
        )

    @app.post("/api/versions", response_model=VersionResponse, status_code=201)
    def freeze_version(body: FreezeRequest, response: Response, store: Store = Depends(get_store)) -> VersionResponse:
        """Freeze a chosen solution as an immutable version. Idempotent: the
        same arrangement + solution yields the same version."""
        try:
            content_hash, request_dict, result_dict = freeze.build_payload(
                body.arrangement, body.solution
            )
        except freeze.InfeasibleSolution as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "solution is not manufacturable",
                    "diagnostics": exc.diagnostics.model_dump(mode="json"),
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        existing = store.get_by_hash(content_hash)
        if existing is not None:
            response.status_code = 200
            return _version_response(existing)
        row = store.insert(
            content_hash,
            freeze.canonical_json(request_dict),
            json.dumps(result_dict, ensure_ascii=False, sort_keys=True),
        )
        return _version_response(row)

    @app.get("/api/versions", response_model=list[VersionSummary])
    def list_versions(store: Store = Depends(get_store)) -> list[VersionSummary]:
        out = []
        for row in store.list():
            request = json.loads(row["request_json"])
            result = json.loads(row["result_json"])
            out.append(
                VersionSummary(
                    id=row["id"],
                    content_hash=row["content_hash"],
                    created_at=row["created_at"],
                    pin_count=len(result["pins"]),
                    deleted_count=len(request["solution"]["deleted_note_ids"]),
                )
            )
        return out

    @app.get("/api/versions/{version_id}", response_model=VersionResponse)
    def get_version(version_id: int, store: Store = Depends(get_store)) -> VersionResponse:
        row = store.get(version_id)
        if row is None:
            raise HTTPException(status_code=404, detail="version not found")
        return _version_response(row)

    @app.get("/api/versions/{version_id}/svg")
    def get_version_svg(version_id: int, store: Store = Depends(get_store)) -> Response:
        row = store.get(version_id)
        if row is None:
            raise HTTPException(status_code=404, detail="version not found")
        svg = json.loads(row["result_json"])["svg"]
        return Response(content=svg, media_type="image/svg+xml")

    @app.post("/api/versions/{version_id}/recompute", response_model=RecomputeResponse)
    def recompute_version(version_id: int, store: Store = Depends(get_store)) -> RecomputeResponse:
        """Recompute a frozen version from its stored request and verify the
        result is bit-identical (same content hash and same SVG)."""
        row = store.get(version_id)
        if row is None:
            raise HTTPException(status_code=404, detail="version not found")
        request_dict = json.loads(row["request_json"])
        stored_result = json.loads(row["result_json"])
        new_hash, new_result = freeze.recompute(request_dict)
        match = new_hash == row["content_hash"] and new_result == stored_result
        return RecomputeResponse(
            id=row["id"],
            match=match,
            stored_hash=row["content_hash"],
            recomputed_hash=new_hash,
        )

    return app


app = create_app()
