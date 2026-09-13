"""HTTP API for the music-box cylinder pin-arrangement service.

Run locally with:  uvicorn musicbox.main:app --reload
The service is fully self-contained: no external services, state lives in a
local SQLite file (env MUSICBOX_DB, default ./musicbox.db).
"""

from __future__ import annotations

import json
import os

from fastapi import Depends, FastAPI, HTTPException, Response

from . import __version__, balance_engine, balance_freeze, engine, freeze
from .balance_models import (
    BalanceAnalyzeRequest,
    BalanceAnalyzeResponse,
    BalanceFreezeRequest,
    BalancePlanResponse,
    BalancePlanSummary,
    BalanceRecomputeResponse,
    BalanceSearchRequest,
    BalanceSearchResponse,
)
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
from .balance_engine import BalancePlanError

IDENTITY = SolutionSpec()


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="Music-Box Cylinder Pin Arranger",
        version=__version__,
        description=(
            "Lay out score notes as pins on a music-box cylinder: geometry "
            "conversion, conflict diagnostics, manufacturability search and "
            "immutable frozen versions with printable SVG unrolls. Includes "
            "two-plane dynamic balancing of the pinned cylinder."
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

    # -- dynamic balancing ----------------------------------------------------

    def _load_source(store: Store, version_id: int) -> balance_engine.SourceInfo:
        row = store.get(version_id)
        if row is None:
            raise HTTPException(status_code=404, detail="source version not found")
        return balance_engine.source_from_version_row(row)

    @app.post("/api/balance/analyze", response_model=BalanceAnalyzeResponse)
    def balance_analyze(
        body: BalanceAnalyzeRequest, store: Store = Depends(get_store)
    ) -> BalanceAnalyzeResponse:
        """Resolve every pin of the source version into a rotating mass vector
        and report centre-of-mass offset, static imbalance, couple about the
        mid-plane, peak bearing loads and per-pin contributions."""
        source = _load_source(store, body.source_version_id)
        try:
            balance_engine.validate_spec(body.spec, source)
            balance_engine.validate_weights(body.spec, source, body.locked_weights, [])
        except BalancePlanError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        pins = balance_engine.pin_points(body.spec, source)
        locked = balance_engine.weight_points(body.spec, source, body.locked_weights, "locked")
        points = pins + locked
        return BalanceAnalyzeResponse(
            source_version_id=source.version_id,
            source_content_hash=source.content_hash,
            derived=balance_engine.derived_info(body.spec, source),
            imbalance=balance_engine.compute_imbalance(body.spec, source, points),
            ideal_correction=balance_engine.ideal_correction(body.spec, source, points),
            pins=balance_engine.pin_contributions(body.spec, source, pins),
            locked_weights=body.locked_weights,
        )

    @app.post("/api/balance/search", response_model=BalanceSearchResponse)
    def balance_search(
        body: BalanceSearchRequest, store: Store = Depends(get_store)
    ) -> BalanceSearchResponse:
        """Search the remaining inventory for weight placements on the two
        correction planes. Placements that would leave the cylinder, violate
        the pin/seam safety distances or overlap mounted weights are excluded.
        Candidates are ranked by residual static imbalance, residual couple,
        added mass and weight count."""
        source = _load_source(store, body.source_version_id)
        try:
            balance_engine.validate_spec(body.spec, source)
            balance_engine.validate_weights(body.spec, source, body.locked_weights, [])
        except BalancePlanError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        pins = balance_engine.pin_points(body.spec, source)
        locked = balance_engine.weight_points(body.spec, source, body.locked_weights, "locked")
        candidates, evaluated = balance_engine.search_weights(
            body.spec, source, pins, locked, body.limits
        )
        return BalanceSearchResponse(
            feasible=any(c.residual.within_limit for c in candidates),
            baseline=balance_engine.compute_imbalance(body.spec, source, pins + locked),
            ideal_correction=balance_engine.ideal_correction(body.spec, source, pins + locked),
            candidates=candidates,
            plans_evaluated=evaluated,
        )

    def _plan_response(row) -> BalancePlanResponse:
        request = json.loads(row["request_json"])
        result = json.loads(row["result_json"])
        return BalancePlanResponse(
            id=row["id"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            source_version_id=request["source"]["version_id"],
            source_content_hash=request["source"]["content_hash"],
            spec=request["spec"],
            locked_weights=request["locked_weights"],
            weights=request["weights"],
            limits=request.get("limits", {}),
            baseline=result["baseline"],
            imbalance=result["imbalance"],
            pins=result["pins"],
            svg=result["svg"],
        )

    @app.post("/api/balance/plans", response_model=BalancePlanResponse, status_code=201)
    def freeze_balance_plan(
        body: BalanceFreezeRequest, response: Response, store: Store = Depends(get_store)
    ) -> BalancePlanResponse:
        """Freeze a chosen balancing plan as an immutable version: source
        snapshot, parameters and input hash are stored, and an SVG unroll
        marking pins, weights and correction planes is generated.
        Idempotent: the same input yields the same plan."""
        source = _load_source(store, body.source_version_id)
        try:
            content_hash, request_dict, result_dict = balance_freeze.build_plan_payload(
                source, body.spec, body.locked_weights, body.weights, body.limits
            )
        except BalancePlanError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        existing = store.get_plan_by_hash(content_hash)
        if existing is not None:
            response.status_code = 200
            return _plan_response(existing)
        row = store.insert_plan(
            content_hash,
            freeze.canonical_json(request_dict),
            json.dumps(result_dict, ensure_ascii=False, sort_keys=True),
        )
        return _plan_response(row)

    @app.get("/api/balance/plans", response_model=list[BalancePlanSummary])
    def list_balance_plans(store: Store = Depends(get_store)) -> list[BalancePlanSummary]:
        out = []
        for row in store.list_plans():
            request = json.loads(row["request_json"])
            result = json.loads(row["result_json"])
            out.append(
                BalancePlanSummary(
                    id=row["id"],
                    content_hash=row["content_hash"],
                    created_at=row["created_at"],
                    source_version_id=request["source"]["version_id"],
                    weight_count=len(request["weights"]),
                    static_gmm=result["imbalance"]["static_gmm"],
                )
            )
        return out

    @app.get("/api/balance/plans/{plan_id}", response_model=BalancePlanResponse)
    def get_balance_plan(plan_id: int, store: Store = Depends(get_store)) -> BalancePlanResponse:
        row = store.get_plan(plan_id)
        if row is None:
            raise HTTPException(status_code=404, detail="balance plan not found")
        return _plan_response(row)

    @app.get("/api/balance/plans/{plan_id}/svg")
    def get_balance_plan_svg(plan_id: int, store: Store = Depends(get_store)) -> Response:
        row = store.get_plan(plan_id)
        if row is None:
            raise HTTPException(status_code=404, detail="balance plan not found")
        svg = json.loads(row["result_json"])["svg"]
        return Response(content=svg, media_type="image/svg+xml")

    @app.post("/api/balance/plans/{plan_id}/recompute", response_model=BalanceRecomputeResponse)
    def recompute_balance_plan(
        plan_id: int, store: Store = Depends(get_store)
    ) -> BalanceRecomputeResponse:
        """Recompute a frozen plan from its stored source snapshot and verify
        the result is bit-identical (same content hash and same SVG)."""
        row = store.get_plan(plan_id)
        if row is None:
            raise HTTPException(status_code=404, detail="balance plan not found")
        request_dict = json.loads(row["request_json"])
        stored_result = json.loads(row["result_json"])
        new_hash, new_result = balance_freeze.recompute_plan(request_dict)
        match = new_hash == row["content_hash"] and new_result == stored_result
        return BalanceRecomputeResponse(
            id=row["id"],
            match=match,
            stored_hash=row["content_hash"],
            recomputed_hash=new_hash,
        )

    return app


app = create_app()
