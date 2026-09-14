"""HTTP API for the music-box cylinder pin-arrangement service.

Run locally with:  uvicorn musicbox.main:app --reload
The service is fully self-contained: no external services, state lives in a
local SQLite file (env MUSICBOX_DB, default ./musicbox.db).
"""

from __future__ import annotations

import json
import os

from fastapi import Depends, FastAPI, HTTPException, Response

from . import __version__, balance_engine, balance_freeze, calibration_engine, calibration_freeze, dynamics_engine, dynamics_freeze, engine, freeze
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
from .calibration_models import (
    CalibrationBatchRequest,
    CalibrationBatchResponse,
    CalibrationBatchSummary,
    CalibrationConfirmRequest,
    CalibrationPlanResponse,
    CalibrationPlanSummary,
    CalibrationRecomputeResponse,
    CalibrationSearchRequest,
    CalibrationSearchResponse,
    CalibrationSpec,
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
from .calibration_engine import CalibrationError
from .dynamics_models import (
    CandidateMetrics,
    DynamicsPlanResponse,
    DynamicsPlanSummary,
    DynamicsRecomputeResponse,
    DynamicsSpec,
    PlanFreezeRequest,
    ScenarioCandidate,
    ScenarioParams,
    ScenarioSearchRequest,
    ScenarioSearchResponse,
    SimulateResponse,
    TrialCreateRequest,
    TrialResponse,
    TrialSummary,
)
from .dynamics_engine import DynamicsError

IDENTITY = SolutionSpec()


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="Music-Box Cylinder Pin Arranger",
        version=__version__,
        description=(
            "Lay out score notes as pins on a music-box cylinder: geometry "
            "conversion, conflict diagnostics, manufacturability search and "
            "immutable frozen versions with printable SVG unrolls. Includes "
            "two-plane dynamic balancing, powertrain dynamics trials and "
            "assembly calibration of the pinned cylinder against the comb."
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

    # -- powertrain dynamics trials ------------------------------------------

    def _load_dynamics_source(store: Store, version_id: int) -> dynamics_engine.DynamicsSource:
        row = store.get(version_id)
        if row is None:
            raise HTTPException(status_code=404, detail="source version not found")
        return dynamics_engine.source_from_version_row(row)

    def _load_trial(store: Store, trial_id: int) -> tuple:
        row = store.get_trial(trial_id)
        if row is None:
            raise HTTPException(status_code=404, detail="dynamics trial not found")
        request = json.loads(row["request_json"])
        result = json.loads(row["result_json"])
        source = dynamics_engine.DynamicsSource.from_snapshot_dict(
            {
                "version_id": request["source_version_id"],
                "content_hash": result["source_content_hash"],
                "design_rpm": result["derived"]["design_rpm"],
                "pins": result["pins"],
            }
        )
        return row, request, result, source

    def _trial_response(row) -> TrialResponse:
        request = json.loads(row["request_json"])
        result = json.loads(row["result_json"])
        return TrialResponse(
            id=row["id"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            source_version_id=request["source_version_id"],
            source_content_hash=result["source_content_hash"],
            spec=request["spec"],
            derived=result["derived"],
            pins=result["pins"],
        )

    @app.post("/api/dynamics/trials", response_model=TrialResponse, status_code=201)
    def create_dynamics_trial(
        body: TrialCreateRequest, response: Response, store: Store = Depends(get_store)
    ) -> TrialResponse:
        """Create an immutable dynamics trial: mainspring torque curve, gear
        ratios and efficiencies, inertia, governor drag curve and per-reed
        pluck energies, referencing a frozen pin-arrangement version. Rejected
        (404/422) when the source version is missing, a curve abscissa is not
        strictly increasing, or the usable mainspring travel is insufficient."""
        source = _load_dynamics_source(store, body.source_version_id)
        try:
            content_hash, request_dict, result_dict = dynamics_freeze.build_trial_payload(
                source, body.spec
            )
        except DynamicsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        existing = store.get_trial_by_hash(content_hash)
        if existing is not None:
            response.status_code = 200
            return _trial_response(existing)
        row = store.insert_trial(
            content_hash,
            freeze.canonical_json(request_dict),
            json.dumps(result_dict, ensure_ascii=False, sort_keys=True),
        )
        return _trial_response(row)

    @app.get("/api/dynamics/trials", response_model=list[TrialSummary])
    def list_dynamics_trials(store: Store = Depends(get_store)) -> list[TrialSummary]:
        out = []
        for row in store.list_trials():
            result = json.loads(row["result_json"])
            out.append(
                TrialSummary(
                    id=row["id"],
                    content_hash=row["content_hash"],
                    created_at=row["created_at"],
                    source_version_id=json.loads(row["request_json"])["source_version_id"],
                    pin_count=len(result["pins"]),
                )
            )
        return out

    @app.get("/api/dynamics/trials/{trial_id}", response_model=TrialResponse)
    def get_dynamics_trial(trial_id: int, store: Store = Depends(get_store)) -> TrialResponse:
        row = store.get_trial(trial_id)
        if row is None:
            raise HTTPException(status_code=404, detail="dynamics trial not found")
        return _trial_response(row)

    @app.post(
        "/api/dynamics/trials/{trial_id}/simulate",
        response_model=SimulateResponse,
    )
    def simulate_dynamics(
        trial_id: int,
        body: ScenarioParams,
        store: Store = Depends(get_store),
    ) -> SimulateResponse:
        """Integrate cylinder angular velocity at the trial's fixed step and
        apply pluck loads as each pin design phase is crossed. Returns speed and
        torque-margin curves, the post-pluck minimum speeds, accumulated run
        time and the pin/time of the first stall, overspeed and beat drift."""
        row, _request, result, source = _load_trial(store, trial_id)
        spec = DynamicsSpec.model_validate(json.loads(row["request_json"])["spec"])
        try:
            sim = dynamics_engine.simulate(spec, source, body, keep_curves=True)
        except DynamicsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SimulateResponse(
            trial_id=trial_id,
            source_content_hash=result["source_content_hash"],
            **sim.model_dump(mode="json"),
        )

    @app.post(
        "/api/dynamics/trials/{trial_id}/search",
        response_model=ScenarioSearchResponse,
    )
    def search_scenarios(
        trial_id: int,
        body: ScenarioSearchRequest,
        store: Store = Depends(get_store),
    ) -> ScenarioSearchResponse:
        """Search prewind × governor coefficient × flywheel inertia grids,
        optionally locking the gear ratio. Plans are ranked by violation count,
        then maximum beat deviation, keeping larger torque margins and larger
        remaining mainspring travel first."""
        row, _request, _result, source = _load_trial(store, trial_id)
        spec = DynamicsSpec.model_validate(json.loads(row["request_json"])["spec"])
        stored_ratio = dynamics_engine.total_ratio(spec)
        if body.gear_ratio_lock is not None:
            ratios = [body.gear_ratio_lock]
        elif body.gear_ratio_candidates:
            ratios = list(body.gear_ratio_candidates)
        else:
            ratios = [stored_ratio]
        try:
            scored, evaluated = dynamics_engine.search_scenarios(
                spec,
                source,
                ratios=ratios,
                prewinds=body.prewind_turns,
                coefficients=body.governor_coefficients,
                inertias=body.flywheel_inertia_g_cm2,
                max_candidates=body.max_candidates,
            )
        except DynamicsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not scored:
            raise HTTPException(
                status_code=422,
                detail="no feasible scenario: every prewind exceeds the usable "
                "mainspring travel",
            )
        candidates = [
            ScenarioCandidate(
                rank=i + 1,
                scenario=c["scenario"],
                metrics=CandidateMetrics(
                    violation_count=c["summary"].violation_count,
                    max_beat_drift_ratio=c["summary"].max_beat_drift_ratio,
                    min_torque_margin_mNm=c["summary"].min_torque_margin_mNm,
                    remaining_spring_turns=c["summary"].remaining_spring_turns,
                    min_rpm=c["summary"].min_rpm,
                    total_time_s=c["summary"].total_time_s,
                    completed=c["summary"].completed,
                ),
                pluck_min_rpm=c["pluck_min_rpm"],
                first_violations=c["first_violations"],
            )
            for i, c in enumerate(scored)
        ]
        return ScenarioSearchResponse(
            feasible=any(c.metrics.violation_count == 0 for c in candidates),
            combinations_evaluated=evaluated,
            candidates=candidates,
        )

    def _dynamics_plan_response(row) -> DynamicsPlanResponse:
        request = json.loads(row["request_json"])
        result = json.loads(row["result_json"])
        return DynamicsPlanResponse(
            id=row["id"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            trial_content_hash=request["trial_content_hash"],
            source_version_id=request["source"]["version_id"],
            source_content_hash=request["source"]["content_hash"],
            spec=request["spec"],
            scenario=result["scenario"],
            dt_s=request["dt_s"],
            result=result,
        )

    @app.post(
        "/api/dynamics/plans", response_model=DynamicsPlanResponse, status_code=201
    )
    def freeze_dynamics_plan(
        body: PlanFreezeRequest, response: Response, store: Store = Depends(get_store)
    ) -> DynamicsPlanResponse:
        """Freeze a chosen scenario: source snapshot, input curves, integration
        step, resolved scenario and input hash; recomputation is self-contained
        and bit-identical. Idempotent: the same input yields the same plan."""
        row, request, _result, source = _load_trial(store, body.trial_id)
        trial_hash = row["content_hash"]
        spec = DynamicsSpec.model_validate(request["spec"])
        try:
            content_hash, request_dict, result_dict = dynamics_freeze.build_plan_payload(
                source, spec, body.scenario
            )
        except DynamicsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        request_dict["trial_content_hash"] = trial_hash
        request_dict["dt_s"] = spec.dt_s
        existing = store.get_dynamics_plan_by_hash(content_hash)
        if existing is not None:
            response.status_code = 200
            return _dynamics_plan_response(existing)
        new_row = store.insert_dynamics_plan(
            content_hash,
            freeze.canonical_json(request_dict),
            json.dumps(result_dict, ensure_ascii=False, sort_keys=True),
        )
        return _dynamics_plan_response(new_row)

    @app.get("/api/dynamics/plans", response_model=list[DynamicsPlanSummary])
    def list_dynamics_plans(store: Store = Depends(get_store)) -> list[DynamicsPlanSummary]:
        out = []
        for row in store.list_dynamics_plans():
            request = json.loads(row["request_json"])
            result = json.loads(row["result_json"])
            out.append(
                DynamicsPlanSummary(
                    id=row["id"],
                    content_hash=row["content_hash"],
                    created_at=row["created_at"],
                    trial_content_hash=request["trial_content_hash"],
                    violation_count=result["summary"]["violation_count"],
                )
            )
        return out

    @app.get("/api/dynamics/plans/{plan_id}", response_model=DynamicsPlanResponse)
    def get_dynamics_plan(
        plan_id: int, store: Store = Depends(get_store)
    ) -> DynamicsPlanResponse:
        row = store.get_dynamics_plan(plan_id)
        if row is None:
            raise HTTPException(status_code=404, detail="dynamics plan not found")
        return _dynamics_plan_response(row)

    @app.post(
        "/api/dynamics/plans/{plan_id}/recompute",
        response_model=DynamicsRecomputeResponse,
    )
    def recompute_dynamics_plan(
        plan_id: int, store: Store = Depends(get_store)
    ) -> DynamicsRecomputeResponse:
        """Recompute a frozen plan from its stored source snapshot and curves;
        the result must be bit-identical (same content hash, same curves)."""
        row = store.get_dynamics_plan(plan_id)
        if row is None:
            raise HTTPException(status_code=404, detail="dynamics plan not found")
        request_dict = json.loads(row["request_json"])
        stored_result = json.loads(row["result_json"])
        new_hash, new_result = dynamics_freeze.recompute_plan(request_dict)
        match = new_hash == row["content_hash"] and new_result == stored_result
        return DynamicsRecomputeResponse(
            id=row["id"],
            match=match,
            stored_hash=row["content_hash"],
            recomputed_hash=new_hash,
        )

    # -- assembly calibration -------------------------------------------------

    def _load_cal_source(store: Store, version_id: int) -> calibration_engine.CalSource:
        row = store.get(version_id)
        if row is None:
            raise HTTPException(status_code=404, detail="source version not found")
        return calibration_engine.source_from_version_row(row)

    def _load_cal_batch(store: Store, batch_id: int) -> tuple:
        row = store.get_cal_batch(batch_id)
        if row is None:
            raise HTTPException(status_code=404, detail="calibration batch not found")
        request = json.loads(row["request_json"])
        result = json.loads(row["result_json"])
        source = calibration_engine.CalSource.from_snapshot_dict(result["source"])
        spec = CalibrationSpec.model_validate(request["spec"])
        return row, request, result, source, spec

    def _cal_batch_response(row, store: Store) -> CalibrationBatchResponse:
        request = json.loads(row["request_json"])
        result = json.loads(row["result_json"])
        plan = store.get_cal_plan_for_batch(row["content_hash"])
        return CalibrationBatchResponse(
            id=row["id"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            status="confirmed" if plan is not None else "collecting",
            confirmed_plan_id=plan["id"] if plan is not None else None,
            source_version_id=request["source_version_id"],
            source_content_hash=result["source_content_hash"],
            spec=request["spec"],
            fit=result["fit"],
            pins=result["pins"],
            diagnostics=result["diagnostics"],
            summary=result["summary"],
        )

    @app.post(
        "/api/calibration/batches",
        response_model=CalibrationBatchResponse,
        status_code=201,
    )
    def create_calibration_batch(
        body: CalibrationBatchRequest, response: Response, store: Store = Depends(get_store)
    ) -> CalibrationBatchResponse:
        """Create an immutable calibration batch (采集中): radial runout at the
        datum stations, bearing heights, pin heights and measured reed tips,
        referencing a frozen pin-arrangement version. The cylinder axis and
        eccentricity are fitted and every pin is resolved against the comb.
        Rejected (404/422) when the source version is missing, the runout grid
        does not cover the pins, or a reed/pin reference is unknown."""
        source = _load_cal_source(store, body.source_version_id)
        try:
            content_hash, request_dict, result_dict = calibration_freeze.build_batch_payload(
                source, body.spec
            )
        except CalibrationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        existing = store.get_cal_batch_by_hash(content_hash)
        if existing is not None:
            response.status_code = 200
            return _cal_batch_response(existing, store)
        row = store.insert_cal_batch(
            content_hash,
            freeze.canonical_json(request_dict),
            json.dumps(result_dict, ensure_ascii=False, sort_keys=True),
        )
        return _cal_batch_response(row, store)

    @app.get("/api/calibration/batches", response_model=list[CalibrationBatchSummary])
    def list_calibration_batches(
        store: Store = Depends(get_store),
    ) -> list[CalibrationBatchSummary]:
        out = []
        for row in store.list_cal_batches():
            request = json.loads(row["request_json"])
            result = json.loads(row["result_json"])
            out.append(
                CalibrationBatchSummary(
                    id=row["id"],
                    content_hash=row["content_hash"],
                    created_at=row["created_at"],
                    status=(
                        "confirmed"
                        if store.get_cal_plan_for_batch(row["content_hash"]) is not None
                        else "collecting"
                    ),
                    source_version_id=request["source_version_id"],
                    pin_count=result["summary"]["pin_count"],
                    violation_count=result["summary"]["violation_count"],
                )
            )
        return out

    @app.get("/api/calibration/batches/{batch_id}", response_model=CalibrationBatchResponse)
    def get_calibration_batch(
        batch_id: int, store: Store = Depends(get_store)
    ) -> CalibrationBatchResponse:
        row = store.get_cal_batch(batch_id)
        if row is None:
            raise HTTPException(status_code=404, detail="calibration batch not found")
        return _cal_batch_response(row, store)

    @app.post(
        "/api/calibration/batches/{batch_id}/search",
        response_model=CalibrationSearchResponse,
    )
    def search_calibration(
        batch_id: int,
        body: CalibrationSearchRequest,
        store: Store = Depends(get_store),
    ) -> CalibrationSearchResponse:
        """Search bearing shim stacks x comb lateral shift x comb height
        adjustment within the maker's locks. Candidates are ranked by
        violation count, then minimum margin (larger first), adjustment
        amount and shim variety count."""
        _row, _request, _result, source, spec = _load_cal_batch(store, batch_id)
        try:
            candidates, evaluated = calibration_engine.search(spec, source, body.limits)
        except CalibrationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return CalibrationSearchResponse(
            feasible=any(c.violation_count == 0 for c in candidates),
            combinations_evaluated=evaluated,
            candidates=candidates,
        )

    def _cal_plan_response(row) -> CalibrationPlanResponse:
        request = json.loads(row["request_json"])
        result = json.loads(row["result_json"])
        return CalibrationPlanResponse(
            id=row["id"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            batch_content_hash=request["batch_content_hash"],
            source_version_id=request["source"]["version_id"],
            source_content_hash=request["source"]["content_hash"],
            spec=request["spec"],
            limits=request["limits"],
            adjustment=result["adjustment"],
            fit=result["fit"],
            pins=result["pins"],
            diagnostics=result["diagnostics"],
            summary=result["summary"],
        )

    @app.post(
        "/api/calibration/batches/{batch_id}/confirm",
        response_model=CalibrationPlanResponse,
        status_code=201,
    )
    def confirm_calibration_batch(
        batch_id: int,
        body: CalibrationConfirmRequest,
        response: Response,
        store: Store = Depends(get_store),
    ) -> CalibrationPlanResponse:
        """Confirm a chosen adjustment: freezes an immutable calibration plan
        (source snapshot, measurements, fit parameters, selected adjustment
        and input hash) and marks the batch 已确认. Idempotent: the same
        adjustment under the same limits yields the same plan."""
        row, _request, _result, source, spec = _load_cal_batch(store, batch_id)
        try:
            content_hash, request_dict, result_dict = calibration_freeze.build_plan_payload(
                source, spec, body.limits, body.adjustment
            )
        except CalibrationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        request_dict["batch_content_hash"] = row["content_hash"]
        existing = store.get_cal_plan_by_hash(content_hash)
        if existing is not None:
            response.status_code = 200
            return _cal_plan_response(existing)
        plan_row = store.insert_cal_plan(
            content_hash,
            row["content_hash"],
            freeze.canonical_json(request_dict),
            json.dumps(result_dict, ensure_ascii=False, sort_keys=True),
        )
        return _cal_plan_response(plan_row)

    @app.get("/api/calibration/plans", response_model=list[CalibrationPlanSummary])
    def list_calibration_plans(
        store: Store = Depends(get_store),
    ) -> list[CalibrationPlanSummary]:
        out = []
        for row in store.list_cal_plans():
            request = json.loads(row["request_json"])
            result = json.loads(row["result_json"])
            out.append(
                CalibrationPlanSummary(
                    id=row["id"],
                    content_hash=row["content_hash"],
                    created_at=row["created_at"],
                    batch_content_hash=request["batch_content_hash"],
                    violation_count=result["summary"]["violation_count"],
                    min_margin_mm=result["summary"]["min_margin_mm"],
                )
            )
        return out

    @app.get("/api/calibration/plans/{plan_id}", response_model=CalibrationPlanResponse)
    def get_calibration_plan(
        plan_id: int, store: Store = Depends(get_store)
    ) -> CalibrationPlanResponse:
        row = store.get_cal_plan(plan_id)
        if row is None:
            raise HTTPException(status_code=404, detail="calibration plan not found")
        return _cal_plan_response(row)

    @app.post(
        "/api/calibration/plans/{plan_id}/recompute",
        response_model=CalibrationRecomputeResponse,
    )
    def recompute_calibration_plan(
        plan_id: int, store: Store = Depends(get_store)
    ) -> CalibrationRecomputeResponse:
        """Recompute a frozen plan from its stored source snapshot and
        measurements; the result must be bit-identical (same content hash)."""
        row = store.get_cal_plan(plan_id)
        if row is None:
            raise HTTPException(status_code=404, detail="calibration plan not found")
        request_dict = json.loads(row["request_json"])
        stored_result = json.loads(row["result_json"])
        new_hash, new_result = calibration_freeze.recompute_plan(request_dict)
        match = new_hash == row["content_hash"] and new_result == stored_result
        return CalibrationRecomputeResponse(
            id=row["id"],
            match=match,
            stored_hash=row["content_hash"],
            recomputed_hash=new_hash,
        )

    return app


app = create_app()
