from pathlib import Path
import asyncio
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.app.services.core import FleetService

ROOT = Path(__file__).resolve().parents[2]
service = FleetService()

app = FastAPI(
    title="FleetAvail",
    version="0.2.0",
    description="Aircraft predictive maintenance and fleet availability decision API",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    aircraft_id: str
    component: str = "ENGINE"
    telemetry: dict[str, float] = Field(default_factory=dict)


class WhatIf(BaseModel):
    aircraft_id: str
    component: str = "ENGINE"
    degradation_pct: float = Field(10, ge=-50, le=200)


class Maintenance(BaseModel):
    aircraft_id: str
    mission_priority: float = Field(1, ge=.1, le=2)


class MaintenanceExecution(BaseModel):
    component: str = "ENGINE"
    action: str = "REPLACE_COMPONENT"
    cycle: int = Field(..., ge=0)


class PlanningOptions(BaseModel):
    mission_priority: float = Field(1, ge=.1, le=2)
    horizon_days: int = Field(7, ge=1, le=30)
    max_daily_hours: float = Field(24, gt=0, le=168)


@app.get("/")
def root():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/health")
def health():
    return service.health()


@app.get("/api/models")
def models():
    return service.runtime.model_status


@app.get("/api/fleet/summary")
def summary():
    return service.fleet_summary()


@app.get("/api/fleet/availability")
def fleet_availability():
    return service.fleet_availability()


@app.get("/api/fleet/aircraft")
def aircraft():
    return service.aircraft_list()


@app.get("/api/fleet/aircraft/{aircraft_id}")
def detail(aircraft_id: str):
    try:
        return service.aircraft_detail(aircraft_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Aircraft not found")


@app.get("/api/fleet/aircraft/{aircraft_id}/twin")
def twin(aircraft_id: str):
    try:
        return service.twin_snapshot(aircraft_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Aircraft not found")


@app.post("/api/fleet/aircraft/{aircraft_id}/maintenance")
def execute_maintenance(aircraft_id: str, x: MaintenanceExecution):
    try:
        return service.record_maintenance(
            aircraft_id,
            x.component.upper(),
            x.action,
            x.cycle,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Aircraft not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/predict")
def predict(x: PredictionRequest):
    try:
        return service.predict(x.aircraft_id, x.component.upper(), x.telemetry)
    except KeyError:
        raise HTTPException(status_code=404, detail="Aircraft not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/simulate/what-if")
def whatif(x: WhatIf):
    try:
        return service.what_if(
            x.aircraft_id,
            x.component.upper(),
            x.degradation_pct,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Aircraft not found")


@app.post("/api/maintenance/recommend")
def recommend(x: Maintenance):
    try:
        return service.maintenance_recommendation(
            x.aircraft_id,
            x.mission_priority,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Aircraft not found")


@app.post("/api/maintenance/plan")
def maintenance_plan(x: PlanningOptions):
    return service.maintenance_plan(
        mission_priority=x.mission_priority,
        horizon_days=x.horizon_days,
        max_daily_hours=x.max_daily_hours,
    )


@app.get("/api/spares")
def spares():
    return service.spares()


@app.post("/api/spares/allocate")
def allocate_spares(x: PlanningOptions):
    return service.allocate_spares_for_plan(
        mission_priority=x.mission_priority,
        horizon_days=x.horizon_days,
        max_daily_hours=x.max_daily_hours,
    )


@app.post("/api/fleet/availability")
def projected_availability(x: PlanningOptions):
    return service.fleet_availability(
        mission_priority=x.mission_priority,
        horizon_days=x.horizon_days,
        max_daily_hours=x.max_daily_hours,
    )


@app.websocket("/ws/telemetry")
async def ws(socket: WebSocket):
    await socket.accept()
    try:
        while True:
            await socket.send_text(json.dumps(service.next_telemetry_event()))
            await asyncio.sleep(1)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
