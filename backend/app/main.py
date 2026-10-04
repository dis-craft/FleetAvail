from pathlib import Path
import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.app.services.core import FleetService

ROOT = Path(__file__).resolve().parents[2]
service = FleetService()

app = FastAPI(
    title="FleetAvail",
    version="0.1.0",
    description="SIH26249 aircraft predictive maintenance and fleet availability prototype",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Prediction(BaseModel):
    aircraft_id: str
    component: str = "ENGINE"
    telemetry: dict[str, float] = Field(default_factory=dict)


class WhatIf(BaseModel):
    aircraft_id: str
    component: str = "ENGINE"
    degradation_pct: float = Field(10, ge=-50, le=200)


class Maintenance(BaseModel):
    aircraft_id: str
    mission_priority: float = Field(1, ge=0.1, le=2)


class PlanningOptions(BaseModel):
    mission_priority: float = Field(1, ge=0.1, le=2)
    horizon_days: int = Field(7, ge=1, le=30)
    max_daily_hours: float = Field(24, gt=0, le=168)


@app.get("/")
def root():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/health")
def health():
    return service.health()


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
    return service.aircraft_detail(aircraft_id)


@app.get("/api/fleet/aircraft/{aircraft_id}/twin")
def twin(aircraft_id: str):
    return service.twin_snapshot(aircraft_id)


@app.post("/api/predict")
def predict(x: Prediction):
    return service.predict(x.aircraft_id, x.component, x.telemetry)


@app.post("/api/simulate/what-if")
def whatif(x: WhatIf):
    return service.what_if(x.aircraft_id, x.component, x.degradation_pct)


@app.post("/api/maintenance/recommend")
def recommend(x: Maintenance):
    return service.maintenance_recommendation(x.aircraft_id, x.mission_priority)


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
    return service.allocate_spares_for_plan(x.mission_priority)


@app.post("/api/fleet/availability")
def projected_availability(x: PlanningOptions):
    return service.fleet_availability(
        mission_priority=x.mission_priority,
        horizon_days=x.horizon_days,
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
