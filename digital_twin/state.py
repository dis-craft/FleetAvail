"""Digital-twin state and persistence for FleetAvail.

The default store is a small atomic JSON state file so the repository can run
without external services. The store API is backend-neutral: PostgreSQL/Redis
can replace this persistence implementation later without changing decision code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import tempfile


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    value = float(value)
    if not value == value or value in (float("inf"), float("-inf")):
        raise ValueError("numeric twin state must be finite")
    return min(high, max(low, value))


@dataclass
class ComponentState:
    name: str
    health: float = 1.0
    degradation: float = 0.0
    rul_cycles: float = 100.0
    failure_probability: float = 0.0
    anomaly_score: float = 0.0
    confidence: float = 1.0
    data_quality: float = 1.0
    lifecycle_status: str = "IN_SERVICE"
    last_update_cycle: int = 0
    last_updated_at: str | None = None


@dataclass
class TwinEvent:
    event_type: str
    aircraft_id: str
    component: str | None
    cycle: int
    payload: dict[str, Any]
    timestamp: str


@dataclass
class AircraftTwin:
    aircraft_id: str
    mission_status: str = "READY"
    location: str = "BASE"
    components: dict[str, ComponentState] = field(default_factory=dict)
    events: list[TwinEvent] = field(default_factory=list)
    maintenance_due: bool = False

    def apply_prediction(
        self,
        component: str,
        *,
        health_score: float,
        rul_cycles: float,
        failure_probability: float,
        anomaly_score: float,
        confidence: float,
        data_quality: float,
        cycle: int,
    ) -> ComponentState:
        if not component:
            raise ValueError("component is required")
        if int(cycle) < 0:
            raise ValueError("cycle cannot be negative")

        state = self.components.setdefault(component, ComponentState(name=component))
        if int(cycle) < state.last_update_cycle:
            raise ValueError("telemetry cycle cannot move backwards")

        state.health = _bounded(float(health_score) / 100.0)
        state.degradation = round(1.0 - state.health, 6)
        state.rul_cycles = max(0.0, float(rul_cycles))
        state.failure_probability = _bounded(failure_probability)
        state.anomaly_score = _bounded(anomaly_score)
        state.confidence = _bounded(confidence)
        state.data_quality = _bounded(data_quality)
        state.last_update_cycle = int(cycle)
        state.last_updated_at = _now()

        state.lifecycle_status = (
            "CRITICAL"
            if state.failure_probability >= 0.85 or state.rul_cycles <= 20
            else "DEGRADED"
            if state.failure_probability >= 0.60 or state.rul_cycles <= 45
            else "WATCH"
            if state.failure_probability >= 0.30 or state.rul_cycles <= 90
            else "IN_SERVICE"
        )
        self.maintenance_due = any(
            item.lifecycle_status in {"CRITICAL", "DEGRADED"}
            for item in self.components.values()
        )
        self.events.append(
            TwinEvent(
                event_type="PREDICTION_UPDATE",
                aircraft_id=self.aircraft_id,
                component=component,
                cycle=int(cycle),
                payload={
                    "health_score": float(health_score),
                    "rul_cycles": float(rul_cycles),
                    "failure_probability": float(failure_probability),
                    "anomaly_score": float(anomaly_score),
                    "confidence": float(confidence),
                    "data_quality": float(data_quality),
                },
                timestamp=_now(),
            )
        )
        self.events = self.events[-100:]
        return state

    def record_maintenance(
        self,
        component: str,
        *,
        action: str,
        cycle: int,
        reset_health: float = 0.98,
        reset_rul: float = 180.0,
    ) -> ComponentState:
        state = self.components.setdefault(component, ComponentState(name=component))
        state.health = _bounded(reset_health)
        state.degradation = round(1.0 - state.health, 6)
        state.rul_cycles = max(0.0, float(reset_rul))
        state.failure_probability = 0.02
        state.anomaly_score = 0.05
        state.confidence = 0.90
        state.data_quality = 1.0
        state.lifecycle_status = "IN_SERVICE"
        state.last_update_cycle = max(state.last_update_cycle, int(cycle))
        state.last_updated_at = _now()
        self.maintenance_due = any(
            item.lifecycle_status in {"CRITICAL", "DEGRADED"}
            for item in self.components.values()
        )
        self.events.append(
            TwinEvent(
                event_type="MAINTENANCE",
                aircraft_id=self.aircraft_id,
                component=component,
                cycle=int(cycle),
                payload={
                    "action": action,
                    "reset_health": reset_health,
                    "reset_rul": reset_rul,
                },
                timestamp=_now(),
            )
        )
        self.events = self.events[-100:]
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "aircraft_id": self.aircraft_id,
            "mission_status": self.mission_status,
            "location": self.location,
            "components": {
                name: asdict(state) for name, state in self.components.items()
            },
            "events": [asdict(event) for event in self.events],
            "maintenance_due": self.maintenance_due,
        }


class DigitalTwinStore:
    """Persistent aircraft/component twin registry with atomic JSON writes."""

    def __init__(self, path: str | Path = "data/digital_twin_state.json"):
        self.path = Path(path)
        self.twins: dict[str, AircraftTwin] = {}
        self.load()

    def get_or_create(self, aircraft_id: str) -> AircraftTwin:
        if not aircraft_id or not aircraft_id.strip():
            raise ValueError("aircraft_id is required")
        return self.twins.setdefault(
            aircraft_id, AircraftTwin(aircraft_id=aircraft_id)
        )

    def upsert(self, twin: AircraftTwin) -> AircraftTwin:
        self.twins[twin.aircraft_id] = twin
        self.save()
        return twin

    def apply_prediction(self, aircraft_id: str, component: str, **kwargs: Any) -> ComponentState:
        twin = self.get_or_create(aircraft_id)
        state = twin.apply_prediction(component, **kwargs)
        self.save()
        return state

    def record_maintenance(self, aircraft_id: str, component: str, **kwargs: Any) -> ComponentState:
        twin = self.get_or_create(aircraft_id)
        state = twin.record_maintenance(component, **kwargs)
        self.save()
        return state

    def snapshot(self, aircraft_id: str) -> dict[str, Any]:
        return self.get_or_create(aircraft_id).to_dict()

    def fleet_snapshot(self) -> list[dict[str, Any]]:
        return [twin.to_dict() for twin in self.twins.values()]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "twins": self.fleet_snapshot()}
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name, dir=self.path.parent, text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.twins = {}
        for item in raw.get("twins", []):
            twin = AircraftTwin(
                aircraft_id=item["aircraft_id"],
                mission_status=item.get("mission_status", "READY"),
                location=item.get("location", "BASE"),
                maintenance_due=bool(item.get("maintenance_due", False)),
            )
            for name, state in item.get("components", {}).items():
                values = {k: v for k, v in state.items() if k != "name"}
                twin.components[name] = ComponentState(name=name, **values)
            for event in item.get("events", []):
                twin.events.append(TwinEvent(**event))
            self.twins[twin.aircraft_id] = twin
