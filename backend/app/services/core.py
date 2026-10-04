from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import random
from pathlib import Path

from decision_engine.fleet_availability import (
    AircraftAvailabilityInput,
    calculate_fleet_availability,
)
from decision_engine.maintenance_optimizer import (
    MaintenanceCandidate,
    build_maintenance_plan,
    plan_to_dict,
)
from decision_engine.spares import (
    SpareRequest,
    allocate_spares,
    inventory_summary,
)
from digital_twin.state import DigitalTwinStore
from ml.cmapss.runtime import CMapssModelRuntime, RAW_FEATURES
from ml.health import fuse_health_state

COMPONENTS = ("ENGINE", "HYDRAULIC", "ELECTRICAL", "LANDING_GEAR")
PARTS = {
    "ENGINE": "ENG-FLT",
    "HYDRAULIC": "HYD-PMP",
    "ELECTRICAL": "ELEC-REG",
    "LANDING_GEAR": "LG-ACT",
}
DURATIONS_HOURS = {
    "ENGINE": 8.0,
    "HYDRAULIC": 5.0,
    "ELECTRICAL": 4.0,
    "LANDING_GEAR": 6.0,
}
@dataclass
class Aircraft:
    id: str
    status: str
    components: dict[str, dict] = field(default_factory=dict)


class FleetService:
    def __init__(self, twin_store: DigitalTwinStore | None = None):
        self.rng = random.Random(26249)
        self.spares_data = [
            {"part_id": "ENG-FLT", "quantity": 9, "lead_time_days": 2},
            {"part_id": "HYD-PMP", "quantity": 4, "lead_time_days": 6},
            {"part_id": "ELEC-REG", "quantity": 6, "lead_time_days": 4},
            {"part_id": "LG-ACT", "quantity": 3, "lead_time_days": 8},
        ]
        self.aircraft: dict[str, Aircraft] = {}
        self.twin_store = twin_store or DigitalTwinStore()
        self.runtime = CMapssModelRuntime(
            os.getenv("FLEETAVAIL_MODEL_DIR", "models/cmapss"),
            rul_architecture=os.getenv("FLEETAVAIL_RUL_MODEL", "lstm").lower(),
        )
        self.latest_predictions: dict[tuple[str, str], dict] = {}
        self.replay_rows: dict[str, list[dict]] = {}
        self.replay_index: dict[str, int] = {}

        for i in range(1, 13):
            comps = {}
            for j, c in enumerate(COMPONENTS):
                h = max(
                    .25,
                    min(.98, .82 - .025 * ((i * (j + 2)) % 7)
                        - (.23 if c == "ENGINE" and i in {3, 7, 11} else 0)),
                )
                comps[c] = {
                    "health": h,
                    "rul": max(8.0, 180.0 * h - i * 2),
                    "risk": max(.01, min(.99, 1.0 - h + (.18 if h < .55 else 0))),
                    "anomaly": 1.0 - h,
                    "confidence": .83,
                    "data_quality": .98,
                    "model_mode": "synthetic_fallback",
                    "model_version": "HEURISTIC_V1",
                    "window_ready": False,
                }
            self.aircraft[f"AF-{i:03d}"] = Aircraft(
                f"AF-{i:03d}",
                ("READY", "READY", "READY", "DEGRADED", "MAINTENANCE")[(i - 1) % 5],
                comps,
            )

        self._load_cmapss_replay()

    def _load_cmapss_replay(self):
        """Use local C-MAPSS test trajectories as realistic demo telemetry when available."""
        root = Path(os.getenv("FLEETAVAIL_CMAPSS_ROOT", "data/raw/cmapss"))
        try:
            from ml.cmapss.data import load_cmapss
            _, test = load_cmapss(root, "FD001")
            units = sorted(test.unit_id.unique())
            for idx, aid in enumerate(self.aircraft):
                if idx >= len(units):
                    break
                unit = units[idx]
                rows = test[test.unit_id == unit].sort_values("cycle")
                self.replay_rows[aid] = rows.to_dict("records")
                self.replay_index[aid] = 0
                self.runtime.seed_history(
                    aid,
                    "ENGINE",
                    self.replay_rows[aid][-self.runtime.window_size:],
                )
                prediction = self.runtime.predict(
                    aid,
                    "ENGINE",
                    self.replay_rows[aid][-1],
                    cycle=int(self.replay_rows[aid][-1]["cycle"]),
                )
                self._apply_prediction(
                    aid,
                    "ENGINE",
                    prediction,
                    cycle=int(self.replay_rows[aid][-1]["cycle"]),
                )
        except Exception:
            self.replay_rows = {}
            self.replay_index = {}

    def _synthetic_telemetry(self, aid: str, component: str) -> dict:
        state = self.aircraft[aid].components[component]
        degradation = 1.0 - state["health"]
        cycle = int(state.get("_cycle", 0)) + 1
        state["_cycle"] = cycle
        row = {
            "op_setting_1": self.rng.gauss(0, .01),
            "op_setting_2": self.rng.gauss(0, .01),
            "op_setting_3": self.rng.gauss(0, .01),
        }
        for i in range(1, 22):
            baseline = 0.0
            if i in {2, 3, 4}:
                baseline = {2: 642.0, 3: 1580.0, 4: 1400.0}[i]
            elif i in {11, 12}:
                baseline = {11: 47.0, 12: 521.0}[i]
            value = baseline * (1 + degradation * .05) if baseline else degradation
            row[f"sensor_{i}"] = value + self.rng.gauss(0, max(abs(value) * .01, .01))
        row["egt_c"] = row["sensor_2"]
        row["vibration_g"] = row["sensor_3"] / 1000.0
        row["oil_pressure_kpa"] = row["sensor_4"] / 3.4
        return row

    def _apply_prediction(
        self,
        aid: str,
        component: str,
        prediction: dict,
        cycle: int | None = None,
    ):
        x = self.aircraft[aid].components[component]
        if prediction.get("rul_cycles") is None:
            return
        risk = prediction["failure_probability"]
        anomaly = prediction["anomaly_score"]
        x.update({
            "health": max(0.0, min(1.0, 1.0 - (0.55 * risk + 0.45 * anomaly))),
            "rul": prediction["rul_cycles"],
            "risk": prediction["failure_probability"],
            "anomaly": prediction["anomaly_score"],
            "confidence": prediction["confidence"],
            "data_quality": prediction["data_quality"],
            "model_mode": prediction["model_mode"],
            "model_version": prediction["model_version"],
            "window_ready": prediction["window_ready"],
        })
        twin = self.twin_store.get_or_create(aid)
        existing = twin.components.get(component)
        resolved_cycle = int(
            cycle if cycle is not None
            else (existing.last_update_cycle + 1 if existing else 1)
        )
        self._sync_twin(aid, component, prediction, resolved_cycle)

    def _sync_twin(
        self,
        aircraft_id: str,
        component: str,
        prediction: dict,
        cycle: int,
    ) -> None:
        self.twin_store.apply_prediction(
            aircraft_id,
            component,
            health_score=prediction["health_score"],
            rul_cycles=prediction["rul_cycles"],
            failure_probability=prediction["failure_probability"],
            anomaly_score=prediction["anomaly_score"],
            confidence=prediction["confidence"],
            data_quality=prediction["data_quality"],
            cycle=cycle,
        )

    def health(self):
        return {
            "status": "healthy",
            "service": "FleetAvail",
            "aircraft_count": len(self.aircraft),
            "mode": self.runtime.mode,
            "models": self.runtime.model_status,
            "decision_layers": [
                "health_fusion",
                "digital_twin",
                "maintenance_optimizer",
                "spare_allocation",
                "fleet_availability",
            ],
        }

    def fused(self, aircraft: Aircraft, component: str) -> dict:
        x = aircraft.components[component]
        state = fuse_health_state(
            health=x["health"],
            anomaly=x["anomaly"],
            data_quality=x["data_quality"],
            risk=x["risk"],
            rul=x["rul"],
            confidence=x["confidence"],
            aircraft_id=aircraft.id,
            component=component,
        )
        return state.to_dict()

    def _sync_twin(
        self,
        aircraft_id: str,
        component: str,
        prediction: dict,
        cycle: int,
    ) -> None:
        self.twin_store.apply_prediction(
            aircraft_id,
            component,
            health_score=prediction["health_score"],
            rul_cycles=prediction["rul_cycles"],
            failure_probability=prediction["failure_probability"],
            anomaly_score=prediction["anomaly_score"],
            confidence=prediction["confidence"],
            data_quality=prediction["data_quality"],
            cycle=cycle,
        )

    def fleet_summary(self):
        total = len(self.aircraft)
        ready = sum(a.status == "READY" for a in self.aircraft.values())
        degraded = sum(a.status == "DEGRADED" for a in self.aircraft.values())
        maint = sum(a.status == "MAINTENANCE" for a in self.aircraft.values())
        critical = sum(
            self.fused(a, "ENGINE")["health_level"] == "CRITICAL"
            for a in self.aircraft.values()
        )
        availability = self.fleet_availability()
        return {
            "total_aircraft": total,
            "ready": ready,
            "degraded": degraded,
            "maintenance": maint,
            "critical_aircraft": critical,
            "current_availability_pct": availability["current_availability_pct"],
            "projected_7_day_availability_pct": availability["projected_availability_pct"],
        }

    def aircraft_list(self):
        return [
            {
                "aircraft_id": aircraft.id,
                "status": aircraft.status,
                "engine": self.fused(aircraft, "ENGINE"),
            }
            for aircraft in self.aircraft.values()
        ]

    def aircraft_detail(self, aircraft_id):
        aircraft = self.aircraft[aircraft_id]
        return {
            "aircraft_id": aircraft_id,
            "status": aircraft.status,
            "components": {
                component: self.fused(aircraft, component)
                for component in COMPONENTS
            },
            "twin_state": self.twin_store.snapshot(aircraft_id),
        }

    def predict(self, aid, component, telemetry):
        if aid not in self.aircraft:
            raise KeyError(aid)
        if component not in COMPONENTS:
            raise ValueError(f"Unsupported component: {component}")

        if component == "ENGINE":
            twin = self.twin_store.get_or_create(aid)
            existing = twin.components.get(component)
            default_cycle = existing.last_update_cycle + 1 if existing else 1
            cycle = int((telemetry or {}).get("cycle", default_cycle))
            prediction = self.runtime.predict(
                aid,
                component,
                telemetry or {},
                cycle=cycle,
            )
            self.latest_predictions[(aid, component)] = prediction
            self._apply_prediction(aid, component, prediction, cycle=cycle)
            return {
                **self.fused(self.aircraft[aid], component),
                "inference": prediction,
            }

        x = self.aircraft[aid].components[component]
        if telemetry:
            x["health"] = max(.05, min(.99, x["health"] - .02))
            x["anomaly"] = 1 - x["health"]
            x["risk"] = max(.01, min(.99, 1 - x["health"]))
        prediction = self.fused(self.aircraft[aid], component)
        twin = self.twin_store.get_or_create(aid)
        existing = twin.components.get(component)
        default_cycle = existing.last_update_cycle + 1 if existing else 1
        cycle = int((telemetry or {}).get("cycle", default_cycle))
        self._sync_twin(
            aid,
            component,
            {
                "health_score": prediction["health_score"],
                "rul_cycles": prediction["rul_cycles"],
                "failure_probability": prediction["failure_probability"],
                "anomaly_score": prediction["anomaly_score"],
                "confidence": prediction["confidence"],
                "data_quality": prediction["data_quality"],
            },
            cycle=cycle,
        )
        return prediction

    def what_if(self, aircraft_id, component, percentage):
        baseline = self.fused(self.aircraft[aircraft_id], component)
        factor = max(0.0, 1.0 - percentage / 100.0)
        health = max(0.0, baseline["health_score"] * factor)
        rul = max(0.0, baseline["rul_cycles"] * factor)
        risk = min(0.99, baseline["failure_probability"] + max(0.0, percentage) / 130.0)
        scenario = fuse_health_state(
            health=health / 100.0,
            anomaly=min(1.0, baseline["anomaly_score"] + max(0.0, percentage) / 100.0),
            data_quality=baseline["data_quality"],
            risk=risk,
            rul=rul,
            confidence=baseline["confidence"],
            aircraft_id=aircraft_id,
            component=component,
        ).to_dict()
        readiness = self.fleet_availability()
        return {
            "baseline": baseline,
            "scenario": {
                "degradation_pct": percentage,
                **scenario,
                "projected_fleet_availability_pct": max(
                    0.0,
                    readiness["projected_availability_pct"] - max(0.0, percentage) * 0.12,
                ),
            },
        }

    def _maintenance_candidates(self, mission_priority: float = 1.0):
        candidates = []
        inventory = {item["part_id"]: item["quantity"] for item in self.spares_data}
        for aircraft in self.aircraft.values():
            for component in COMPONENTS:
                p = self.fused(aircraft, component)
                candidates.append(
                    MaintenanceCandidate(
                        aircraft_id=aircraft.id,
                        component=component,
                        mission_priority=mission_priority,
                        failure_probability=p["failure_probability"],
                        rul_cycles=p["rul_cycles"],
                        maintenance_duration_hours=DURATIONS_HOURS[component],
                        spare_part_id=PARTS[component],
                        spare_available=inventory.get(PARTS[component], 0),
                    )
                )
        return candidates

    def maintenance_plan(
        self,
        mission_priority: float = 1.0,
        horizon_days: int = 7,
        max_daily_hours: float = 24.0,
    ):
        plan = build_maintenance_plan(
            self._maintenance_candidates(mission_priority),
            horizon_days=horizon_days,
            max_maintenance_hours_per_day=max_daily_hours,
        )
        return {
            "horizon_days": horizon_days,
            "max_daily_hours": max_daily_hours,
            "items": plan_to_dict(plan),
        }

    def maintenance_recommendation(self, aircraft_id, mission_priority):
        plan = self.maintenance_plan(mission_priority=mission_priority)
        rows = [x for x in plan["items"] if x["aircraft_id"] == aircraft_id]
        rows.sort(key=lambda item: (-item["priority_score"], item["component"]))
        chosen = rows[0]
        prediction = self.fused(self.aircraft[aircraft_id], chosen["component"])
        part = next(x for x in self.spares_data if x["part_id"] == chosen["spare_part_id"])
        return {
            "aircraft_id": aircraft_id,
            "priority_component": chosen["component"],
            "recommended_action": chosen["action"],
            "scheduled_day": chosen["scheduled_day"],
            "priority_score": chosen["priority_score"],
            "rul_cycles": prediction["rul_cycles"],
            "failure_probability": prediction["failure_probability"],
            "health_level": prediction["health_level"],
            "spare": part,
            "reason_codes": chosen["reason_codes"],
        }

    def allocate_spares_for_plan(
        self,
        mission_priority: float = 1.0,
        horizon_days: int = 7,
        max_daily_hours: float = 24.0,
    ):
        plan = self.maintenance_plan(
            mission_priority=mission_priority,
            horizon_days=horizon_days,
            max_daily_hours=max_daily_hours,
        )
        inventory = {item["part_id"]: item["quantity"] for item in self.spares_data}
        requests = []
        for item in plan["items"]:
            if item["action"] == "MONITOR":
                continue
            prediction = self.fused(self.aircraft[item["aircraft_id"]], item["component"])
            requests.append(
                SpareRequest(
                    aircraft_id=item["aircraft_id"],
                    component=item["component"],
                    part_id=item["spare_part_id"],
                    quantity=1,
                    priority_score=item["priority_score"],
                    delay_cost=(
                        prediction["failure_probability"] * 10.0
                        + 1.0 / (1.0 + prediction["rul_cycles"]) * 20.0
                    ),
                    compatible=True,
                )
            )

        allocations = allocate_spares(requests, inventory)
        summary = inventory_summary(inventory, allocations)
        return {
            "requests": [
                {
                    "aircraft_id": request.aircraft_id,
                    "component": request.component,
                    "part_id": request.part_id,
                    "quantity": request.quantity,
                    "priority_score": request.priority_score,
                    "delay_cost": round(request.delay_cost, 4),
                }
                for request in requests
            ],
            "allocations": [
                {
                    "aircraft_id": item.aircraft_id,
                    "component": item.component,
                    "part_id": item.part_id,
                    "allocated_quantity": item.allocated_quantity,
                    "unmet_quantity": item.unmet_quantity,
                    "allocation_rank": item.allocation_rank,
                    "reason": item.reason,
                }
                for item in allocations
            ],
            "inventory": summary,
        }

    def fleet_availability(
        self,
        mission_priority: float = 1.0,
        horizon_days: int = 7,
        max_daily_hours: float = 24.0,
    ):
        plan = self.maintenance_plan(
            mission_priority=mission_priority,
            horizon_days=horizon_days,
            max_daily_hours=max_daily_hours,
        )
        allocation = self.allocate_spares_for_plan(
            mission_priority=mission_priority,
            horizon_days=horizon_days,
            max_daily_hours=max_daily_hours,
        )
        allocated_keys = {
            (item["aircraft_id"], item["component"])
            for item in allocation["allocations"]
            if item["allocated_quantity"] > 0
            and item["unmet_quantity"] == 0
        }

        inputs = []
        for aircraft in self.aircraft.values():
            plan_rows = [
                row
                for row in plan["items"]
                if row["aircraft_id"] == aircraft.id
                and row["action"] not in {"MONITOR", "ORDER_SPARE_AND_HOLD"}
            ]
            plan_rows.sort(key=lambda row: (row["scheduled_day"], -row["priority_score"]))
            chosen = plan_rows[0] if plan_rows else None
            spare_ok = bool(
                chosen
                and (chosen["aircraft_id"], chosen["component"]) in allocated_keys
            )
            engine = self.fused(aircraft, "ENGINE")
            inputs.append(
                AircraftAvailabilityInput(
                    aircraft_id=aircraft.id,
                    current_status=aircraft.status,
                    maintenance_day=chosen["scheduled_day"] if chosen else None,
                    maintenance_duration_hours=chosen["duration_hours"] if chosen else 0.0,
                    spare_available=spare_ok,
                    critical=engine["health_level"] == "CRITICAL",
                )
            )

        result = calculate_fleet_availability(inputs, horizon_days=horizon_days)
        result["maintenance_plan_items"] = len(plan["items"])
        result["spare_unmet_requests"] = allocation["inventory"]["total_unmet"]
        return result

    def record_maintenance(self, aircraft_id, component, action, cycle):
        self.aircraft[aircraft_id].status = "READY"
        state = self.twin_store.record_maintenance(
            aircraft_id,
            component,
            action=action,
            cycle=cycle,
        )
        return {
            "aircraft_id": aircraft_id,
            "component": component,
            "action": action,
            "cycle": cycle,
            "component_state": {
                "health": state.health,
                "rul_cycles": state.rul_cycles,
                "failure_probability": state.failure_probability,
                "anomaly_score": state.anomaly_score,
                "lifecycle_status": state.lifecycle_status,
            },
        }

    def twin_snapshot(self, aircraft_id):
        return self.twin_store.snapshot(aircraft_id)

    def spares(self):
        return self.spares_data

    def next_telemetry_event(self):
        aircraft = self.rng.choice(list(self.aircraft.values()))
        prediction = self.fused(aircraft, "ENGINE")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "telemetry_update",
            "aircraft_id": aircraft.id,
            "component": "ENGINE",
            "health_score": prediction["health_score"],
            "rul_cycles": prediction["rul_cycles"],
            "failure_probability": prediction["failure_probability"],
            "alert_level": prediction["alert_level"],
            "health_level": prediction["health_level"],
        }
