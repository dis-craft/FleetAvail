from dataclasses import dataclass, field
from datetime import datetime, timezone
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

        for i in range(1, 13):
            comps = {}
            for j, component in enumerate(COMPONENTS):
                health = max(
                    0.25,
                    min(
                        0.98,
                        0.82 - 0.025 * ((i * (j + 2)) % 7)
                        - (0.23 if component == "ENGINE" and i in {3, 7, 11} else 0),
                    ),
                )
                comps[component] = {
                    "health": health,
                    "rul": max(8.0, 180.0 * health - i * 2),
                    "risk": max(
                        0.01,
                        min(0.99, 1.0 - health + (0.18 if health < 0.55 else 0)),
                    ),
                    "anomaly": 1.0 - health,
                    "confidence": 0.83,
                    "data_quality": 0.98,
                }
            self.aircraft[f"AF-{i:03d}"] = Aircraft(
                f"AF-{i:03d}",
                ("READY", "READY", "READY", "DEGRADED", "MAINTENANCE")[(i - 1) % 5],
                comps,
            )

    def health(self):
        return {
            "status": "healthy",
            "service": "FleetAvail",
            "aircraft_count": len(self.aircraft),
            "mode": "synthetic_demo",
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

    def predict(self, aircraft_id, component, telemetry):
        aircraft = self.aircraft[aircraft_id]
        x = aircraft.components[component]

        if telemetry:
            penalty = 0.0
            if component == "ENGINE":
                penalty = (
                    max(0.0, (telemetry.get("egt_c", 650.0) - 720.0) / 600.0)
                    + max(0.0, (telemetry.get("vibration_g", 0.12) - 0.5) / 1.5)
                    + max(0.0, (330.0 - telemetry.get("oil_pressure_kpa", 410.0)) / 500.0)
                )
            x["health"] = max(0.05, min(0.99, x["health"] - min(0.12, penalty * 0.05)))
            x["anomaly"] = max(0.0, min(1.0, 1.0 - x["health"]))
            x["risk"] = max(0.01, min(0.99, 1.0 - x["health"]))
            x["rul"] = max(3.0, x["rul"] * (0.96 if penalty else 1.005))

        prediction = self.fused(aircraft, component)
        existing_twin = self.twin_store.get_or_create(aircraft_id)
        existing_component = existing_twin.components.get(component)
        default_cycle = (existing_component.last_update_cycle + 1) if existing_component else 1
        cycle = int(telemetry.get("cycle", default_cycle)) if telemetry else default_cycle
        self._sync_twin(aircraft_id, component, prediction, cycle)
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
