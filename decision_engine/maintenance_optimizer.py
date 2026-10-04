"""Constraint-aware maintenance planning for FleetAvail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MaintenanceCandidate:
    aircraft_id: str
    component: str
    mission_priority: float
    failure_probability: float
    rul_cycles: float
    maintenance_duration_hours: float
    spare_part_id: str
    spare_available: int


@dataclass(frozen=True)
class MaintenancePlanItem:
    aircraft_id: str
    component: str
    action: str
    priority_score: float
    scheduled_day: int
    duration_hours: float
    spare_part_id: str
    reason_codes: tuple[str, ...]


def maintenance_score(
    failure_probability: float,
    rul_cycles: float,
    mission_priority: float,
    *,
    spare_available: bool = True,
) -> float:
    risk = max(0.0, min(1.0, float(failure_probability)))
    urgency = 1.0 / (1.0 + max(0.0, float(rul_cycles)))
    mission = max(0.1, min(2.0, float(mission_priority)))
    spare_bonus = 0.10 if spare_available else -0.05
    return round(0.65 * risk + 8.0 * urgency + 0.30 * (2.0 - mission) + spare_bonus, 4)


def build_maintenance_plan(
    candidates: Iterable[MaintenanceCandidate],
    *,
    horizon_days: int = 7,
    max_maintenance_hours_per_day: float = 24.0,
) -> list[MaintenancePlanItem]:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if max_maintenance_hours_per_day <= 0:
        raise ValueError("max_maintenance_hours_per_day must be positive")

    ranked = []
    for candidate in candidates:
        if candidate.maintenance_duration_hours <= 0:
            raise ValueError("maintenance_duration_hours must be positive")
        spare_ok = candidate.spare_available > 0
        score = maintenance_score(
            candidate.failure_probability,
            candidate.rul_cycles,
            candidate.mission_priority,
            spare_available=spare_ok,
        )
        ranked.append((score, candidate))

    ranked.sort(key=lambda item: (-item[0], item[1].aircraft_id, item[1].component))
    daily_load = [0.0] * horizon_days
    plan: list[MaintenancePlanItem] = []

    for score, candidate in ranked:
        urgent = candidate.failure_probability >= 0.85 or candidate.rul_cycles <= 20
        preferred = candidate.failure_probability >= 0.60 or candidate.rul_cycles <= 45
        max_day = min(1 if urgent else 3 if preferred else horizon_days, horizon_days) - 1
        chosen_day = horizon_days - 1

        for day in range(max_day + 1):
            if daily_load[day] + candidate.maintenance_duration_hours <= max_maintenance_hours_per_day:
                chosen_day = day
                break

        if candidate.spare_available <= 0:
            action = "ORDER_SPARE_AND_HOLD"
            reasons = ("SPARE_SHORTAGE", "FLEET_IMPACT", "RISK_OR_URGENCY")
        elif urgent:
            action = "GROUND_AND_MAINTAIN"
            reasons = ("CRITICAL_RISK", "LOW_RUL", "FLEET_IMPACT")
        elif preferred:
            action = "SCHEDULE_MAINTENANCE"
            reasons = ("ELEVATED_RISK", "RUL_URGENCY")
        else:
            action = "MONITOR"
            reasons = ("NO_IMMEDIATE_TRIGGER",)

        daily_load[chosen_day] += candidate.maintenance_duration_hours
        plan.append(
            MaintenancePlanItem(
                aircraft_id=candidate.aircraft_id,
                component=candidate.component,
                action=action,
                priority_score=score,
                scheduled_day=chosen_day,
                duration_hours=candidate.maintenance_duration_hours,
                spare_part_id=candidate.spare_part_id,
                reason_codes=reasons,
            )
        )

    return plan


def plan_to_dict(items: Iterable[MaintenancePlanItem]) -> list[dict]:
    return [
        {
            "aircraft_id": item.aircraft_id,
            "component": item.component,
            "action": item.action,
            "priority_score": item.priority_score,
            "scheduled_day": item.scheduled_day,
            "duration_hours": item.duration_hours,
            "spare_part_id": item.spare_part_id,
            "reason_codes": list(item.reason_codes),
        }
        for item in items
    ]
