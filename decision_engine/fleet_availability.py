"""Fleet availability calculations and decision what-if planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AircraftAvailabilityInput:
    aircraft_id: str
    current_status: str
    maintenance_day: int | None = None
    maintenance_duration_hours: float = 0.0
    spare_available: bool = True
    critical: bool = False


def calculate_fleet_availability(
    aircraft: Iterable[AircraftAvailabilityInput],
    horizon_days: int = 7,
) -> dict:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    items = list(aircraft)
    total = len(items)
    if total == 0:
        return {
            "total_aircraft": 0,
            "current_available": 0,
            "current_availability_pct": 0.0,
            "projected_available": 0,
            "projected_availability_pct": 0.0,
            "horizon_days": horizon_days,
            "blocked_aircraft": [],
            "recovered_aircraft": [],
        }

    current_available = sum(
        item.current_status == "READY" and not item.critical for item in items
    )
    blocked = {
        item.aircraft_id
        for item in items
        if item.current_status != "READY" or item.critical
    }
    recovered: set[str] = set()

    for item in items:
        can_recover = (
            item.spare_available
            and item.maintenance_day is not None
            and 0 <= item.maintenance_day < horizon_days
            and 0 < item.maintenance_duration_hours <= 24
        )
        if item.aircraft_id in blocked and can_recover:
            recovered.add(item.aircraft_id)

    projected_blocked = blocked - recovered
    projected_available = min(total, current_available + len(recovered))

    return {
        "total_aircraft": total,
        "current_available": current_available,
        "current_availability_pct": round(100.0 * current_available / total, 1),
        "projected_available": projected_available,
        "projected_availability_pct": round(100.0 * projected_available / total, 1),
        "horizon_days": horizon_days,
        "blocked_aircraft": sorted(projected_blocked),
        "recovered_aircraft": sorted(recovered),
    }


def project_after_decision(
    current_available: int,
    total_aircraft: int,
    aircraft_recovered: int,
    aircraft_blocked: int = 0,
) -> dict[str, float]:
    if total_aircraft <= 0:
        raise ValueError("total_aircraft must be positive")
    available = max(
        0,
        min(total_aircraft, current_available + aircraft_recovered - aircraft_blocked),
    )
    return {
        "available_aircraft": float(available),
        "availability_pct": round(100.0 * available / total_aircraft, 1),
    }
