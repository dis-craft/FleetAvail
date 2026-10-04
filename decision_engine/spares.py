"""Constrained spare allocation for a fleet maintenance plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SpareRequest:
    aircraft_id: str
    component: str
    part_id: str
    quantity: int
    priority_score: float
    delay_cost: float
    compatible: bool = True


@dataclass(frozen=True)
class SpareAllocation:
    aircraft_id: str
    component: str
    part_id: str
    allocated_quantity: int
    unmet_quantity: int
    allocation_rank: int
    reason: str


def allocate_spares(
    requests: Iterable[SpareRequest],
    inventory: dict[str, int],
) -> list[SpareAllocation]:
    stock = {str(k): max(0, int(v)) for k, v in inventory.items()}
    ranked = sorted(
        requests,
        key=lambda r: (
            not r.compatible,
            -float(r.priority_score),
            -float(r.delay_cost),
            r.aircraft_id,
            r.component,
        ),
    )

    results: list[SpareAllocation] = []
    for rank, request in enumerate(ranked, start=1):
        if request.quantity <= 0:
            raise ValueError("spare request quantity must be positive")
        available = stock.get(request.part_id, 0) if request.compatible else 0
        allocated = min(available, request.quantity)
        stock[request.part_id] = available - allocated
        unmet = request.quantity - allocated
        reason = (
            "ALLOCATED_FULL"
            if unmet == 0
            else "ALLOCATED_PARTIAL"
            if allocated > 0
            else "WAITLIST_NO_COMPATIBLE_STOCK"
        )
        results.append(
            SpareAllocation(
                aircraft_id=request.aircraft_id,
                component=request.component,
                part_id=request.part_id,
                allocated_quantity=allocated,
                unmet_quantity=unmet,
                allocation_rank=rank,
                reason=reason,
            )
        )
    return results


def inventory_summary(
    inventory: dict[str, int],
    allocations: Iterable[SpareAllocation],
) -> dict[str, object]:
    items = list(allocations)
    used: dict[str, int] = {}
    for item in items:
        used[item.part_id] = used.get(item.part_id, 0) + item.allocated_quantity
    remaining = {
        part: max(0, int(quantity) - used.get(part, 0))
        for part, quantity in inventory.items()
    }
    return {
        "initial_inventory": {k: int(v) for k, v in inventory.items()},
        "allocated": used,
        "remaining": remaining,
        "total_requested": sum(i.allocated_quantity + i.unmet_quantity for i in items),
        "total_allocated": sum(i.allocated_quantity for i in items),
        "total_unmet": sum(i.unmet_quantity for i in items),
    }
