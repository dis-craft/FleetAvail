import pytest

from decision_engine.fleet_availability import (
    AircraftAvailabilityInput,
    calculate_fleet_availability,
)
from decision_engine.maintenance_optimizer import (
    MaintenanceCandidate,
    build_maintenance_plan,
)
from decision_engine.spares import SpareRequest, allocate_spares, inventory_summary
from digital_twin.state import DigitalTwinStore


def test_maintenance_plan_prioritizes_urgent_work_and_skips_monitor_capacity():
    candidates = [
        MaintenanceCandidate("AF-001", "ENGINE", 1.0, 0.95, 12, 8, "ENG-FLT", 2),
        MaintenanceCandidate("AF-002", "ENGINE", 1.0, 0.20, 140, 8, "ENG-FLT", 2),
    ]
    plan = build_maintenance_plan(candidates, horizon_days=7, max_maintenance_hours_per_day=8)
    assert plan[0].aircraft_id == "AF-001"
    assert plan[0].action == "GROUND_AND_MAINTAIN"
    assert plan[0].scheduled_day == 0
    assert plan[1].action == "MONITOR"


def test_spare_allocation_respects_inventory_and_priority():
    requests = [
        SpareRequest("AF-001", "ENGINE", "ENG-FLT", 1, 9.0, 5.0),
        SpareRequest("AF-002", "ENGINE", "ENG-FLT", 1, 4.0, 2.0),
    ]
    allocations = allocate_spares(requests, {"ENG-FLT": 1})
    assert allocations[0].allocated_quantity == 1
    assert allocations[1].allocated_quantity == 0
    assert allocations[1].unmet_quantity == 1
    summary = inventory_summary({"ENG-FLT": 1}, allocations)
    assert summary["total_allocated"] == 1
    assert summary["total_unmet"] == 1


def test_incompatible_spare_is_never_allocated():
    request = SpareRequest("AF-001", "ENGINE", "ENG-FLT", 1, 9.0, 10.0, compatible=False)
    item = allocate_spares([request], {"ENG-FLT": 9})[0]
    assert item.allocated_quantity == 0
    assert item.unmet_quantity == 1


def test_fleet_availability_recovers_blocked_aircraft_when_maintenance_and_spare_exist():
    items = [
        AircraftAvailabilityInput("AF-001", "READY"),
        AircraftAvailabilityInput(
            "AF-002",
            "MAINTENANCE",
            maintenance_day=1,
            maintenance_duration_hours=8,
            spare_available=True,
        ),
    ]
    result = calculate_fleet_availability(items, horizon_days=7)
    assert result["current_availability_pct"] == 50.0
    assert result["projected_availability_pct"] == 100.0
    assert result["recovered_aircraft"] == ["AF-002"]


def test_fleet_availability_does_not_recover_without_spare():
    items = [
        AircraftAvailabilityInput(
            "AF-002",
            "MAINTENANCE",
            maintenance_day=1,
            maintenance_duration_hours=8,
            spare_available=False,
        )
    ]
    result = calculate_fleet_availability(items, horizon_days=7)
    assert result["projected_available"] == 0
    assert result["blocked_aircraft"] == ["AF-002"]


def test_digital_twin_persists_predictions_and_maintenance(tmp_path):
    path = tmp_path / "twin.json"
    store = DigitalTwinStore(path)
    store.apply_prediction(
        "AF-001",
        "ENGINE",
        health_score=40,
        rul_cycles=18,
        failure_probability=0.90,
        anomaly_score=0.88,
        confidence=0.82,
        data_quality=0.97,
        cycle=30,
    )
    before = store.snapshot("AF-001")
    assert before["components"]["ENGINE"]["lifecycle_status"] == "CRITICAL"
    assert len(before["events"]) == 1

    reloaded = DigitalTwinStore(path)
    restored = reloaded.snapshot("AF-001")
    assert restored["components"]["ENGINE"]["rul_cycles"] == 18.0

    reloaded.record_maintenance(
        "AF-001", "ENGINE", action="REPLACE_COMPONENT", cycle=30
    )
    after = reloaded.snapshot("AF-001")
    assert after["components"]["ENGINE"]["lifecycle_status"] == "IN_SERVICE"
    assert len(after["events"]) == 2


def test_twin_rejects_out_of_order_cycles(tmp_path):
    store = DigitalTwinStore(tmp_path / "twin.json")
    store.apply_prediction(
        "AF-001", "ENGINE",
        health_score=80, rul_cycles=100, failure_probability=0.1,
        anomaly_score=0.1, confidence=0.9, data_quality=1.0, cycle=10
    )
    with pytest.raises(ValueError):
        store.apply_prediction(
            "AF-001", "ENGINE",
            health_score=80, rul_cycles=99, failure_probability=0.1,
            anomaly_score=0.1, confidence=0.9, data_quality=1.0, cycle=9
        )
