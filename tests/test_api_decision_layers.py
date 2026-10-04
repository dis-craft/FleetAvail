from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_decision_api_surface():
    assert client.get("/health").status_code == 200
    assert client.get("/api/fleet/summary").status_code == 200
    assert client.get("/api/fleet/availability").status_code == 200

    twin = client.get("/api/fleet/aircraft/AF-001/twin")
    assert twin.status_code == 200
    assert twin.json()["aircraft_id"] == "AF-001"

    plan = client.post(
        "/api/maintenance/plan",
        json={"mission_priority": 1, "horizon_days": 7, "max_daily_hours": 24},
    )
    assert plan.status_code == 200
    assert len(plan.json()["items"]) == 48

    allocation = client.post(
        "/api/spares/allocate",
        json={"mission_priority": 1, "horizon_days": 7, "max_daily_hours": 24},
    )
    assert allocation.status_code == 200
    assert "inventory" in allocation.json()

    projection = client.post(
        "/api/fleet/availability",
        json={"mission_priority": 1, "horizon_days": 7, "max_daily_hours": 24},
    )
    assert projection.status_code == 200
    assert projection.json()["total_aircraft"] == 12


def test_prediction_updates_digital_twin():
    response = client.post(
        "/api/predict",
        json={
            "aircraft_id": "AF-003",
            "component": "ENGINE",
            "telemetry": {
                "cycle": 42,
                "egt_c": 790,
                "vibration_g": 0.9,
                "oil_pressure_kpa": 290,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["aircraft_id"] == "AF-003"
    assert payload["component"] == "ENGINE"

    twin = client.get("/api/fleet/aircraft/AF-003/twin").json()
    assert twin["components"]["ENGINE"]["last_update_cycle"] == 42
    assert twin["events"][-1]["event_type"] == "PREDICTION_UPDATE"
