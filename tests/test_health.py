from ml.health import fuse_health


def test_health_states():
    assert fuse_health(0.95, 0.05, 1.0, 0.05, 120)["operational_state"] == "NORMAL"
    assert fuse_health(0.75, 0.50, 1.0, 0.35, 80)["operational_state"] == "WATCH"
    assert fuse_health(0.55, 0.75, 1.0, 0.65, 40)["operational_state"] == "DEGRADED"
    assert fuse_health(0.30, 0.95, 0.8, 0.90, 10)["operational_state"] == "CRITICAL"


def test_compatibility_alert_level_matches_state():
    result = fuse_health(0.9, 0.1, 1.0, 0.1, 100)
    assert result["alert_level"] == result["operational_state"]
