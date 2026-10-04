import pytest

from ml.health import HealthLevel, HealthState, fuse_health, fuse_health_state


def test_normal_state_is_typed_and_serializable():
    state = fuse_health_state(0.95, 0.05, 0.99, 0.05, 140, 0.95)
    assert isinstance(state, HealthState)
    assert state.health_level is HealthLevel.NORMAL
    assert state.alert_level == "SAFE"
    assert state.to_dict()["health_level"] == "NORMAL"


def test_watch_state_from_moderate_risk():
    state = fuse_health_state(0.90, 0.20, 0.98, 0.35, 120, 0.90)
    assert state.health_level is HealthLevel.WATCH
    assert "WATCH_FAILURE_RISK" in state.reason_codes


def test_degraded_state_from_short_rul():
    state = fuse_health_state(0.80, 0.35, 0.98, 0.30, 40, 0.90)
    assert state.health_level is HealthLevel.DEGRADED
    assert "SHORT_RUL" in state.reason_codes


def test_critical_state_from_high_risk():
    state = fuse_health_state(0.70, 0.50, 0.98, 0.90, 80, 0.90)
    assert state.health_level is HealthLevel.CRITICAL
    assert "HIGH_FAILURE_RISK" in state.reason_codes


def test_low_quality_and_confidence_are_reported():
    state = fuse_health_state(0.95, 0.05, 0.50, 0.05, 140, 0.50)
    assert "DATA_QUALITY_DEGRADED" in state.reason_codes
    assert "LOW_MODEL_CONFIDENCE" in state.reason_codes


def test_legacy_dict_contains_old_and_new_level_names():
    result = fuse_health(0.95, 0.05, 0.99, 0.05, 140, 0.95)
    assert result["alert_level"] == "SAFE"
    assert result["health_level"] == "NORMAL"


def test_invalid_values_fail_fast():
    with pytest.raises(ValueError):
        fuse_health_state(0.9, 0.1, 0.9, 0.1, -1)
