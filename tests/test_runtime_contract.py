from ml.cmapss.runtime import CMapssModelRuntime


def test_runtime_normalizes_legacy_engine_fields(tmp_path):
    runtime = CMapssModelRuntime(tmp_path)
    row = runtime.normalize_telemetry(
        {"egt_c": 700, "vibration_g": 0.2, "oil_pressure_kpa": 400},
        cycle=7,
    )
    assert row["cycle"] == 7
    assert row["sensor_2"] == 700
    assert row["sensor_3"] == 0.2
    assert row["sensor_4"] == 400
    assert len([k for k in row if k.startswith("sensor_")]) == 21


def test_runtime_has_cold_start_contract(tmp_path):
    runtime = CMapssModelRuntime(tmp_path)
    result = runtime.predict("AF-001", "ENGINE", {"sensor_2": 700}, cycle=1)
    assert result["window_ready"] is False
    assert result["samples_available"] == 1
    assert result["rul_cycles"] is None
    assert 0 <= result["data_quality"] <= 1
