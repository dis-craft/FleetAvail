"""Runtime adapter connecting raw telemetry to all trained C-MAPSS model branches.

The runtime deliberately sits between FastAPI and the model artifacts. It owns:
- raw telemetry normalization
- bounded 30-cycle history
- temporal feature construction
- model/scaler loading
- RUL + failure-risk + anomaly inference
- cold-start fallback metadata

The API never calls sklearn/TensorFlow models directly.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from ml.cmapss.data import COLUMN_NAMES
from ml.cmapss.features import add_temporal_features, clean
from ml.cmapss.model import CMapssRULModel
from ml.cmapss.temporal_rul import TemporalRULModel
from ml.cmapss.failure_risk import XGBoostFailureRiskModel
from ml.cmapss.anomaly import IsolationForestAnomalyModel


RAW_FEATURES = tuple(COLUMN_NAMES[2:])
WINDOW_SIZE = 30


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


@dataclass
class LoadedBranch:
    name: str
    model: Any
    metadata: dict[str, Any]
    path: str


class CMapssModelRuntime:
    def __init__(
        self,
        model_dir: str | Path = "models/cmapss",
        *,
        rul_architecture: str = "lstm",
        window_size: int = WINDOW_SIZE,
    ):
        self.model_dir = Path(model_dir)
        self.window_size = window_size
        self.rul_architecture = rul_architecture
        self.histories: dict[tuple[str, str], deque[dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self.branches: dict[str, LoadedBranch] = {}
        self._load_branches()

    def _load_branches(self) -> None:
        # Prefer the temporal model selected by configuration. If it is absent,
        # fall back to the existing HistGradientBoosting baseline.
        temporal_path = self.model_dir / f"fd001_{self.rul_architecture}_rul.keras"
        temporal_meta = temporal_path.with_suffix(".json")
        if temporal_path.exists() and temporal_meta.exists():
            try:
                meta = self._read_json(temporal_meta)
                scaler_path = meta.get("scaler")
                if scaler_path and not Path(scaler_path).is_absolute() and not Path(scaler_path).exists():
                    scaler_path = self.model_dir.parent.parent / scaler_path
                scaler = joblib.load(scaler_path) if scaler_path else None
                model = TemporalRULModel.load(
                    temporal_path,
                    meta["features"],
                    int(meta["window"]),
                    scaler=scaler,
                    architecture=self.rul_architecture,
                )
                self.branches["rul"] = LoadedBranch(
                    self.rul_architecture.upper(), model, meta, str(temporal_path)
                )
            except Exception:
                pass

        baseline_path = self.model_dir / "fd001_rul.joblib"
        if "rul" not in self.branches and baseline_path.exists():
            try:
                model = CMapssRULModel.load(baseline_path)
                meta_path = baseline_path.with_suffix(".json")
                meta = self._read_json(meta_path) if meta_path.exists() else {}
                self.branches["rul"] = LoadedBranch(
                    "HIST_GRADIENT_BOOSTING", model, meta, str(baseline_path)
                )
            except Exception:
                pass

        for key, filename, cls in (
            ("failure", "fd001_failure.joblib", XGBoostFailureRiskModel),
            ("anomaly", "fd001_isolation_forest.joblib", IsolationForestAnomalyModel),
        ):
            path = self.model_dir / filename
            if not path.exists():
                continue
            try:
                model = cls.load(path)
                meta_path = path.with_suffix(".json")
                meta = self._read_json(meta_path) if meta_path.exists() else {}
                self.branches[key] = LoadedBranch(key.upper(), model, meta, str(path))
            except Exception:
                continue

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        import json
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def mode(self) -> str:
        names = [b.name for b in self.branches.values()]
        return "ml" if names else "synthetic_fallback"

    @property
    def model_status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "branches": {
                key: {
                    "model": branch.name,
                    "path": branch.path,
                    "loaded": True,
                }
                for key, branch in self.branches.items()
            },
            "required_window": self.window_size,
        }

    def normalize_telemetry(self, telemetry: Mapping[str, Any], *, cycle: int) -> dict[str, float]:
        """Accept a C-MAPSS row or compatible API telemetry payload."""
        row = {
            "cycle": float(cycle),
            "op_setting_1": _finite(telemetry.get("op_setting_1"), 0.0),
            "op_setting_2": _finite(telemetry.get("op_setting_2"), 0.0),
            "op_setting_3": _finite(telemetry.get("op_setting_3"), 0.0),
        }
        for i in range(1, 22):
            row[f"sensor_{i}"] = _finite(telemetry.get(f"sensor_{i}"), 0.0)

        # Compatibility aliases for the existing demo/API contract.
        row["sensor_2"] = _finite(telemetry.get("sensor_2", telemetry.get("egt_c")), row["sensor_2"])
        row["sensor_3"] = _finite(telemetry.get("sensor_3", telemetry.get("vibration_g")), row["sensor_3"])
        row["sensor_4"] = _finite(telemetry.get("sensor_4", telemetry.get("oil_pressure_kpa")), row["sensor_4"])
        return row

    def _feature_frame(self, history: list[dict[str, float]], feature_columns: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(history)
        frame["unit_id"] = 1
        frame["rul"] = 0.0
        base = [
            c for c in RAW_FEATURES
            if c in frame.columns and c in feature_columns
        ]
        # Temporal model metadata contains its complete post-feature columns.
        if not base:
            base = [c for c in RAW_FEATURES if c in frame.columns]
        frame = clean(frame)
        frame = add_temporal_features(frame, base)
        missing = [c for c in feature_columns if c not in frame.columns]
        if missing:
            raise ValueError("Runtime telemetry cannot construct model features: " + ", ".join(missing))
        return frame

    def _rul_predict(self, frame: pd.DataFrame, sequence: np.ndarray) -> tuple[float, str]:
        branch = self.branches.get("rul")
        if branch is None:
            return 100.0, "HEURISTIC_FALLBACK"

        model = branch.model
        if isinstance(model, TemporalRULModel):
            scaled = frame.loc[:, model.feature_columns].copy()
            if model.scaler is None:
                raise ValueError("Temporal RUL model is missing its scaler")
            scaled.loc[:, model.feature_columns] = model.scaler.transform(scaled)
            x = scaled.loc[:, model.feature_columns].to_numpy(dtype=np.float32)
            return float(model.predict(x[None, ...])[0]), branch.name

        if isinstance(model, CMapssRULModel):
            scaled = frame.loc[:, model.feature_columns].copy()
            if model.scaler is not None:
                scaled.loc[:, model.feature_columns] = model.scaler.transform(scaled)
            x = scaled.loc[:, model.feature_columns].to_numpy(dtype=np.float32)
            return float(model.predict(x[None, ...])[0]), branch.name

        raise TypeError("Unsupported RUL provider")

    def predict(
        self,
        aircraft_id: str,
        component: str,
        telemetry: Mapping[str, Any],
        *,
        cycle: int | None = None,
    ) -> dict[str, Any]:
        key = (aircraft_id, component)
        history = self.histories[key]
        next_cycle = int(cycle if cycle is not None else (history[-1]["cycle"] + 1 if history else 1))
        row = self.normalize_telemetry(telemetry, cycle=next_cycle)
        history.append(row)

        if len(history) < self.window_size:
            quality = len(history) / self.window_size
            return {
                "model_mode": self.mode,
                "model_version": "COLD_START",
                "window_size": self.window_size,
                "window_ready": False,
                "samples_available": len(history),
                "data_quality": round(quality, 3),
                "rul_cycles": None,
                "failure_probability": None,
                "anomaly_score": None,
                "confidence": round(quality * 0.5, 3),
            }

        history_rows = list(history)
        branch_features = []
        for branch in self.branches.values():
            if hasattr(branch.model, "feature_columns"):
                branch_features.extend(branch.model.feature_columns)
            branch_features.extend(branch.metadata.get("features", []))
            branch_features.extend(branch.metadata.get("base_features", []))
        feature_columns = list(dict.fromkeys(c for c in branch_features if c))
        if not feature_columns:
            return {
                "model_mode": "synthetic_fallback",
                "model_version": "HEURISTIC_FALLBACK",
                "window_size": self.window_size,
                "window_ready": True,
                "samples_available": len(history),
                "data_quality": 1.0,
                "rul_cycles": None,
                "failure_probability": None,
                "anomaly_score": None,
                "confidence": 0.0,
            }

        frame = self._feature_frame(history_rows, feature_columns)
        sequence = frame.to_numpy(dtype=np.float32)

        rul, rul_version = self._rul_predict(frame, sequence)
        risk = None
        anomaly = None
        anomaly_detected = False
        anomaly_raw = None

        failure = self.branches.get("failure")
        if failure is not None:
            risk = float(failure.model.predict_proba(frame.iloc[[-1]])[0])
        anomaly_branch = self.branches.get("anomaly")
        if anomaly_branch is not None:
            anomaly_raw = float(anomaly_branch.model.score_samples(frame.iloc[[-1]])[0])
            threshold = float(anomaly_branch.model.threshold)
            anomaly_detected = anomaly_raw >= threshold
            # Convert the raw Isolation Forest score into a bounded fusion signal.
            anomaly = float(np.clip(0.5 + (anomaly_raw - threshold) / max(abs(threshold), 1e-6), 0.0, 1.0))

        risk = 0.05 if risk is None else float(np.clip(risk, 0.0, 1.0))
        anomaly = 0.0 if anomaly is None else float(np.clip(anomaly, 0.0, 1.0))
        data_quality = 1.0

        health = float(np.clip(1.0 - (0.55 * risk + 0.45 * anomaly), 0.0, 1.0))
        confidence = float(np.clip(
            0.45 * (1.0 - risk) + 0.35 * (1.0 - anomaly) + 0.20 * data_quality,
            0.0, 1.0
        ))
        if rul_version == "HEURISTIC_FALLBACK":
            confidence *= 0.65

        return {
            "model_mode": self.mode,
            "model_version": rul_version,
            "window_size": self.window_size,
            "window_ready": True,
            "samples_available": len(history),
            "data_quality": round(data_quality, 3),
            "rul_cycles": round(float(rul), 2),
            "failure_probability": round(risk, 4),
            "anomaly_score": round(anomaly, 4),
            "anomaly_detected": anomaly_detected,
            "anomaly_score_raw": None if anomaly_raw is None else round(anomaly_raw, 6),
            "confidence": round(confidence, 4),
        }

    def seed_history(self, aircraft_id: str, component: str, rows: list[Mapping[str, Any]]) -> None:
        for row in rows[-self.window_size:]:
            self.predict(aircraft_id, component, row, cycle=int(row.get("cycle", 0) or 0))
