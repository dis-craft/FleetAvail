"""Formal, typed health-fusion layer for FleetAvail.

The fusion layer is deliberately independent of any particular ML model. It accepts
already-produced health, RUL, failure-risk, anomaly, confidence, and data-quality
signals and converts them into one operational HealthState.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class HealthLevel(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


_LEGACY_ALERT = {
    HealthLevel.NORMAL: "SAFE",
    HealthLevel.WATCH: "WARNING",
    HealthLevel.DEGRADED: "DEGRADED",
    HealthLevel.CRITICAL: "CRITICAL",
}


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    value = float(value)
    if not _is_finite(value):
        raise ValueError("health-fusion inputs must be finite")
    return min(high, max(low, value))


def _rul(value: float) -> float:
    value = float(value)
    if not _is_finite(value):
        raise ValueError("rul_cycles must be finite")
    if value < 0:
        raise ValueError("rul_cycles cannot be negative")
    return value


@dataclass(frozen=True)
class HealthState:
    """Unified operational health state produced by model-output fusion."""

    health_score: float
    failure_probability: float
    anomaly_score: float
    rul_cycles: float
    confidence: float
    data_quality: float
    health_level: HealthLevel
    reason_codes: tuple[str, ...] = ()
    aircraft_id: str | None = None
    component: str | None = None

    @property
    def alert_level(self) -> str:
        """Legacy display value retained for existing API/dashboard consumers."""
        return _LEGACY_ALERT[self.health_level]

    def to_dict(self) -> dict[str, Any]:
        return {
            "aircraft_id": self.aircraft_id,
            "component": self.component,
            "health_score": self.health_score,
            "failure_probability": self.failure_probability,
            "anomaly_score": self.anomaly_score,
            "rul_cycles": self.rul_cycles,
            "confidence": self.confidence,
            "data_quality": self.data_quality,
            "health_level": self.health_level.value,
            "operational_state": self.health_level.value,
            "alert_level": self.alert_level,
            "reason_codes": list(self.reason_codes),
        }


def _classify(
    *,
    health_score: float,
    failure_probability: float,
    anomaly_score: float,
    rul_cycles: float,
) -> tuple[HealthLevel, tuple[str, ...]]:
    critical_reasons: list[str] = []
    if failure_probability >= 0.85:
        critical_reasons.append("HIGH_FAILURE_RISK")
    if rul_cycles <= 20:
        critical_reasons.append("LOW_RUL")
    if anomaly_score >= 0.90:
        critical_reasons.append("SEVERE_ANOMALY")
    if health_score < 35:
        critical_reasons.append("LOW_HEALTH_SCORE")
    if critical_reasons:
        return HealthLevel.CRITICAL, tuple(dict.fromkeys(critical_reasons))

    degraded_reasons: list[str] = []
    if failure_probability >= 0.60:
        degraded_reasons.append("ELEVATED_FAILURE_RISK")
    if rul_cycles <= 45:
        degraded_reasons.append("SHORT_RUL")
    if anomaly_score >= 0.70:
        degraded_reasons.append("HIGH_ANOMALY")
    if health_score < 55:
        degraded_reasons.append("DEGRADED_HEALTH")
    if degraded_reasons:
        return HealthLevel.DEGRADED, tuple(dict.fromkeys(degraded_reasons))

    watch_reasons: list[str] = []
    if failure_probability >= 0.30:
        watch_reasons.append("WATCH_FAILURE_RISK")
    if rul_cycles <= 90:
        watch_reasons.append("WATCH_RUL")
    if anomaly_score >= 0.45:
        watch_reasons.append("WATCH_ANOMALY")
    if health_score < 75:
        watch_reasons.append("WATCH_HEALTH")
    if watch_reasons:
        return HealthLevel.WATCH, tuple(dict.fromkeys(watch_reasons))

    return HealthLevel.NORMAL, ()


def fuse_health_state(
    health: float,
    anomaly: float,
    data_quality: float,
    risk: float,
    rul: float,
    confidence: float = 1.0,
    *,
    aircraft_id: str | None = None,
    component: str | None = None,
) -> HealthState:
    """Fuse normalized signals into a deterministic operational HealthState.

    health is the component health fraction in [0, 1].
    anomaly, risk, confidence and data_quality are normalized to [0, 1].
    RUL is measured in cycles and must be non-negative.
    """

    health = _clamp(health)
    anomaly = _clamp(anomaly)
    data_quality = _clamp(data_quality)
    risk = _clamp(risk)
    confidence = _clamp(confidence)
    rul = _rul(rul)

    score = 100.0 * (
        0.50 * health
        + 0.25 * (1.0 - anomaly)
        + 0.25 * data_quality
    )
    score = round(min(100.0, max(0.0, score)), 1)

    level, reasons = _classify(
        health_score=score,
        failure_probability=risk,
        anomaly_score=anomaly,
        rul_cycles=rul,
    )

    extra_reasons: list[str] = list(reasons)
    if data_quality < 0.80:
        extra_reasons.append("DATA_QUALITY_DEGRADED")
    if confidence < 0.60:
        extra_reasons.append("LOW_MODEL_CONFIDENCE")

    return HealthState(
        health_score=score,
        failure_probability=round(risk, 3),
        anomaly_score=round(anomaly, 3),
        rul_cycles=round(rul, 1),
        confidence=round(confidence, 3),
        data_quality=round(data_quality, 3),
        health_level=level,
        reason_codes=tuple(dict.fromkeys(extra_reasons)),
        aircraft_id=aircraft_id,
        component=component,
    )


def fuse_health(health, anomaly, data_quality, risk, rul, confidence=1.0):
    """Backward-compatible dictionary API used by the original prototype."""

    return fuse_health_state(
        health=health,
        anomaly=anomaly,
        data_quality=data_quality,
        risk=risk,
        rul=rul,
        confidence=confidence,
    ).to_dict()
