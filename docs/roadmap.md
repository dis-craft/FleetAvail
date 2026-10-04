# FleetAvail implementation roadmap

This roadmap follows the completion chain in order. Each layer is independently testable,
while the existing synthetic dashboard remains runnable during the transition to trained ML.

| Stage | Feature | Current implementation |
|---|---|---|
| 1 | C-MAPSS baseline/provider | ✅ Leakage-aware preprocessing, 30-cycle windows, baseline RUL model and provider boundary |
| 2 | Telemetry sequence buffer | ✅ Bounded per-aircraft/component 30-cycle history with ordering validation |
| 3 | XGBoost failure model | ✅ Near-term failure risk with engine-aware splitting and probability calibration |
| 4 | Isolation Forest | ✅ Normal-reference anomaly detector with calibrated threshold and persistence |
| 5 | Health Fusion | ✅ Typed NORMAL/WATCH/DEGRADED/CRITICAL state with reason codes |
| 6 | LSTM/TCN RUL | ✅ LSTM and causal dilated TCN training/inference branch; offline comparison only |
| 7 | Real-time ML inference | ⏳ Wire trained XGBoost/RUL/anomaly providers into telemetry and FastAPI |
| 8 | Digital Twin persistence | ✅ Persistent aircraft/component lifecycle state and event history using atomic local JSON; backend-neutral store API |
| 9 | Maintenance optimizer | ✅ Constraint-aware 7-day scheduling with risk/RUL/mission priority, duration and spare constraints |
| 10 | Spare allocation | ✅ Priority-aware allocation that never exceeds compatible inventory and reports unmet demand |
| 11 | Fleet availability | ✅ Current/projected readiness calculation using maintenance schedules and spare feasibility |
| 12 | MLOps/explainability | ⏳ Versioning, drift, uncertainty, SHAP, prediction monitoring and audit trail |
| 13 | Final frontend | ⏳ React/Next.js + Three.js operational dashboard |

## Current architecture

Telemetry + maintenance history
→ validation / data quality
→ leakage-aware preprocessing
→ XGBoost failure risk + LSTM/TCN RUL + Isolation Forest anomaly
→ Health Fusion
→ Digital Twin
→ Maintenance Optimizer
→ Spare Allocation
→ Fleet Availability
→ API / dashboard

## Important implementation boundary

The repository now contains the decision layers requested in stages 5 and 8–11.
The FastAPI service still uses the deterministic synthetic fleet state for the demo; stage 7
is where trained model artifacts become the runtime inference source.

## Feature 5 — Health Fusion verification

Run:
pytest -q tests/test_health_fusion.py

The tests cover the four operational states, reason codes, legacy compatibility, and invalid inputs.

## Feature 8 — Digital Twin verification

The store persists component predictions, lifecycle status, maintenance events and the latest bounded
event history. It defaults to data/digital_twin_state.json.

Run:
pytest -q tests/test_decision_layers.py -k digital_twin

The persistence file is local runtime state and is ignored by Git.

## Feature 9 — Maintenance optimizer verification

Run:
pytest -q tests/test_decision_layers.py -k maintenance

The planner ranks candidates using failure risk, RUL urgency, mission priority and spare availability,
then respects a daily maintenance-hour capacity.

API:
curl.exe -X POST http://127.0.0.1:8000/api/maintenance/plan -H "Content-Type: application/json" -d "{\"mission_priority\":1,\"horizon_days\":7,\"max_daily_hours\":24}"

## Feature 10 — Spare allocation verification

Run:
pytest -q tests/test_decision_layers.py -k spare

The allocator uses only compatible stock and never allocates more than the available quantity.
The response reports allocated and unmet demand.

API:
curl.exe -X POST http://127.0.0.1:8000/api/spares/allocate -H "Content-Type: application/json" -d "{\"mission_priority\":1,\"horizon_days\":7,\"max_daily_hours\":24}"

## Feature 11 — Fleet availability verification

Run:
pytest -q tests/test_decision_layers.py -k fleet_availability

The calculation reports current readiness, projected readiness, blocked aircraft and aircraft that
can recover within the planning horizon when maintenance and compatible spares are available.

API:
curl.exe http://127.0.0.1:8000/api/fleet/availability

or:
curl.exe -X POST http://127.0.0.1:8000/api/fleet/availability -H "Content-Type: application/json" -d "{\"mission_priority\":1,\"horizon_days\":7,\"max_daily_hours\":24}"

## Full verification

From the repository root:
pip install -r requirements.txt
pytest -q tests/test_health_fusion.py
pytest -q tests/test_decision_layers.py
pytest -q tests/test_api_decision_layers.py
pytest -q
python -m uvicorn backend.app.main:app --reload --port 8000

## C-MAPSS training remains separate

The NASA dataset is not committed. Put these files under data/raw/cmapss/FD001/:
train.txt
test.txt
RUL_test.txt

LSTM/TCN additionally require TensorFlow in the local Python environment because TensorFlow is intentionally
not part of the base requirements file.

## Next implementation target

Stage 7: real-time ML inference. That is where the sequence buffer, trained XGBoost, trained RUL model,
and Isolation Forest become one runtime inference pipeline and feed the formal Health Fusion layer.
