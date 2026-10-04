# FleetAvail Setup & Verification Guide

## Prerequisites
- Windows 10/11, Linux or macOS
- Python 3.11 recommended
- Git

Node.js is not required for the current first-pass dashboard because it is served directly by FastAPI.

## 1. Clone
git clone https://github.com/dis-craft/FleetAvail.git
cd FleetAvail

## 2. Create environment

Windows PowerShell:
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

If activation is blocked:
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

Linux/macOS:
python3.11 -m venv .venv
source .venv/bin/activate

## 3. Install
python -m pip install --upgrade pip
pip install -r requirements.txt

## 4. Generate demo data
python scripts/generate_demo_data.py

Expected: data/demo_telemetry.csv containing 720 synthetic engine telemetry records.

## 5. Start
python -m uvicorn backend.app.main:app --reload --port 8000

Open:
- Dashboard: http://127.0.0.1:8000/
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 6. Verify
In a second terminal:
python scripts/run_smoke_test.py
pytest -q

## 7. Manual demo
Fleet: verify 12 aircraft, readiness, current availability, projected availability and critical count.

What-if: select an aircraft, set degradation to 20%, click Simulate. Health/RUL and projected fleet availability should decrease.

Maintenance: select an aircraft and click Recommend. The response returns the priority component, action, schedule day, RUL, risk, spare and reason codes.
Maintenance execution: POST /api/fleet/aircraft/AF-004/maintenance with a component, action and cycle; then GET the twin endpoint to see the MAINTENANCE event.

Decision APIs:
- GET /api/fleet/availability — current and projected fleet readiness
- GET /api/fleet/aircraft/AF-001/twin — persisted digital-twin state
- POST /api/maintenance/plan — 7-day constrained maintenance plan
- POST /api/spares/allocate — constrained spare allocation
- POST /api/fleet/availability — projected readiness using the decision plan

Live telemetry: the dashboard connects to /ws/telemetry and updates approximately once per second.

## 8. API examples

PowerShell:
Invoke-RestMethod http://127.0.0.1:8000/api/fleet/summary

Prediction:
$body = @{aircraft_id="AF-003"; component="ENGINE"; telemetry=@{egt_c=790; vibration_g=0.9; oil_pressure_kpa=290}} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/predict -Method Post -ContentType "application/json" -Body $body

## Troubleshooting
If uvicorn is not recognized:
python -m uvicorn backend.app.main:app --reload --port 8000

If backend imports fail, run Uvicorn from the repository root.

If port 8000 is busy:
python -m uvicorn backend.app.main:app --reload --port 8010

The decision-layer features added in October 2026 do not require TensorFlow or PyTorch. LSTM/TCN training remains a separate optional environment step.
The digital twin uses a local JSON persistence file by default so the prototype remains runnable without PostgreSQL/Redis. The store API is intentionally replaceable by those services later.

## Data boundary
Put downloaded C-MAPSS files under data/raw/. Do not commit proprietary or sensitive aircraft data.
