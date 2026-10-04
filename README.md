# Urban Flood Nowcasting System (HydroCast AI)

A complete, production-grade Urban Flood Risk Nowcasting and Forecasting platform with Machine Learning backend (FastAPI + Scikit-Learn) and modern real-time interactive frontend dashboard.

---

## Architecture Overview

1. **Python FastAPI Backend (`main.py`)**:
   - Machine Learning: `scikit-learn` `RandomForestRegressor` trained on startup with physical urban hydrology equations (Rational Method runoff, soil infiltration threshold, drainage conveyance deficit, elevation convergence).
   - Validates input features using Pydantic.
   - Outputs 7-day daily forecasts, risk scores (0–100), alert levels (`Critical`, `High`, `Moderate`, `Safe`), and hydrometeorological analysis.
   - Interactive Swagger docs at `/docs` and health probe at `/health`.
   - Enabled CORS for frontend dashboard communication.

2. **Frontend Dashboard (`index.html` / React Web App)**:
   - Interactive web form with inputs for all 11 weather and geographic features + 4 instant presets.
   - Maximum flood risk score display with color-coded severity gauge.
   - 4 key metric summary cards (Max Risk Score, Average 7-Day Rainfall, Critical Days, High Risk Days).
   - 7-Day horizontal timeline with risk bars.
   - Real-time Alert list with severity filters and civil defense action items.
   - 3 interactive Chart.js visualizations:
     1. Daily Rainfall Forecast Line Chart
     2. Flood Risk Scores Color-Coded Bar Chart
     3. Dual-Axis Temperature & Humidity Trends
   - 7-Day Forecast Data Table with tabular alignment.
   - Atmospheric, Soil, and Geographic Environmental Analysis cards.
   - PDF Export capability using `html2pdf.js`.
   - Dark theme glassmorphism aesthetic with high WCAG contrast.

---

## Quickstart: Python FastAPI Backend

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Run Backend Server
```bash
uvicorn main:app --reload --port 8000
```
or run directly:
```bash
python main.py
```

- API Base URL: `http://localhost:8000`
- Swagger UI Documentation: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`
- Predict Endpoint: `POST http://localhost:8000/predict`

---

## Deploy to Replit

1. Create a new Python Repl.
2. Upload `main.py` and `requirements.txt`.
3. In Replit's `.replit` configuration, set the run command:
   ```toml
   run = "uvicorn main:app --host 0.0.0.0 --port 8000"
   ```
4. Click **Run**. Replit will provide a public HTTPS URL (e.g. `https://hydrocast.<user>.repl.co`).
5. Open the frontend dashboard and enter your Replit URL in the **Backend Settings** modal!

---

## API Request Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Central Metro Basin",
    "rainfall": 85.0,
    "temperature": 28.0,
    "humidity": 88.0,
    "pressure": 998.0,
    "cloudCover": 90.0,
    "soilMoisture": 84.0,
    "drainageQuality": 4.0,
    "elevation": 12.0,
    "urbanDevPercent": 75.0,
    "windSpeed": 25.0,
    "forecastHours": 168
  }'
```
