"""
Vercel Serverless Function Entrypoint for Samsung Health PINN Forecaster.
Routes incoming /api/* requests and fallback root requests.
"""
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dashboard.server import app, engine, live_manager, LiveSampleInput
from fastapi.responses import JSONResponse
from fastapi import Query

# Register route aliases without /api prefix for robust Vercel routing
@app.get("/config")
def api_alias_config():
    limits = engine.get_limits()
    limits["local_ip"] = "10.168.207.145"
    limits["ingest_url"] = "/api/live/ingest"
    return limits

@app.get("/telemetry")
def api_alias_telemetry(
    cutoff: int = Query(180),
    artifact_intensity: float = Query(1.0),
    noise_level: float = Query(0.2)
):
    return JSONResponse(content=engine.compute_telemetry(cutoff, artifact_intensity, noise_level))

@app.get("/benchmarks")
def api_alias_benchmarks():
    return JSONResponse(content=engine.benchmarks)

@app.get("/plots")
def api_alias_plots():
    from dashboard.server import get_plots
    return get_plots()

@app.get("/live/telemetry")
def api_alias_live_telemetry():
    return JSONResponse(content=live_manager.get_payload())

@app.post("/live/ingest")
def api_alias_live_ingest(sample: LiveSampleInput):
    return JSONResponse(content=live_manager.ingest_sample(sample))

@app.post("/live/reset")
def api_alias_live_reset():
    live_manager.reset()
    return JSONResponse(content={"status": "cleared", "total_samples": 0})

@app.get("/live/status")
def api_alias_live_status():
    from dashboard.server import get_live_status
    return get_live_status()
