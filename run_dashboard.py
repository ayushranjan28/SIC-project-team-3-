"""
Samsung Galaxy Health - PINN Telemetry Dashboard Launcher
Starts the FastAPI telemetry server and automatically opens the browser.
"""
import webbrowser
import os
import sys
import time
import threading
import uvicorn

def main():
    port = 8000
    url = f"http://localhost:{port}"
    print("=" * 75)
    print("  SAMSUNG GALAXY HEALTH &bull; PINN TELEMETRY WORKSTATION")
    print("  Real-Time Wearable Physiological Forecasting Console")
    print(f"  Server URL: {url}")
    print("=" * 75)
    
    def open_browser():
        time.sleep(1.2)
        print(f"--> Launching web browser at {url} ...")
        webbrowser.open(url)
    
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=port, reload=False, log_level="info")

if __name__ == "__main__":
    main()

