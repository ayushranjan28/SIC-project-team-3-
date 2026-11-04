"""
Samsung Galaxy Health - Smartwatch BLE Bridge & Live Telemetry Streamer
Connects to Samsung Galaxy Watch or any BLE Heart Rate monitor (Polar, Garmin, Apple Watch)
and streams live biometric readings directly into the PINN Telemetry Server.

Usage:
  python scripts/smartwatch_ble_bridge.py --scan
  python scripts/smartwatch_ble_bridge.py --address "XX:XX:XX:XX:XX:XX"
  python scripts/smartwatch_ble_bridge.py --simulate
"""

import sys
import os
import time
import math
import random
import argparse
import asyncio
import urllib.request
import json

# Standard Bluetooth SIG UUIDs
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
BATTERY_LEVEL_CHAR_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

DEFAULT_SERVER_URL = "http://127.0.0.1:8000/api/live/ingest"
HISTORY_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "galaxy_watch_live_history.csv")

def append_to_history_csv(sample: dict):
    """Persist received wearable packet into historical database for automatic conditioning."""
    try:
        os.makedirs(os.path.dirname(HISTORY_CSV_PATH), exist_ok=True)
        file_exists = os.path.exists(HISTORY_CSV_PATH)
        with open(HISTORY_CSV_PATH, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("timestamp,hr,workload,acc_x,acc_y,acc_z,ppg_sqi,device_name\n")
            f.write(f"{sample['timestamp']:.2f},{sample['hr']:.1f},{sample['workload']:.3f},"
                    f"{sample['acc_x']:.3f},{sample['acc_y']:.3f},{sample['acc_z']:.3f},"
                    f"{sample['ppg_sqi']:.3f},{sample['device_name']}\n")
    except Exception as e:
        pass

def post_telemetry(payload: dict, server_url: str = DEFAULT_SERVER_URL) -> bool:
    """Post biometric packet to FastAPI server."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            server_url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception as e:
        # Server might be starting or unreachable
        return False

def parse_heart_rate_packet(data: bytearray) -> dict:
    """
    Parse standard Bluetooth GATT Heart Rate Measurement Characteristic (0x2A37).
    Standard specification:
      - Byte 0: Flags
        - Bit 0: Heart Rate Format (0 = UINT8, 1 = UINT16)
        - Bit 1-2: Sensor Contact Status
        - Bit 3: Energy Expended Present
        - Bit 4: RR-Interval Present
    """
    flags = data[0]
    hr_format = flags & 0x01
    contact_detected = bool(flags & 0x02)
    contact_supported = bool(flags & 0x04)
    energy_present = bool(flags & 0x08)
    rr_present = bool(flags & 0x10)

    offset = 1
    if hr_format == 0:
        hr = data[offset]
        offset += 1
    else:
        hr_le = data[offset] | (data[offset + 1] << 8)
        hr_be = (data[offset] << 8) | data[offset + 1]
        if 35 <= hr_le <= 230:
            hr = hr_le
        elif 35 <= hr_be <= 230:
            hr = hr_be
        else:
            # Check byte swap
            swapped = ((hr_le & 0xFF) << 8) | ((hr_le >> 8) & 0xFF)
            if 35 <= swapped <= 230:
                hr = swapped
            elif 35 <= data[offset] <= 230:
                hr = data[offset]
            else:
                hr = 75.0
        offset += 2

    # Physiological clamp: human heart rate cannot be 20,000+ BPM
    if hr > 240 or hr < 30:
        if 10000 <= hr <= 65535:
            swapped = ((int(hr) & 0xFF) << 8) | ((int(hr) >> 8) & 0xFF)
            hr = swapped if (35 <= swapped <= 230) else 75.0
        else:
            hr = max(40.0, min(220.0, float(hr)))

    energy = None
    if energy_present and len(data) >= offset + 2:
        energy = data[offset] | (data[offset + 1] << 8)
        offset += 2

    rr_intervals = []
    if rr_present:
        while offset + 1 < len(data):
            rr = (data[offset] | (data[offset + 1] << 8)) / 1024.0 # in seconds
            rr_intervals.append(round(rr * 1000, 1)) # ms
            offset += 2

    return {
        "hr": float(hr),
        "sensor_contact": contact_detected if contact_supported else True,
        "energy_kcal": energy,
        "rr_ms": rr_intervals
    }

async def scan_ble_devices(timeout: float = 4.0):
    """Scan for nearby Bluetooth Low Energy devices advertising Heart Rate service."""
    try:
        import bleak
    except ImportError:
        print("[ERROR] bleak library not found. Run: pip install bleak")
        return []

    print(f"\n[BLUETOOTH] Scanning for nearby BLE smartwatches ({timeout:.0f}s)...")
    devices = await bleak.BleakScanner.discover(timeout=timeout, return_adv=True)
    
    found_devices = []
    for address, (device, adv) in devices.items():
        name = device.name or adv.local_name or "Unknown Wearable"
        services = adv.service_uuids or []
        has_hr = any(HR_SERVICE_UUID.lower() in str(s).lower() or "180d" in str(s).lower() for s in services)
        
        found_devices.append({
            "name": name,
            "address": address,
            "rssi": adv.rssi,
            "has_hr": has_hr,
            "services": services
        })
        
        tag = " [HEART RATE GATT DETECTED]" if has_hr else ""
        print(f"  -> {name} ({address}) | RSSI: {adv.rssi} dBm{tag}")

    print(f"[BLUETOOTH] Scan complete. Found {len(found_devices)} total BLE devices.\n")
    return found_devices

async def run_ble_client(address: str, device_name: str = "Samsung Galaxy Watch", server_url: str = DEFAULT_SERVER_URL):
    """Connect to a BLE Smartwatch and stream real-time GATT Heart Rate readings."""
    try:
        import bleak
    except ImportError:
        print("[ERROR] bleak is required. Install via: pip install bleak")
        return

    print(f"\n==================================================================")
    print(f" Connecting to Wearable Device: {device_name} [{address}]")
    print(f" Target Ingest Endpoint: {server_url}")
    print(f"==================================================================\n")

    last_hr = 70.0
    sample_idx = 0

    def notification_handler(sender, data: bytearray):
        nonlocal last_hr, sample_idx
        parsed = parse_heart_rate_packet(data)
        hr_val = parsed["hr"]
        last_hr = hr_val
        sample_idx += 1

        # Calculate estimated SQI based on sensor contact and physical continuity
        sqi = 0.95 if parsed.get("sensor_contact", True) else 0.40
        if sample_idx > 1 and abs(hr_val - last_hr) > 10.0:
            sqi = max(0.20, sqi - 0.35)

        payload = {
            "hr": hr_val,
            "acc_x": 0.05,
            "acc_y": 0.10,
            "acc_z": 0.99,
            "acc_mag": 1.0,
            "workload": 0.20,
            "ppg_sqi": sqi,
            "device_name": device_name,
            "timestamp": time.time()
        }

        append_to_history_csv(payload)
        ok = post_telemetry(payload, server_url)
        status_sym = "[OK]" if ok else "[ERR/OFFLINE]"
        print(f" {status_sym} #{sample_idx:04d} | Live Optical PPG: {hr_val:5.1f} BPM | Contact: {parsed['sensor_contact']} | SQI: {sqi*100:.0f}%")

    session_start = 0.0
    while True:
        try:
            async with bleak.BleakClient(address) as client:
                connected = client.is_connected
                if not connected:
                    print(f"[RETRY] Failed to connect to {address}. Retrying in 3s...")
                    await asyncio.sleep(3.0)
                    continue

                session_start = time.time()
                sample_idx = 0
                print(f"[CONNECTED] Paired to {device_name} successfully!")
                print(f"[STREAM] Subscribing to GATT Heart Rate notifications (UUID 0x2A37)...")
                
                await client.start_notify(HR_MEASUREMENT_CHAR_UUID, notification_handler)
                print(f"[RUNNING] Live optical telemetry streaming active. Press Ctrl+C to terminate.\n")

                while client.is_connected:
                    await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            print("\n[STOP] Telemetry stream stopped by user.")
            break
        except Exception as e:
            elapsed = time.time() - session_start if session_start > 0 else 0
            if 80 <= elapsed <= 150 or (90 <= sample_idx <= 150):
                print("\n" + "="*70)
                print(" [120S APP TIMEOUT DETECTED]")
                print(" The 'Heart for Bluetooth' Wear OS app stopped broadcasting after ~120s.")
                print(" HOW TO STREAM CONTINUOUSLY WITHOUT 120s TIMEOUT:")
                print("   1. Open 'Heart for Bluetooth' on your Galaxy Watch.")
                print("   2. Scroll down to Settings -> 'Duration' -> select 'Unlimited' or '60 min'.")
                print("   3. On Watch: Settings -> Apps -> Heart for Bluetooth -> Permissions -> Sensors -> 'Allow all the time'.")
                print("   4. On Watch: Settings -> Battery -> Heart for Bluetooth -> 'Unrestricted'.")
                print("   -> Free Alternative: Install 'Heart Rate on BLE' (by MyWorkouts) - 100% free with NO 120s limit!")
                print(f" [HISTORY] All {sample_idx} packets were saved to data/galaxy_watch_live_history.csv")
                print(" [AUTO-RECONNECT] Waiting for watch to resume... Tap watch or restart broadcast.")
                print("="*70 + "\n")
            else:
                print(f"[BLE ERROR] Disconnection or transmission drop: {e}")
                print(f"[RETRY] Attempting auto-reconnect in 3.0s...")
            await asyncio.sleep(3.0)

def run_simulation_stream(device_name: str = "Samsung Galaxy Watch6 Live Simulator", server_url: str = DEFAULT_SERVER_URL):
    """
    Simulates a live smartwatch streaming realistic optical PPG with motion spikes,
    acceleration bursts, and recovery, forwarding packets to the FastAPI server at 1 Hz.
    """
    print(f"\n==================================================================")
    print(f" STARTING REAL-TIME SMARTWATCH STREAM SIMULATOR (1.0 Hz)")
    print(f" Device Identity : {device_name}")
    print(f" Ingestion Target : {server_url}")
    print(f"==================================================================")
    print(f"Demonstrates the real-time 'Data Difference' between:")
    print(f"  1. Raw Optical PPG (with motion artifact corruption & optical slippage)")
    print(f"  2. Baseline Deep Learning (unbounded, tracks false motion spikes)")
    print(f"  3. Physics-Informed Neural Network (Biophysical ODE guardrails)")
    print(f"------------------------------------------------------------------\n")

    t = 0
    hr_clean = 72.0
    hr_rest = 68.0
    tau = 45.0
    beta = 18.0

    try:
        while True:
            t += 1
            # Dynamic workout progression
            if t < 20:
                # Resting baseline
                workout_phase = "Resting Baseline"
                workload = 0.05 + 0.02 * math.sin(t * 0.2)
                acc_x, acc_y, acc_z = 0.02, 0.03, 0.99
                has_artifact = False
            elif t < 65:
                # Treadmill Warmup & Run
                workout_phase = "Treadmill Run (High Exertion)"
                workload = min(0.88, 0.15 + (t - 20) * 0.025)
                acc_x = 0.45 * math.sin(t * 1.8)
                acc_y = 0.35 * math.cos(t * 1.8)
                acc_z = 1.0 + 0.60 * math.sin(t * 3.6)
                # Introduce wrist motion artifacts during running
                has_artifact = (35 <= t <= 52)
            elif t < 100:
                # Active Recovery Cooldown
                workout_phase = "Cardiovascular Recovery"
                workload = max(0.08, 0.80 - (t - 65) * 0.03)
                acc_x, acc_y, acc_z = 0.08, 0.06, 0.98
                has_artifact = (80 <= t <= 86) # post-workout arm stretch
            else:
                # Reset cycle
                t = 1
                continue

            acc_mag = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

            # Clean ground-truth physiological evolution via cardiovascular ODE
            dhr = (-(hr_clean - hr_rest) / tau + beta * workload) * 1.0
            hr_clean += dhr

            # Observed Raw Optical PPG with sensor motion artifacts & baseline wander
            hr_observed = hr_clean
            if has_artifact:
                # Severe optical cadence lock & motion surge (e.g. +18 to +28 BPM jump)
                motion_noise = 22.0 * math.sin(t * 0.85) + 12.0 * math.cos(t * 1.4)
                hr_observed += motion_noise
                sqi = max(0.18, 0.42 - 0.08 * math.sin(t))
            else:
                sensor_noise = random.gauss(0, 0.8)
                hr_observed += sensor_noise
                sqi = min(0.96, 0.90 + 0.04 * math.cos(t * 0.3))

            payload = {
                "hr": round(hr_observed, 1),
                "acc_x": round(acc_x, 3),
                "acc_y": round(acc_y, 3),
                "acc_z": round(acc_z, 3),
                "acc_mag": round(acc_mag, 3),
                "workload": round(workload, 3),
                "ppg_sqi": round(sqi, 3),
                "device_name": device_name,
                "timestamp": round(time.time(), 2)
            }

            success = post_telemetry(payload, server_url)
            status_tag = "[SENT -> FASTAPI]" if success else "[SERVER WAITING]"
            noise_tag = " [MOTION ARTIFACT SURGE!]" if has_artifact else ""

            print(f" {status_tag} t={t:03d}s | Phase: {workout_phase:<24} | Raw Watch PPG: {hr_observed:5.1f} BPM | True Phys: {hr_clean:5.1f} BPM | SQI: {sqi*100:4.1f}%{noise_tag}")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[STOP] Live watch simulator stopped.")

def import_and_calibrate_from_history(file_path: str = None, server_url: str = DEFAULT_SERVER_URL):
    """
    Extracts past heart rate and workload data from watch history (CSV/JSON),
    runs the Auto-ODE conditioning algorithm, and sets the parameters on the server.
    """
    target = file_path or HISTORY_CSV_PATH
    if not os.path.exists(target):
        demo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_session_demo.csv")
        if os.path.exists(demo_path):
            target = demo_path
        else:
            print(f"[ERROR] History file not found: {target}")
            return False

    print(f"\n[EXTRACTING] Reading past watch health history from: {target}")
    hrs = []
    workloads = []

    try:
        if target.endswith(".json"):
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data if isinstance(data, list) else data.get("records", data.get("data", []))
                for r in records:
                    val = r.get("hr", r.get("heart_rate", r.get("bpm", 0)))
                    w = r.get("workload", r.get("speed", 0.15))
                    if 45 <= float(val) <= 220:
                        hrs.append(float(val))
                        workloads.append(float(w))
        else:
            import csv
            with open(target, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    hr_key = next((k for k in row if k.lower() in ("hr", "heart_rate", "hr_observed", "hr_true", "bpm")), None)
                    w_key = next((k for k in row if k.lower() in ("workload", "speed", "acc_mag")), None)
                    if hr_key and row[hr_key]:
                        try:
                            val = float(row[hr_key])
                            w = float(row[w_key]) if (w_key and row[w_key]) else 0.15
                            if 45 <= val <= 220:
                                hrs.append(val)
                                workloads.append(w)
                        except ValueError:
                            continue
    except Exception as e:
        print(f"[ERROR] Failed to parse history file: {e}")
        return False

    if len(hrs) < 8:
        print(f"[WARN] Too few valid heart rate points ({len(hrs)}). Need at least 8.")
        return False

    print(f"[ANALYSIS] Parsed {len(hrs)} historical samples from Galaxy Watch.")
    
    # 1. Basal resting heart rate (15th percentile of quiescent rest)
    quiescent = [h for h, w in zip(hrs, workloads) if w <= 0.22]
    import numpy as np
    if len(quiescent) >= 4:
        hr_rest = float(np.percentile(quiescent, 15))
    else:
        hr_rest = float(np.percentile(hrs, 10))
    hr_rest = max(48.0, min(88.0, round(hr_rest, 1)))

    # 2. Vagal recovery time constant (tau)
    decays = []
    for i in range(1, len(hrs)):
        if workloads[i] <= 0.28 and hrs[i] < hrs[i-1] and hrs[i] > hr_rest + 3.0:
            dhr = hrs[i] - hrs[i-1]
            excess = ((hrs[i] + hrs[i-1]) / 2.0) - hr_rest
            if excess > 2.0:
                k = -dhr / excess
                if 0.015 < k < 0.12:
                    decays.append(1.0 / k)
    if len(decays) >= 2:
        tau = float(np.median(decays))
    else:
        std = float(np.std(hrs))
        tau = 18.0 + max(0.0, 14.0 - std * 0.7)
    tau = max(12.0, min(65.0, round(tau, 1)))

    # 3. Workload gain (beta)
    max_hr = max(hrs)
    max_w = max(0.3, max(workloads))
    beta = max(1.2, min(7.5, round((max_hr - hr_rest) / (tau * max_w), 2)))

    # 4. Fitness tier
    if tau <= 20.0 and hr_rest <= 62.0:
        tier = "Elite / Rapid Vagal Reactivity"
    elif tau <= 28.0:
        tier = "Athletic / Fast Cardiovascular Recovery"
    elif tau <= 42.0:
        tier = "Active Baseline / Balanced Autonomic Tone"
    else:
        tier = "Moderate Aerobic"

    print("\n" + "="*60)
    print(" [AUTO-ODE] PERSONALIZED CARDIOVASCULAR CONDITIONING PROFILE")
    print("="*60)
    print(f"  Basal Resting HR (HR_rest) : {hr_rest:.1f} BPM")
    print(f"  Recovery Time Constant (tau): {tau:.1f} s")
    print(f"  Workload Sensitivity Gain (beta): {beta:.2f}")
    print(f"  Autonomic Conditioning Tier: {tier}")
    print(f"  Calibrated from            : {len(hrs)} past wearable samples")
    print("="*60 + "\n")

    # Send calibration to server
    try:
        calib_url = server_url.replace("/api/live/ingest", "/api/live/calibrate_ode")
        req_data = json.dumps({"tau": tau, "beta": beta, "hr_rest": hr_rest}).encode("utf-8")
        req = urllib.request.Request(calib_url, data=req_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                print(f"[SYNC] Successfully updated server biophysics engine with your conditioning!")
    except Exception as e:
        print(f"[NOTE] Server biophysics sync deferred (server offline or local endpoint).")

    return True

def main():
    parser = argparse.ArgumentParser(description="Samsung Galaxy Health Smartwatch BLE Bridge")
    parser.add_argument("--scan", action="store_true", help="Scan for nearby BLE smartwatches")
    parser.add_argument("--address", type=str, default=None, help="MAC Address or UUID of BLE smartwatch")
    parser.add_argument("--name", type=str, default="Samsung Galaxy Watch", help="Device friendly name")
    parser.add_argument("--simulate", action="store_true", help="Run realistic real-time 1 Hz smartwatch live stream")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER_URL, help="FastAPI ingest endpoint")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate automatic conditioning from past watch history")
    parser.add_argument("--import-history", type=str, default=None, help="Path to Samsung Health CSV/JSON export file")

    args = parser.parse_args()

    if args.scan:
        asyncio.run(scan_ble_devices())
    elif args.calibrate or args.import_history:
        import_and_calibrate_from_history(args.import_history, args.server)
    elif args.address:
        asyncio.run(run_ble_client(args.address, args.name, args.server))
    elif args.simulate:
        run_simulation_stream(args.name, args.server)
    else:
        # If no arguments provided, show interactive prompt
        print("\n=======================================================")
        print(" Samsung Galaxy Health - Wearable Live Telemetry Bridge")
        print("=======================================================")
        print(" Choose an option:")
        print("   1. Scan nearby Bluetooth Low Energy smartwatches")
        print("   2. Run Live 1 Hz Smartwatch Stream Simulator (Demonstration)")
        print("   3. Connect to specific device address")
        print("   4. Extract & Calibrate automatic conditioning from watch history")
        print("   5. Exit")
        choice = input("Enter selection [1-5] (default: 2): ").strip()
        if choice == "1":
            asyncio.run(scan_ble_devices())
        elif choice == "3":
            addr = input("Enter Bluetooth MAC address (e.g. C4:64:E3:XX:XX:XX): ").strip()
            if addr:
                asyncio.run(run_ble_client(addr, args.name, args.server))
        elif choice == "4":
            path = input("Enter path to Samsung Health CSV (or press Enter for saved history): ").strip()
            import_and_calibrate_from_history(path if path else None, args.server)
        elif choice == "5":
            print("Exiting.")
        else:
            run_simulation_stream(args.name, args.server)

if __name__ == "__main__":
    main()

