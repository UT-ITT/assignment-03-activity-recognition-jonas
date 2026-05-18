"""
gather_data.py — captures accelerometer and gyroscope data from DIPPID device
Usage: python gather_data.py <name> <activity> <number>
  e.g. python gather_data.py jonas running 1
Press any DIPPID button (1–4) to start a 10-second recording.
"""

import sys
import time
import pandas as pd
from DIPPID import SensorUDP

PORT = 5700
DURATION = 10       # seconds per recording
TARGET_HZ = 100     # resample rate

# DIPPID may send bools, ints, or strings from JSON; []/None before first packet.
BUTTON_KEYS = ("button_1", "button_2", "button_3", "button_4")


def _button_pressed(raw) -> bool:
    if raw is None or raw == []:
        return False
    if isinstance(raw, dict):
        for k in ("pressed", "value", "state"):
            if k in raw:
                return _button_pressed(raw[k])
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "on", "yes")
    if raw is True or raw is False:
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    return bool(raw)


def main():
    if len(sys.argv) != 4:
        print("Usage: python gather_data.py <name> <activity> <number>")
        print("  e.g. python gather_data.py jonas running 1")
        sys.exit(1)

    name, activity, number = sys.argv[1], sys.argv[2], sys.argv[3]
    filename = f"data/{name}-{activity}-{number}.csv"

    sensor = SensorUDP(PORT)
    print(f"Listening on UDP port {PORT}...")
    print(f"Will save to: {filename}")
    print("Press any DIPPID button (1–4) to start recording.")

    recording = False
    samples = []
    start_time = None
    # last gyro dict for packets without gyro; reset when a new recording starts
    last_gyro = {"x": 0.0, "y": 0.0, "z": 0.0}

    def on_button(data):
        nonlocal recording, start_time, samples, last_gyro
        if not recording and _button_pressed(data):
            print(f"Recording started! Perform '{activity}' for {DURATION} seconds...")
            recording = True
            start_time = time.time()
            samples = []
            last_gyro = {"x": 0.0, "y": 0.0, "z": 0.0}

    for _btn in BUTTON_KEYS:
        sensor.register_callback(_btn, on_button)

    # DIPPID only calls callbacks when a value *changes*. A still sensor sends no
    # accelerometer callbacks, so we must poll on a fixed clock while recording.
    sample_period = 1.0 / TARGET_HZ

    def _vec3(d):
        if isinstance(d, dict) and all(k in d for k in ("x", "y", "z")):
            return float(d["x"]), float(d["y"]), float(d["z"])
        return None

    try:
        while True:
            if not recording:
                for btn_name in BUTTON_KEYS:
                    v = sensor.get_value(btn_name)
                    if _button_pressed(v):
                        on_button(v)
                        break
                time.sleep(0.001)
                continue

            if time.time() - start_time >= DURATION:
                break

            acc = sensor.get_value("accelerometer")
            gyro = sensor.get_value("gyroscope")
            g3 = _vec3(gyro)
            if g3 is not None:
                last_gyro["x"], last_gyro["y"], last_gyro["z"] = g3

            a3 = _vec3(acc)
            if a3 is not None:
                elapsed = time.time() - start_time
                samples.append({
                    "timestamp": elapsed,
                    "acc_x": a3[0],
                    "acc_y": a3[1],
                    "acc_z": a3[2],
                    "gyro_x": last_gyro["x"],
                    "gyro_y": last_gyro["y"],
                    "gyro_z": last_gyro["z"],
                })

            time.sleep(sample_period)

    except KeyboardInterrupt:
        print("Interrupted.")
        sensor.disconnect()
        sys.exit(0)

    sensor.disconnect()

    if not samples:
        print("No data captured! (Kein Accelerometer mit x/y/z im UDP-Stream während der Aufnahme.)")
        sys.exit(1)

    # Build DataFrame and resample to TARGET_HZ
    df = pd.DataFrame(samples)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp")

    # Create uniform time index at TARGET_HZ
    t_end = df.index.max()
    uniform_index = pd.RangeIndex(start=0, stop=int(t_end * TARGET_HZ) + 1)
    uniform_times = [i / TARGET_HZ for i in uniform_index]

    # Merge original timestamps with uniform grid, interpolate
    all_times = sorted(set(df.index.tolist() + uniform_times))
    df_reindexed = df.reindex(all_times).interpolate(method="index")
    df_resampled = df_reindexed.loc[uniform_times].reset_index(drop=True)
    df_resampled.index.name = "id"
    df_resampled = df_resampled.reset_index()
    df_resampled.insert(1, "timestamp", uniform_times)

    df_resampled.to_csv(filename, index=False)
    print(f"Saved {len(df_resampled)} samples to {filename}")


if __name__ == "__main__":
    main()
