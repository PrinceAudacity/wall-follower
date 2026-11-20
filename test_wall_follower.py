"""Offline test for wall_follower.py -- the robot never moves.

Run:  python3 test_wall_follower.py

A fake MBot feeds wall_follower.py a simulated Lidar scan of a flat wall on
the robot's LEFT and records every drive()/stop() call.
"""
import runpy
import sys
import time
import types

import numpy as np

N_SCANS = 5
N_RAYS = 360
MAX_RANGE = 10.0  # Rays longer than this count as "no hit" (0).


def make_scan(wall_dist):
    """Flat wall on the left (y = wall_dist). Theta 0 = front, CCW positive."""
    thetas = np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False)
    ranges = np.zeros(N_RAYS)
    s = np.sin(thetas)
    hit = s > 1e-6
    ranges[hit] = wall_dist / s[hit]
    ranges[ranges > MAX_RANGE] = 0
    # Bad reading: zero out the ray pointing straight at the wall.
    ranges[np.argmin(np.abs(thetas - np.pi / 2))] = 0
    return ranges.tolist(), thetas.tolist()


def run_case(wall_dist):
    calls = {"drive": [], "stop": 0, "scans": 0}

    class FakeMBot:
        def read_lidar(self):
            if calls["scans"] >= N_SCANS:
                raise KeyboardInterrupt
            calls["scans"] += 1
            return make_scan(wall_dist)

        def drive(self, vx, vy, wz):
            calls["drive"].append((float(vx), float(vy), float(wz)))

        def stop(self):
            calls["stop"] += 1

    api = types.ModuleType("mbot_bridge.api")
    api.MBot = FakeMBot
    pkg = types.ModuleType("mbot_bridge")
    pkg.api = api
    sys.modules["mbot_bridge"] = pkg
    sys.modules["mbot_bridge.api"] = api

    real_sleep = time.sleep
    time.sleep = lambda *_: None
    try:
        runpy.run_path("wall_follower.py", run_name="__main__")
    finally:
        time.sleep = real_sleep
    return calls


def check(wall_dist, expect):
    calls = run_case(wall_dist)
    problems = []
    if len(calls["drive"]) != N_SCANS:
        problems.append(f"expected {N_SCANS} drive calls, got {len(calls['drive'])}")
    for vx, vy, wz in calls["drive"]:
        sp = np.hypot(vx, vy)
        if abs(sp - 0.3) > 0.01:
            problems.append(f"speed {sp:.3f} != 0.3")
        if wz != 0:
            problems.append(f"wz {wz} != 0")
        if expect == "zero" and abs(vy) > 0.02:
            problems.append(f"vy {vy:.3f} not ~0")
        if expect == "pos" and not vy > 0:
            problems.append(f"vy {vy:.3f} not > 0")
        if expect == "neg" and not vy < 0:
            problems.append(f"vy {vy:.3f} not < 0")
    if calls["stop"] < 1:
        problems.append("stop() never called")
    vx, vy, wz = calls["drive"][-1] if calls["drive"] else (0, 0, 0)
    status = "PASS" if not problems else "FAIL: " + "; ".join(sorted(set(problems)))
    print(f"wall at {wall_dist} m: vx={vx:+.3f} vy={vy:+.3f} wz={wz:+.1f} "
          f"stop() x{calls['stop']}  -> {status}")
    return not problems


results = [check(0.5, "zero"), check(0.9, "pos"), check(0.2, "neg")]
print("\nALL PASSED" if all(results) else f"\n{results.count(False)} FAILED")
sys.exit(0 if all(results) else 1)
