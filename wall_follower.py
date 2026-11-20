import time
import numpy as np
from mbot_bridge.api import MBot


def find_min_dist(ranges, thetas):
    """Returns the length and angle of the shortest valid ray (ignores 0s)."""
    min_dist, min_angle = None, None
    for r, th in zip(ranges, thetas):
        if r <= 0:
            continue  # Zero means a bad reading -- ignore it.
        if min_dist is None or r < min_dist:
            min_dist, min_angle = r, th
    return min_dist, min_angle


def cross_product(v1, v2):
    """Cross product of two length-3 vectors."""
    res = np.zeros(3)
    res[0] = v1[1] * v2[2] - v1[2] * v2[1]
    res[1] = v1[2] * v2[0] - v1[0] * v2[2]
    res[2] = v1[0] * v2[1] - v1[1] * v2[0]
    return res


robot = MBot()
setpoint = 0.5         # Distance to hold from the wall (meters).
margin = 0.05          # Dead zone around the setpoint (meters).
correction_mag = 0.15  # How hard to steer toward/away from the wall.
speed = 0.3            # Driving speed along the wall (m/s).

try:
    while True:
        ranges, thetas = robot.read_lidar()

        # 1. Nearest point on the wall.
        dist_to_wall, angle_to_wall = find_min_dist(ranges, thetas)
        if dist_to_wall is None:
            robot.stop()
            time.sleep(0.1)
            continue

        # 2. Unit vector pointing at the wall (z = 0).
        v_to_wall = [np.cos(angle_to_wall), np.sin(angle_to_wall), 0]

        # 3. Cross with "up" to get a unit vector parallel to the wall.
        #    Swap to cross_product(v_to_wall, [0, 0, 1]) to go the other way.
        v_fwd = cross_product([0, 0, 1], v_to_wall)

        # 4. Bang-bang correction to hold the setpoint.
        error = dist_to_wall - setpoint
        if error > margin:
            correction = correction_mag * np.array(v_to_wall)    # Too far: steer in.
        elif error < -margin:
            correction = -correction_mag * np.array(v_to_wall)   # Too close: steer out.
        else:
            correction = np.zeros(3)

        # 5. Combine, rescale to speed, drive. No spinning.
        cmd = speed * v_fwd + correction
        norm = np.linalg.norm(cmd[:2])
        if norm > 0:
            cmd = cmd / norm * speed
        robot.drive(cmd[0], cmd[1], 0)

        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopping (Ctrl+C).")
finally:
    robot.stop()
