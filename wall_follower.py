import time
import numpy as np
from mbot_bridge.api import MBot


def find_min_dist(ranges, thetas):
    """Finds the length and angle of the minimum ray in the scan.
    Make sure you ignore any rays with length 0! Those are invalid.
    Args:
        ranges (list): The length of each ray in the Lidar scan.
        thetas (list): The angle of each ray in the Lidar scan.
    Returns:
        tuple: The length and angle of the shortest ray in the Lidar scan.
    """
    min_dist, min_angle = None, None
    ranges = np.array(ranges)
    thetas = np.array(thetas)

    # Ignore invalid (zero-length) rays.
    valid_idx = (ranges > 0).nonzero()
    valid_ranges = ranges[valid_idx]
    valid_thetas = thetas[valid_idx]

    if len(valid_ranges) == 0:
        return min_dist, min_angle

    # Find the index of the shortest valid ray.
    min_idx = np.argmin(valid_ranges)
    min_dist = valid_ranges[min_idx]
    min_angle = valid_thetas[min_idx]

    return min_dist, min_angle


def cross_product(v1, v2):
    """Computes the cross product of two 3-element vectors.
    Args:
        v1 (list): The first vector, [x, y, z].
        v2 (list): The second vector, [x, y, z].
    Returns:
        list: The cross product v1 x v2, as [x, y, z].
    """
    x = v1[1] * v2[2] - v1[2] * v2[1]
    y = v1[2] * v2[0] - v1[0] * v2[2]
    z = v1[0] * v2[1] - v1[1] * v2[0]
    return [x, y, z]


robot = MBot()

setpoint = 0.5          # TODO: Pick your setpoint (meters). Distance to hold from the wall.
margin = 0.05            # TODO: Dead zone (meters) around the setpoint where no correction is applied.
correction_mag = 0.15    # TODO: Speed (m/s) of the correction pushing toward/away from the wall.
speed = 0.3              # TODO: Tangential driving speed (m/s) along the wall.

try:
    while True:
        # Read the latest Lidar scan.
        ranges, thetas = robot.read_lidar()

        # Get the distance and angle to the nearest point on the wall.
        dist_to_wall, angle_to_wall = find_min_dist(ranges, thetas)

        if dist_to_wall is not None:
            # Vector from the robot to the nearest wall point.
            wall_vec = [dist_to_wall * np.cos(angle_to_wall),
                        dist_to_wall * np.sin(angle_to_wall),
                        0]

            # Tangent direction along the wall: cross [0, 0, 1] with wall_vec.
            # NOTE: The sign of this cross product sets whether the robot
            # follows the wall clockwise or counterclockwise. If it drives
            # the wrong way, try cross_product(wall_vec, [0, 0, 1]) instead
            # (or negate tangent_vec) -- this depends on the Lidar's angle
            # convention.
            tangent_vec = np.array(cross_product([0, 0, 1], wall_vec)[:2])
            tangent_norm = np.linalg.norm(tangent_vec)
            if tangent_norm > 0:
                tangent_vec = tangent_vec / tangent_norm

            # Unit vector pointing from the robot toward the nearest wall point.
            wall_dir = np.array(wall_vec[:2])
            wall_dir_norm = np.linalg.norm(wall_dir)
            if wall_dir_norm > 0:
                wall_dir = wall_dir / wall_dir_norm

            # Bang-bang correction to hold the setpoint distance from the wall.
            error = dist_to_wall - setpoint
            if error > margin:
                # Too far from the wall -- drive toward it.
                # NOTE: Flip this sign if "toward the wall" ends up being
                # away from it for your Lidar's angle convention.
                correction = correction_mag * wall_dir
            elif error < -margin:
                # Too close to the wall -- drive away from it.
                correction = -correction_mag * wall_dir
            else:
                # Within the margin of the setpoint -- no correction needed.
                correction = np.zeros(2)

            # Combine the tangential (wall-following) motion with the
            # correction, then normalize and scale to the desired speed.
            combined = speed * tangent_vec + correction
            combined_norm = np.linalg.norm(combined)
            if combined_norm > 0:
                combined = (combined / combined_norm) * speed
            vx, vy = combined[0], combined[1]
        else:
            # No valid readings this scan -- don't move.
            vx, vy = 0, 0

        # Command the robot: forward/lateral speed, no rotation.
        robot.drive(vx, vy, 0)

        # Optionally, sleep for a bit before reading a new scan.
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopping (Ctrl+C).")
finally:
    # Always stop the robot, whether we quit or something crashed.
    robot.stop()
