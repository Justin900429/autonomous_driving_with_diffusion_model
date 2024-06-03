import numpy as np

from .pid import PIDController


class Controller:
    def __init__(self, cfg):
        self.turn_controller = PIDController(
            K_P=cfg.PID.TURN_KP,
            K_I=cfg.PID.TURN_KI,
            K_D=cfg.PID.TURN_KD,
            n=cfg.PID.TURN_N,
        )
        self.speed_controller = PIDController(
            K_P=cfg.PID.SPEED_KP,
            K_I=cfg.PID.SPEED_KI,
            K_D=cfg.PID.SPEED_KD,
            n=cfg.PID.SPEED_N,
        )

        self.aim_dist = cfg.CONTROL.AIM_DIST
        self.angle_thresh = cfg.CONTROL.ANGLE_THRESH
        self.dist_thresh = cfg.CONTROL.DIST_THRESH
        self.brake_speed = cfg.CONTROL.BRAKE_SPEED
        self.brake_ratio = cfg.CONTROL.BRAKE_RATIO
        self.clip_delta = cfg.CONTROL.CLIP_DELTA
        self.max_throttle = cfg.CONTROL.MAX_THROTTLE

    def control_pid(self, waypoints, velocity, target):
        waypoints = waypoints[0].data.cpu().numpy()
        target = target.squeeze().data.cpu().numpy()
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            desired_speed += np.linalg.norm(waypoints[i + 1] - waypoints[i]) * 2.0 / num_pairs

            norm = np.linalg.norm((waypoints[i + 1] + waypoints[i]) / 2.0)
            if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        use_target_to_aim = use_target_to_aim or (
            np.abs(angle_target - angle_last) > self.angle_thresh and target[1] < self.dist_thresh
        )
        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        speed = velocity[0].data.cpu().numpy()
        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        return steer, throttle, brake
