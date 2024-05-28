import math
from collections import OrderedDict, deque

import cv2
import numpy as np
import torch
import torch.nn.functional
from hydra import compose, initialize
from torchvision import transforms as T

from e2e_driving.config import GlobalConfig
from e2e_driving.model import Planner
from env_agents import create_env


class Agent:
    def __init__(self, env_config_path, model_checkpoint):
        with initialize(config_path="../config"):
            cfg = compose(config_name=env_config_path)
        self.env, self.server_manager = create_env(cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.status = 0
        self.alpha = 0.3
        self.steer_step = 0
        self.last_steers = deque()

    def process_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image).unsqueeze(0).to(self.device)
        return image

    def process_next_waypoint(self, next_point, cur_point, yaw):
        if math.isnan(yaw):
            yaw = 0.0

        rad = yaw * math.pi / 180.0
        rad = rad + math.pi / 2.0
        R = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])

        local_command_point = np.array(
            [next_point[0][0] - cur_point[0][0], next_point[0][1] - cur_point[0][1]]
        )
        local_command_point = R.T.dot(local_command_point)

        target_point = torch.FloatTensor(
            [[local_command_point[0], local_command_point[1]]]
        ).to(self.device)
        return target_point

    def process_next_command(self, command):
        if command < 0:
            command = 4
        command -= 1
        one_hot_command = torch.nn.functional.one_hot(
            torch.tensor([command]), num_classes=6
        ).to(self.device, dtype=torch.float32)

        return one_hot_command

    def process_speed(self, speed):
        speed = torch.FloatTensor([[speed]]).to(self.device)
        return speed

    def process_control(self, pred, state, target_point):
        gt_velocity = torch.FloatTensor([state["state"][0][0]]).to(
            self.device, dtype=torch.float32
        )
        steer_ctrl, throttle_ctrl, brake_ctrl, _ = self.net.process_action(
            pred, state["next_command"], gt_velocity, target_point
        )

        steer_traj, throttle_traj, brake_traj, _ = self.net.control_pid(
            pred["pred_wp"], gt_velocity, target_point
        )

        if brake_traj < 0.05:
            brake_traj = 0.0
        if throttle_traj > brake_traj:
            brake_traj = 0.0

        if self.status == 0:
            self.alpha = 0.3
            throttle_res = np.clip(
                self.alpha * throttle_ctrl + (1 - self.alpha) * throttle_traj, 0, 0.75
            )
            steer_res = np.clip(
                self.alpha * steer_ctrl + (1 - self.alpha) * steer_traj, -1, 1
            )
            brake_res = np.clip(
                self.alpha * brake_ctrl + (1 - self.alpha) * brake_traj, 0, 1
            )
        else:
            self.alpha = 0.3
            throttle_res = np.clip(
                self.alpha * throttle_traj + (1 - self.alpha) * throttle_ctrl, 0, 0.75
            )
            steer_res = np.clip(
                self.alpha * steer_traj + (1 - self.alpha) * steer_ctrl, -1, 1
            )
            brake_res = np.clip(
                self.alpha * brake_traj + (1 - self.alpha) * brake_ctrl, 0, 1
            )

        if brake_res > 0.5:
            throttle_res = float(0)

        num = 0
        for s in self.last_steers:
            num += s > 0.10
        if num > 10:
            self.status = 1
            self.steer_step += 1
        else:
            self.status = 0

        return np.array([throttle_res, steer_res, brake_res])

    def run(self):
        state = self.env.reset()
        done = False

        while True:
            # image = self.process_image(state["camera"][0])
            # one_hot_command = self.process_next_command(state["next_command"][0])
            # target_point = self.process_next_waypoint(
            #     next_point=state["next_waypoint"],
            #     cur_point=state["cur_waypoint"],
            #     yaw=state["state"][0][1],
            # )
            # speed = self.process_speed(state["state"][0][0])

            # input_state = torch.cat(
            #     [speed / 12.0, target_point, one_hot_command], dim=1
            # )
            # with torch.no_grad():
            #     pred = self.net(image, input_state, target_point)
            # control = self.process_control(pred, state, target_point)
            input_control = {0: None}
            state, reward, done, *_ = self.env.step(input_control)
            
            camera = state["camera"]
            bev = state["bev"]
            
            cv2.imwrite("camera.png", camera[0])
            # if done:
            #     break
        self.server_manager.stop()
        print("Finished!")

    def __del__(self):
        self.server_manager.stop()
        print("Finished!")


if __name__ == "__main__":
    agent = Agent(env_config_path="train_rl", model_checkpoint="log/model.pth")
    agent.run()
