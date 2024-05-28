import json
import os
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional
from hydra import compose, initialize
from PIL import Image
from torchvision import transforms as T

from env_agents import create_env


def way_point_to_pixel(waypoint):
    pixel_val = waypoint / 23.315 * 256
    return int(256 - pixel_val)


class Agent:
    def __init__(self, env_config_path, save_root):
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
        self.magic_number = 23.315

        self.save_root = save_root
        os.makedirs(os.path.join(self.save_root, "front"), exist_ok=True)
        os.makedirs(os.path.join(self.save_root, "bev"), exist_ok=True)
        self.target_traj_num = 15
        self.get_every_n_frame = 2
        self.tota_frame_should_pass = self.target_traj_num * self.get_every_n_frame
        self.save_num = 0

    def run(self):
        state = self.env.reset()
        cur_traj = []
        target_bev = None
        init_compass = 0.0
        big_record = []
        
        while self.save_num < 20:
            input_control = {0: None}
            state, *_ = self.env.step(input_control)
            cur_pos = state["cur_waypoint"][0]
            camera = state["camera"][0]
            bev = state["bev"][0]
            
            if len(cur_traj) == 0:
                save_front_path = os.path.join(self.save_root, "front", f"{self.save_num:06d}.png")
                Image.fromarray(camera).save(save_front_path)
                target_bev = bev
                init_compass = state["compass"][0]
            
            cur_traj.append(cur_pos)
            
            if len(cur_traj) == self.tota_frame_should_pass:
                save_bev_path = os.path.join(self.save_root, "bev", f"{self.save_num:06d}.png")
                added_traj = []
                for traj in cur_traj:
                    theta = init_compass + np.pi / 2
                    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    traj = np.array([traj[0] - cur_traj[0][0], traj[1] - cur_traj[0][1]])
                    traj = R.T.dot(traj).reshape(-1)
                    pixel_x = way_point_to_pixel(traj[1])
                    pixel_y = way_point_to_pixel(-traj[0])
                    target_bev = cv2.circle(
                        target_bev, (int(pixel_x), int(pixel_y)), 3, (0, 255, 0), -1
                    )
                    added_traj.append((traj[1] / self.magic_number, -traj[0] / self.magic_number))
                big_record.append({
                    "traj": added_traj,
                    "image": f"{self.save_num:06d}.png"
                })
                Image.fromarray(target_bev).save(save_bev_path)
                cur_traj.clear()
                self.save_num += 1
        
        with open(os.path.join(self.save_root, "waypoint.json"), "w") as f:
            json.dump(big_record, f)
        self.server_manager.stop()
        print("Finished!")

    def __del__(self):
        self.server_manager.stop()
        print("Finished!")


if __name__ == "__main__":
    agent = Agent(env_config_path="train_rl", save_root="data")
    agent.run()
