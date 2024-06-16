import argparse
import glob
import os
import time

import carla
import cv2
import numpy as np
import torch
import torch.nn.functional
from create_agent import create_env, create_server
from hydra import compose, initialize
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Data Collection")
    parser.add_argument("--save-path", default="data", type=str, help="Path to save the data")
    parser.add_argument(
        "--config-path", default="data_collect", type=str, help="Path to the config file"
    )
    parser.add_argument("--save-num", default=5000, type=int, help="The number of data to save")
    parser.add_argument(
        "--save-every-n-frame", default=2, type=int, help="Save the data every n frames"
    )
    parser.add_argument(
        "--off-screen",
        default=False,
        action="store_true",
        help="Run the simulation off-screen",
    )
    return parser.parse_args()


def get_random_seed():
    t = int(time.time() * 1000.0)
    t = (
        ((t & 0xFF000000) >> 24)
        + ((t & 0x00FF0000) >> 8)
        + ((t & 0x0000FF00) << 8)
        + ((t & 0x000000FF) << 24)
    )
    return t


def way_point_to_pixel(waypoint):
    pixel_val = waypoint / 23.315 * 256
    return int(256 - pixel_val)


class Agent:
    def __init__(
        self,
        env_config_path,
        save_root,
        total_to_save,
        save_every_n_frame,
        step_to_reset,
        off_screen,
        seed,
    ):
        with initialize(config_path="../configs", version_base="1.3.2"):
            cfg = compose(config_name=env_config_path)
        self.server_manager = create_server(cfg, off_screen)
        self.env = create_env(cfg, self.server_manager, seed)
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.magic_number = 23.315

        self.save_root = save_root
        os.makedirs(os.path.join(self.save_root, "front"), exist_ok=True)
        os.makedirs(os.path.join(self.save_root, "bev"), exist_ok=True)
        os.makedirs(os.path.join(self.save_root, "waypoints"), exist_ok=True)
        self.target_traj_num = 16
        self.total_frame_should_pass = self.target_traj_num
        self.total_to_save = total_to_save
        save_front = len(list(glob.glob(os.path.join(self.save_root, "front/*.png"))))
        save_bev = len(list(glob.glob(os.path.join(self.save_root, "bev/*.png"))))
        save_waypoints = len(list(glob.glob(os.path.join(self.save_root, "waypoints/*.txt"))))
        self.cur_save = min(save_front, save_bev, save_waypoints)
        self.save_every_n_frame = save_every_n_frame
        self.buffer_frames = 50
        self.step_to_reset = step_to_reset
        self.car_agent = None

    def do_buffer(self, num_buffer):
        for _ in range(num_buffer):
            self.env.step({0: None})

    def get_car_agent(self):
        ev_handler = self.env.envs[0].env.unwrapped.ev_handler
        if ev_handler is not None:
            self.car_agent = ev_handler.ego_vehicles["hero"].vehicle

    def world_to_agent(self, world_pos, agent_pos, agent_yaw):
        x, y = world_pos
        x -= agent_pos[0]
        y -= agent_pos[1]
        theta = agent_yaw
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        x, y = R.T.dot(np.array([x, y])).reshape(-1)
        return x, y

    def run(self):
        state = self.env.reset()
        self.get_car_agent()
        cur_traj = []
        target_bev = None
        init_compass = 0.0
        target_pos = None
        prev_red = False
        count_to_collect = 0
        step_to_reset = 0

        self.do_buffer(self.buffer_frames)

        while self.cur_save < self.total_to_save:
            if not prev_red:
                input_control = {0: None}
            else:
                input_control = {0: np.array([0.0, 0.0, 1.0])}
            state, _, done, *_ = self.env.step(input_control)
            cur_pos = state["cur_waypoint"][0]
            cur_control = state["state"][0][:5]
            cur_control[0] = cur_control[0] / 180  # yaw
            cur_control[1] = cur_control[1] / self.cfg.envs.env_configs.target_speed  # speed
            camera = state["camera"][0]
            bev = state["bev"][0]

            # If the episode is done, clear the current trajectory to avoid inconsistency
            if done:
                cur_traj.clear()
                count_to_collect = 0
                step_to_reset = 0
                self.do_buffer(self.buffer_frames)
                self.get_car_agent()
                continue

            if state["at_red_light"][0] == 1 and prev_red:
                continue

            if count_to_collect % self.save_every_n_frame != 0:
                count_to_collect += 1
                continue

            if len(cur_traj) == 0:
                save_front_path = os.path.join(self.save_root, "front", f"{self.cur_save:06d}.png")
                Image.fromarray(camera).save(save_front_path)
                target_bev = np.copy(bev)
                init_compass = state["compass"][0]
                target_pos = state["next_waypoint"][0]

                if state["at_red_light"][0] == 1:
                    for _ in range(self.total_frame_should_pass):
                        cur_traj.append(
                            np.concatenate([cur_pos, np.array([0.0, 0.0, 0.0, 0.0, 1.0])])
                        )
                        prev_red = True
                else:
                    prev_red = False

            if len(cur_traj) < self.total_frame_should_pass + 1:
                cur_traj.append(np.concatenate((cur_pos, cur_control)))

            if len(cur_traj) != self.total_frame_should_pass + 1:
                count_to_collect += 1
            else:
                save_bev_path = os.path.join(self.save_root, "bev", f"{self.cur_save:06d}.png")
                added_traj = []
                theta = init_compass + np.pi / 2
                for idx in range(len(cur_traj) - 1):
                    traj = np.copy(cur_traj[idx][:2])
                    car_state = np.copy(cur_traj[idx][2:4])
                    action = np.copy(cur_traj[idx + 1][-3:])
                    car_state[0] -= cur_traj[0][2]
                    if car_state[0] > 1:
                        car_state[0] -= 1
                    elif car_state[0] < -1:
                        car_state[0] += 1
                    traj = self.world_to_agent(traj, cur_traj[0][:2], theta)
                    pixel_x = way_point_to_pixel(traj[1])
                    pixel_y = way_point_to_pixel(-traj[0])
                    target_bev = cv2.circle(
                        target_bev, (int(pixel_x), int(pixel_y)), 3, (0, 255, 0), -1
                    )
                    added_traj.append(
                        (
                            traj[1] / self.magic_number,
                            -traj[0] / self.magic_number,
                            *car_state.tolist(),
                            *action.tolist(),
                        )
                    )
                target_pos_traj = self.world_to_agent(target_pos, cur_traj[0][:2], theta)
                with open(
                    os.path.join(self.save_root, "waypoints", f"{self.cur_save:06d}.txt"), "w"
                ) as f:
                    f.write(
                        f"{target_pos_traj[1] / self.magic_number} {-target_pos_traj[0] / self.magic_number}\n"
                    )
                    for traj in added_traj:
                        f.write(f"{' '.join(map(str, traj))}\n")
                Image.fromarray(target_bev).save(save_bev_path)
                cur_traj.clear()
                self.cur_save += 1
                count_to_collect = 0

                if (
                    prev_red is True
                    and self.car_agent is not None
                    and self.car_agent.is_alive
                    and self.car_agent.is_at_traffic_light()
                ):
                    traffic_light = self.car_agent.get_traffic_light()
                    if (traffic_light.get_state() == carla.TrafficLightState.Red) or (
                        traffic_light.get_state() == carla.TrafficLightState.Yellow
                    ):
                        traffic_light.set_state(carla.TrafficLightState.Green)
                    continue

                if step_to_reset > self.step_to_reset:
                    self.env.reset()
                    self.get_car_agent()
                    step_to_reset = 0

                # Skip the next 50 frames to increase the diversity of the data
                self.do_buffer(self.buffer_frames)
            step_to_reset += 1

        self.server_manager.stop()
        print("Finished!")

    def __del__(self):
        self.server_manager.stop()
        print("Finished!")


if __name__ == "__main__":
    args = parse_args()
    agent = Agent(
        env_config_path=args.config_path,
        save_root=args.save_path,
        total_to_save=args.save_num,
        off_screen=args.off_screen,
        save_every_n_frame=args.save_every_n_frame,
        step_to_reset=args.save_every_n_frame * 100,
        seed=get_random_seed(),
    )
    agent.run()
