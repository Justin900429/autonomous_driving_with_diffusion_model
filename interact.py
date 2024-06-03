import argparse
import math

import cv2
import numpy as np
import torch
import torch.nn.functional
from diffusers.schedulers import DDPMScheduler
from hydra import compose, initialize
from torchvision import transforms as T

from config import create_cfg, merge_possible_with_base, show_config
from control import Controller
from misc.create_agent import create_env
from misc.load_param import copy_parameters
from modeling import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None, type=str)
    return parser.parse_args()


class Agent:
    def __init__(self, cfg):
        with initialize(config_path="configs"):
            env_config = compose(config_name=cfg.ENV.CONFIG_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env, self.server_manager = create_env(env_config)
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.controller = Controller(cfg)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.TRAIN.SAMPLE_STEPS,
            prediction_type=cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE,
            beta_schedule=cfg.TRAIN.NOISE_SCHEDULER.TYPE,
            # For linear only
            beta_start=cfg.TRAIN.NOISE_SCHEDULER.BETA_START,
            beta_end=cfg.TRAIN.NOISE_SCHEDULER.BETA_END,
            clip_sample=False,
        )
        self.model = build_model(cfg)
        if cfg.EVAL.CHECKPOINT:
            weight = torch.load(cfg.EVAL.CHECKPOINT, map_location=self.device)
            self.model.load_state_dict(weight["state_dict"])
            copy_parameters(weight["ema_state_dict"]["shadow_params"], self.model.parameters())
            del weight
            torch.cuda.empty_cache()
            print("weights are laoded")

    @torch.inference_mode()
    def generate_traj(self, image):
        self.model.eval()
        traj_shape = (
            1,
            cfg.MODEL.HORIZON,
            cfg.MODEL.TRANSITION_DIM,
        )
        trajs = torch.randn(traj_shape, device=self.device)
        image = image.to(self.device)

        self.noise_scheduler.set_timesteps(self.cfg.EVAL.SAMPLE_STEPS, device=self.device)
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(
                trajs,
                image,
                t.reshape(-1),
            )
            trajs = self.noise_scheduler.step(model_output, t, trajs).prev_sample
        trajs = trajs.to(torch.float32).clamp(-1, 1) * self.model.magic_number
        return trajs

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

        target_point = torch.FloatTensor([[local_command_point[0], local_command_point[1]]]).to(
            self.device
        )
        return target_point

    def process_next_command(self, command):
        if command < 0:
            command = 4
        command -= 1
        one_hot_command = torch.nn.functional.one_hot(torch.tensor([command]), num_classes=6).to(
            self.device, dtype=torch.float32
        )

        return one_hot_command

    def process_speed(self, speed):
        speed = torch.FloatTensor([[speed]]).to(self.device)
        return speed

    def process_control(self, pred, state, target_point):
        gt_velocity = torch.FloatTensor([state["state"][0][0]]).to(self.device, dtype=torch.float32)
        steer_res, throttle_res, brake_res = self.controller.control_pid(
            pred, gt_velocity, target_point
        )

        if brake_res < 0.05:
            brake_res = 0.0
        if throttle_res > brake_res:
            brake_res = 0.0

        if brake_res > 0.5:
            throttle_res = float(0)

        num = 0
        for s in self.last_steers:
            num += s > 0.10

        return np.array([throttle_res, steer_res, brake_res])

    def run(self):
        state = self.env.reset()
        done = False

        while True:
            image = self.process_image(state["camera"][0])
            target_point = self.process_next_waypoint(
                next_point=state["next_waypoint"],
                cur_point=state["cur_waypoint"],
                yaw=state["state"][0][1],
            )

            traj = self.generate_traj(image)
            control = self.process_control(traj, state, target_point)
            input_control = {0: control}
            state, *_ = self.env.step(input_control)
            if done:
                break
        self.server_manager.stop()
        print("Finished!")

    def __del__(self):
        self.server_manager.stop()
        print("Finished!")


if __name__ == "__main__":
    args = parse_args()
    cfg = create_cfg()
    if args.config is not None:
        merge_possible_with_base(cfg, args.config)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    show_config(cfg)

    agent = Agent(cfg)
    agent.run()
