import argparse
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional
from diffusers.schedulers import (DDIMScheduler, DDPMScheduler,
                                  DPMSolverMultistepScheduler)
from hydra import compose, initialize
from torchvision import transforms as T

from config import create_cfg, merge_possible_with_base, show_config
from control import Controller, GuidanceLoss
from misc.create_agent import create_env, create_server
from misc.load_param import copy_parameters
from modeling import build_model

SCHEDULER_FUNC = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "dpm": DPMSolverMultistepScheduler,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--save-bev-path", default=None, type=str)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None, type=str)
    return parser.parse_args()


def way_point_to_pixel(waypoint):
    pixel_val = waypoint * 256
    return int(256 - pixel_val)


def get_random_seed():
    t = int(time.time() * 1000.0)
    t = (
        ((t & 0xFF000000) >> 24)
        + ((t & 0x00FF0000) >> 8)
        + ((t & 0x0000FF00) << 8)
        + ((t & 0x000000FF) << 24)
    )
    return t


class Agent:
    def __init__(self, cfg, bev_save_path=None, off_screen=False, seed=None):
        with initialize(config_path="configs", version_base="1.3.2"):
            env_config = compose(config_name=cfg.ENV.CONFIG_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.server_manager = create_server(env_config, off_screen)
        self.env = create_env(
            env_config, self.server_manager, get_random_seed() if seed is None else seed
        )
        self.cfg = cfg
        
        self.use_guidance = self.cfg.GUIDANCE.LOSS_LIST is not None
        if self.use_guidance:
            self.guidance_loss = GuidanceLoss(cfg)
            
        self.bev_save_path = bev_save_path
        if bev_save_path:
            os.makedirs(bev_save_path, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.controller = Controller(cfg)
        scheduler_kwargs = dict(
            num_train_timesteps=cfg.TRAIN.SAMPLE_STEPS,
            prediction_type=cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE,
            beta_schedule=cfg.TRAIN.NOISE_SCHEDULER.TYPE,
            # For linear only
            beta_start=cfg.TRAIN.NOISE_SCHEDULER.BETA_START,
            beta_end=cfg.TRAIN.NOISE_SCHEDULER.BETA_END,
            # thresholding=True,
        )

        if cfg.EVAL.SCHEDULER == "dpm":
            scheduler_kwargs["lambda_min_clipped"] = -5.1
        self.noise_scheduler = SCHEDULER_FUNC[cfg.EVAL.SCHEDULER](**scheduler_kwargs)
        self.model = build_model(cfg).to(self.device)
        if cfg.EVAL.CHECKPOINT:
            weight = torch.load(cfg.EVAL.CHECKPOINT, map_location=self.device)
            self.model.load_state_dict(weight["state_dict"])
            copy_parameters(
                weight["ema_state_dict"]["shadow_params"], self.model.parameters()
            )
            del weight
            torch.cuda.empty_cache()
            print("weights are loaded")

    def generate_traj(self, image, yaw, target=None):
        self.model.eval()
        traj_shape = (
            1,
            cfg.MODEL.HORIZON,
            cfg.MODEL.TRANSITION_DIM,
        )
        trajs = torch.randn(traj_shape, device=self.device)
        image = image.to(self.device)
        
        trajs[:, 0, :2] = 0.
        trajs[:, 0, 3] = yaw

        self.noise_scheduler.set_timesteps(
            self.cfg.EVAL.SAMPLE_STEPS, device=self.device
        )
        prev_t = None
        for t in self.noise_scheduler.timesteps:
            with torch.no_grad():
                model_output = self.model(
                    trajs,
                    image,
                    t.reshape(-1),
                )
            if t > 0 and prev_t is not None and self.use_guidance:
                posterior_variance = self.noise_scheduler._get_variance(t, prev_t)
                model_std = torch.exp(0.5 * posterior_variance)
                model_output = self.guidance_loss(model_output, target, model_std)
            trajs = self.noise_scheduler.step(model_output, t, trajs).prev_sample
            trajs[:, 0, :2] = 0.
            trajs[:, 0, 3] = yaw
            prev_t = t
            
        trajs = trajs.to(torch.float32).clamp(-1, 1) * self.model.magic_num
        trajs[..., 0] *= -1

        return trajs

    def process_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image).unsqueeze(0).to(self.device)
        return image

    def process_next_waypoint(self, next_point, cur_point, yaw):
        if math.isnan(yaw):
            yaw = 0.0

        yaw = yaw + math.pi / 2.0
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

        local_command_point = np.array(
            [next_point[0][0] - cur_point[0][0], next_point[0][1] - cur_point[0][1]]
        )
        local_command_point = R.T.dot(local_command_point).reshape(-1)

        target_point = torch.FloatTensor(
            [local_command_point[1] / self.model.magic_num, -local_command_point[0] / self.model.magic_num]
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

    def process_control(self, traj, state, target_point):
        gt_velocity = torch.FloatTensor([state["state"][0][0]]).to(
            self.device, dtype=torch.float32
        )
        steer_res, throttle_res, brake_res = self.controller.control_pid(
            traj[:, :-1, :2], gt_velocity, target_point
        )

        if brake_res < 0.05:
            brake_res = 0.0
        if throttle_res > brake_res:
            brake_res = 0.0

        if brake_res > 0.5:
            throttle_res = float(0)

        return np.array([throttle_res, steer_res, brake_res])

    def plot_to_bev(self, bev_image, traj, filename="test.jpg"):
        for x, y in traj:
            x, y = x / self.model.magic_num * -1, y / self.model.magic_num
            pixel_x = way_point_to_pixel(x.item())
            pixel_y = way_point_to_pixel(y.item())
            bev_image = cv2.circle(bev_image, (pixel_x, pixel_y), 3, (0, 0, 255), -1)
        cv2.imwrite(
            filename, cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR)
        )

    def run(self):
        state = self.env.reset()
        done = False
        count = 0

        while True:
            image = self.process_image(state["camera"][0])
            if self.use_guidance:
                target_point = self.process_next_waypoint(
                    next_point=state["next_waypoint"],
                    cur_point=state["cur_waypoint"],
                    yaw=state["compass"][0],
                )
            bev_image = state["bev"][0]
            yaw = state["compass"][0] + math.pi / 2.0
            traj = self.generate_traj(image, yaw, target_point if self.use_guidance else None)
            if self.bev_save_path:
                self.plot_to_bev(bev_image, traj[0, :, :2], f"{self.bev_save_path}/bev_{count}.jpg")
                count += 1
            if traj.size(-1) > 2:
                control = traj[0][0][-3:].cpu().numpy()
            else:
                control = self.process_control(traj, state, target_point if self.use_guidance else traj[0][-1])
            input_control = {0: control}
            state, *_ = self.env.step(input_control)
            if done:
                break
        self.server_manager.stop()

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

    agent = Agent(cfg, args.save_bev_path)
    agent.run()
