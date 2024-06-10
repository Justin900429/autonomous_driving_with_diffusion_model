import argparse
import math
import os
import time

import carla
import cv2
import numpy as np
import torch
import torch.nn.functional
from hydra import compose, initialize
from torchvision import transforms as T

from config import create_cfg, merge_possible_with_base, show_config
from control import Controller
from misc.constant import GuidanceType
from misc.create_agent import create_env, create_server
from misc.load_param import copy_parameters
from modeling import build_model
from scheduler import GuidanceDDIMScheduler, GuidanceDDPMScheduler

SCHEDULER_FUNC = {
    "ddpm": GuidanceDDPMScheduler,
    "ddim": GuidanceDDIMScheduler,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--plot-on-world", default=False, action="store_true")
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
    def __init__(self, cfg, plot_on_world=False, bev_save_path=None, off_screen=False, seed=None):
        with initialize(config_path="configs", version_base="1.3.2"):
            env_config = compose(config_name=cfg.ENV.CONFIG_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.server_manager = create_server(env_config, off_screen)
        self.env = create_env(
            env_config, self.server_manager, get_random_seed() if seed is None else seed
        )
        self.env_injector = self.env.envs[0].env.unwrapped
        self.car_agent = None
        self.cfg = cfg

        self.plot_on_world = plot_on_world
        self.bev_save_path = bev_save_path
        if bev_save_path is not None:
            os.makedirs(bev_save_path, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.use_guidance_type = GuidanceType[cfg.GUIDANCE.USE_COND]
        self.controller = Controller(cfg)
        scheduler_kwargs = dict(
            num_train_timesteps=cfg.TRAIN.SAMPLE_STEPS,
            prediction_type=cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE,
            beta_schedule=cfg.TRAIN.NOISE_SCHEDULER.TYPE,
            # For linear only
            beta_start=cfg.TRAIN.NOISE_SCHEDULER.BETA_START,
            beta_end=cfg.TRAIN.NOISE_SCHEDULER.BETA_END,
            thresholding=True,
            cfg=cfg,
        )

        if cfg.EVAL.SCHEDULER == "dpm":
            scheduler_kwargs["lambda_min_clipped"] = -5.1
        self.noise_scheduler = SCHEDULER_FUNC[cfg.EVAL.SCHEDULER](**scheduler_kwargs)
        traj_shape = (
            1,
            cfg.MODEL.HORIZON,
            cfg.MODEL.TRANSITION_DIM,
        )
        self.init_trajs = torch.randn(traj_shape, device=self.device)
        self.model = build_model(cfg).to(self.device)
        if cfg.EVAL.CHECKPOINT:
            weight = torch.load(cfg.EVAL.CHECKPOINT, map_location=self.device)
            self.model.load_state_dict(weight["state_dict"])
            copy_parameters(weight["ema_state_dict"]["shadow_params"], self.model.parameters())
            del weight
            torch.cuda.empty_cache()
            print("weights are loaded")

    def get_car_agent(self):
        ev_handler = self.env_injector.ev_handler
        if ev_handler is not None:
            self.car_agent = ev_handler.ego_vehicles["hero"]

    def generate_traj(self, image, target=None):
        self.model.eval()
        trajs = self.init_trajs.clone().detach()
        image = image.to(self.device)
        if target is not None and self.use_guidance_type == GuidanceType.FREE_GUIDANCE:
            target = target.repeat(trajs.size(0), 1) if target is not None else None
            target = torch.cat(
                [
                    target,
                    torch.zeros_like(target),
                ],
                dim=0,
            )

        trajs[:, 0, :3] = 0.0
        self.noise_scheduler.set_timesteps(self.cfg.EVAL.SAMPLE_STEPS, device=self.device)
        action = None
        for t in self.noise_scheduler.timesteps:
            input_trajs = torch.cat([trajs, trajs], dim=0)  # [B * 2, H, 7]
            with torch.no_grad():
                model_output_with_cond, model_output_without_cond = self.model(
                    input_trajs,
                    image,
                    t.reshape(-1),
                    cond=target,
                    return_action_only=self.use_guidance_type == GuidanceType.CLASSIFIER_GUIDANCE,
                ).chunk(2, dim=0)
            model_output = model_output_without_cond + self.cfg.GUIDANCE.FREE_SCALE * (
                model_output_with_cond - model_output_without_cond
            )  # If no free guidance apply, this will be the same as model_output_without_cond
            if self.use_guidance_type == GuidanceType.CLASSIFIER_GUIDANCE:
                action = model_output
                if not action.requires_grad:
                    action.requires_grad_()
                time_embed = self.model.time_mlp(t.reshape(-1))
                state = self.model.state_pred(action[:, :-1], time_embed)
                state = torch.cat([torch.zeros_like(state[:, :1]), state], dim=1)
                model_output = torch.cat([state, action], dim=-1)
            trajs = self.noise_scheduler.step(
                model_output, t, trajs, target=target, action=action
            ).prev_sample
            trajs[:, 0, :3] = 0.0

        trajs = trajs.to(torch.float32).clamp(-1, 1)
        trajs[..., :2] *= self.model.magic_num
        return trajs

    def process_image(self, image):
        image = self.image_transform(image).unsqueeze(0).to(self.device)
        return image

    def get_future_waypoints(self, num_points=16):
        self.get_car_agent()
        waypoints = self.car_agent._global_route[1 : num_points + 1]
        waypoints = np.array(
            [
                [waypoint[0].transform.location.x, waypoint[0].transform.location.y]
                for waypoint in waypoints
            ]
        )
        return waypoints

    def process_next_waypoint(self, next_point, cur_point, yaw):
        if math.isnan(yaw):
            yaw = 0.0

        yaw = yaw + math.pi / 2.0
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        local_command_point = next_point - cur_point  # (H, 2)
        local_command_point = R.T.dot(local_command_point.T).T  # (H, 2)
        target_point = torch.FloatTensor(
            np.stack(
                [
                    local_command_point[:, 1] / self.model.magic_num,
                    -local_command_point[:, 0] / self.model.magic_num,
                ],
                axis=-1,
            )
        ).to(self.device)
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

    def post_process_control(self, throttle_res, steer_res, brake_res):
        if brake_res < 0.05:
            brake_res = 0.0
        if throttle_res > brake_res:
            brake_res = 0.0

        if brake_res > 0.5:
            throttle_res = float(0)

        return np.array([throttle_res, steer_res, brake_res])

    def process_control(self, traj, state, target_point):
        gt_velocity = torch.FloatTensor([state["state"][0][1]]).to(self.device, dtype=torch.float32)
        renew_traj = torch.stack((-traj[..., 0], traj[..., 1]), dim=-1)
        renew_target = torch.stack([-target_point[0], target_point[1]], dim=-1)
        throttle_res, steer_res, brake_res = self.controller.control_pid(
            renew_traj, gt_velocity, renew_target
        )

        return self.post_process_control(throttle_res, steer_res, brake_res)

    def plot_to_bev(self, bev_image, traj, filename="test.jpg"):
        for x, y in traj:
            x, y = x / self.model.magic_num, y / self.model.magic_num
            pixel_x = way_point_to_pixel(x.item())
            pixel_y = way_point_to_pixel(y.item())
            bev_image = cv2.circle(bev_image, (pixel_x, pixel_y), 3, (0, 0, 255), -1)
        cv2.imwrite(filename, cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR))

    def agent_to_world(self, agent_pos, yaw, cur_pos):
        """
        agent_pos: [H, 2]
        """
        if math.isnan(yaw):
            yaw = 0.0
        yaw = yaw + np.pi / 2.0
        agent_pos = agent_pos.cpu().numpy()
        agent_pos = np.stack([-agent_pos[:, 1], agent_pos[:, 0]], axis=-1)
        R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        world_pos = R.T.dot(agent_pos.T).T  # [H, 2]
        return world_pos + cur_pos[None]

    def plot_to_world(self, trajs):
        world = self.env_injector.world
        for x, y in trajs:
            world.debug.draw_string(
                carla.Location(float(x), float(y), 0.5),
                "x",
                draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255),
                life_time=-1,
                persistent_lines=True,
            )

    def run(self):
        state = self.env.reset()
        done = False
        count = 0

        while True:
            target_point = None
            image = self.process_image(state["camera"][0])
            if self.use_guidance_type != GuidanceType.NO_GUIDANCE:
                next_point = (
                    state["next_waypoint"]
                    if self.use_guidance_type == GuidanceType.FREE_GUIDANCE
                    else self.get_future_waypoints()
                )
                target_point = self.process_next_waypoint(
                    next_point=next_point,
                    cur_point=state["cur_waypoint"],
                    yaw=state["compass"][0][0],
                )
            bev_image = state["bev"][0]
            traj = self.generate_traj(
                image,
                target_point if self.use_guidance_type != GuidanceType.NO_GUIDANCE else None,
            )
            if self.bev_save_path:
                self.plot_to_bev(bev_image, traj[0, :, :2], f"{self.bev_save_path}/{count:06d}.jpg")
                count += 1
            if traj.size(-1) > 2:
                control = self.post_process_control(*traj[0, 0, -3:].cpu().numpy())
            else:
                control = self.process_control(
                    traj[0, :4, :2],
                    state,
                    (
                        target_point
                        if self.use_guidance_type != GuidanceType.NO_GUIDANCE
                        else traj[0, 4, :2]
                    ),
                )
            if self.plot_on_world:
                world_point = self.agent_to_world(
                    traj[0, :, :2], state["compass"][0][0], state["cur_waypoint"][0]
                )
                self.plot_to_world(world_point)
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

    agent = Agent(cfg, args.plot_on_world, args.save_bev_path)
    agent.run()
