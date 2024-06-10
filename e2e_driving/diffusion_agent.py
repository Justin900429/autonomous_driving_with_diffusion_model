import datetime
import json
import math
import os
import pathlib
import time
from collections import OrderedDict, deque

import carla
import cv2
import numpy as np
import torch
from leaderboard.autoagents import autonomous_agent
from PIL import Image
from torchvision import transforms as T

from config import create_cfg
from control import GuidanceLoss
from e2e_driving.planner import RoutePlanner
from misc.constant import GuidanceType
from misc.load_param import copy_parameters
from modeling import build_model
from scheduler import GuidanceDDIMScheduler, GuidanceDDPMScheduler

SAVE_PATH = os.environ.get("SAVE_PATH", None)
COLOR_LIST = [
    (255, 0, 0),
    (253, 60, 60),
    (256, 126, 126),
    (255, 168, 168),
]
SCHEDULER_FUNC = {
    "ddpm": GuidanceDDPMScheduler,
    "ddim": GuidanceDDIMScheduler,
}


def way_point_to_pixel(waypoint):
    pixel_val = waypoint / 23.315 * 256
    return int(256 - pixel_val)


def get_entry_point():
    return "DiffusionAgent"


class DiffusionAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.alpha = 0.3
        self.status = 0
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = deque()

        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        cfg = create_cfg()
        if self.config_path is not None:
            cfg.merge_from_file(self.config_path)
        self.use_guidance_type = GuidanceType[cfg.GUIDANCE.USE_COND]
        self.cfg = cfg
        self.model = build_model(self.cfg).to(self.device)
        if cfg.EVAL.CHECKPOINT:
            weight = torch.load(cfg.EVAL.CHECKPOINT, map_location=self.device)
            self.model.load_state_dict(weight["state_dict"])
            copy_parameters(weight["ema_state_dict"]["shadow_params"], self.model.parameters())
            del weight
            torch.cuda.empty_cache()
            print("weights are loaded")
        self.model.eval()

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
        self.init_trajs = torch.randn(traj_shape, device="cuda")
        self.save_path = None
        self.image_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.last_steers = deque()
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: f"{x:02d}",
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )

            print(string)

            self.save_path = pathlib.Path(os.environ["SAVE_PATH"]) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / "rgb").mkdir()
            (self.save_path / "meta").mkdir()
            (self.save_path / "bev").mkdir()

    def _init(self):
        self._route_planner = RoutePlanner(7.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": -1.5,
                "y": 0.0,
                "z": 2.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 900,
                "height": 256,
                "fov": 100,
                "id": "rgb",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 0.0,
                "y": 0.0,
                "z": 50.0,
                "roll": 0.0,
                "pitch": -90.0,
                "yaw": 0.0,
                "width": 512,
                "height": 512,
                "fov": 5 * 10.0,
                "id": "bev",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

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

    def tick(self, input_data):
        self.step += 1
        rgb = cv2.cvtColor(input_data["rgb"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        bev = cv2.cvtColor(input_data["bev"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]

        if math.isnan(compass):
            compass = 0.0

        result = {
            "rgb": rgb,
            "gps": gps,
            "speed": speed,
            "compass": compass,
            "bev": bev,
        }

        pos = self._get_position(result)
        result["gps"] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_command"] = next_cmd.value

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point).reshape(-1)

        result["target_point"] = (
            np.array([local_command_point[1], -local_command_point[0]]) / self.model.magic_num
        )

        return result

    def post_process_control(self, throttle_res, steer_res, brake_res):
        if brake_res < 0.05:
            brake_res = 0.0
        if throttle_res > brake_res:
            brake_res = 0.0

        if brake_res > 0.5:
            throttle_res = float(0)

        return throttle_res, steer_res, brake_res

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        if self.step < self.cfg.ENV.AGENT_WARMUP:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0

            return control

        if self.use_guidance_type != GuidanceType.NO_GUIDANCE:
            target_point = torch.FloatTensor(tick_data["target_point"]).to(self.device)
        image = self.image_transform(tick_data["rgb"]).unsqueeze(0).to(self.device)
        traj = self.generate_traj(
            image, target_point if self.use_guidance_type != GuidanceType.NO_GUIDANCE else None
        )

        throttle, steer, brake = self.post_process_control(*traj[0, 0, -3:].cpu().numpy())
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data, traj[0, :, :2].cpu().numpy())
        return control

    def save(self, tick_data, traj_data):
        frame = self.step // 10
        for x, y in traj_data:
            x, y = x / self.model.magic_num, y / self.model.magic_num
            pixel_x = way_point_to_pixel(x)
            pixel_y = way_point_to_pixel(y)
            tick_data["bev"] = cv2.circle(tick_data["bev"], (pixel_x, pixel_y), 3, (0, 0, 255), -1)
        Image.fromarray(tick_data["rgb"]).save(self.save_path / "rgb" / ("%04d.png" % frame))

        Image.fromarray(tick_data["bev"]).save(self.save_path / "bev" / ("%04d.png" % frame))

    def destroy(self):
        del self.net
        torch.cuda.empty_cache()
