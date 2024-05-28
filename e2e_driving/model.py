from collections import deque
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta

from e2e_driving.resnet import resnet34


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class TrajPredict(nn.Module):
    def __init__(
        self,
        pred_len: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.target_encoder = nn.Linear(2, hidden_dim)
        self.query_embed = nn.Embedding(pred_len, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.decoder_traj = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.output_traj = nn.Linear(hidden_dim, 2)

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, memory, target_points):
        target_points = self.target_encoder(target_points).unsqueeze(1)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(memory.size(0), 1, 1)
        query_embed = query_embed + target_points
        output = self.decoder_traj(query_embed, memory)
        return self.output_traj(output), output


class Planner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.turn_controller = PIDController(
            K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n
        )
        self.speed_controller = PIDController(
            K_P=config.speed_KP,
            K_I=config.speed_KI,
            K_D=config.speed_KD,
            n=config.speed_n,
        )

        self.perception = resnet34(pretrained=True)
        in_shape = self.perception.fc.in_features
        self.perception.fc = nn.Identity()
        self.proj = nn.Linear(in_shape, 256)

        self.measurements = nn.Sequential(
            nn.Linear(1 + 2 + 6, 256, bias=False),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
        )

        self.speed_branch = nn.Sequential(
            nn.Linear(in_shape, 256, bias=False),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.traj_to_crtl_proj = nn.Linear(256, 256)
        self.value_branch = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.traj_predict = TrajPredict(
            pred_len=config.pred_len,
            hidden_dim=256,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256, bias=False),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1, bias=False),
        )
        self.dist_mu = nn.Sequential(nn.Linear(256, 2), nn.Softplus())
        self.dist_sigma = nn.Sequential(nn.Linear(256, 2), nn.Softplus())

    def forward(
        self, img: torch.Tensor, state: torch.Tensor, target_point: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Model forward function to predict speed and waypoints.

        Parameters
        - img: torch.Tensor, shape=(B, 3, 256, 256)
        - state: torch.Tensor, shape=(B, 9)
        - target_point: torch.Tensor, shape=(B, 2)

        Returns
        - Dict[str, torch.Tensor]: Dictionary containing the following
            - pred_speed: torch.Tensor, shape=(B, 1)
            - pred_wp: torch.Tensor, shape=(B, 4, 2)
        """
        outputs = {}

        feature_emb, img_feature = self.perception(img)
        measurement_feature = self.measurements(state)
        outputs["pred_speed"] = self.speed_branch(feature_emb)
        traj_feature = self.proj(
            img_feature.flatten(2).permute(0, 2, 1)
        ) + measurement_feature.unsqueeze(1)

        outputs["pred_value_traj"] = self.value_branch(traj_feature.mean(dim=1))

        pred_wp, traj_hidden_state = self.traj_predict(traj_feature, target_point)
        traj_feature_from_control = (
            self.traj_to_crtl_proj(traj_hidden_state).mean(dim=1) + measurement_feature
        )
        outputs["pred_value_ctrl"] = self.value_branch(traj_feature_from_control)
        outputs["pred_wp"] = pred_wp
        policy = self.policy_head(traj_feature_from_control)
        value = self.value_head(traj_feature_from_control)
        outputs["mu_branches"] = self.dist_mu(policy)
        outputs["sigma_branches"] = self.dist_sigma(policy)
        outputs["value"] = value

        return outputs

    def action_to_control(self, action):
        acc, steer = action.cpu().numpy()[0].astype(np.float64)
        if acc >= 0.0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.abs(acc)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)

        return steer, throttle, brake

    def process_action(self, pred, command, speed, target_point):
        action = self._get_action_beta(
            pred["mu_branches"].view(1, 2), pred["sigma_branches"].view(1, 2)
        )
        steer, throttle, brake = self.action_to_control(action)

        metadata = {
            "speed": float(speed.cpu().numpy().astype(np.float64)),
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "command": command,
            "target_point": tuple(
                target_point[0].data.cpu().numpy().astype(np.float64)
            ),
        }
        return steer, throttle, brake, metadata

    def _get_action_beta(self, alpha, beta, unscale=True):
        x = torch.zeros_like(alpha)
        x[:, 1] += 0.5
        mask1 = (alpha > 1) & (beta > 1)
        x[mask1] = (alpha[mask1] - 1) / (alpha[mask1] + beta[mask1] - 2)

        mask2 = (alpha <= 1) & (beta > 1)
        x[mask2] = 0.0

        mask3 = (alpha > 1) & (beta <= 1)
        x[mask3] = 1.0

        # mean
        mask4 = (alpha <= 1) & (beta <= 1)
        x[mask4] = alpha[mask4] / torch.clamp((alpha[mask4] + beta[mask4]), min=1e-5)

        if unscale:
            x = x * 2 - 1  # unscale

        return x

    def act(self, image, state, target_point):
        output = self.forward(image, state, target_point)
        mu, sigma = output["mu_branches"], output["sigma_branches"]
        action = self._get_action_beta(mu.view(1, 2), sigma.view(1, 2), unscale=False)
        dist = Beta(mu, sigma)
        action_logprob = dist.log_prob(action)
        action = 2 * action - 1
        state_val = output["value"]
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate_actions(self, image, state, target_point, action):
        output = self.forward(image, state, target_point)
        mu, sigma = output["mu_branches"], output["sigma_branches"]
        dist = Beta(mu, sigma)
        dist_entropy = dist.entropy()
        action_logprob = dist.log_prob(action)
        state_val = output["value"]
        return action_logprob, state_val, dist_entropy

    def control_pid(self, waypoints, velocity, target):
        """Predicts vehicle control with a PID controller.
        Args:
                waypoints (tensor): output of self.plan()
                velocity (tensor): speedometer input
        """
        assert waypoints.size(0) == 1
        waypoints = waypoints[0].data.cpu().numpy()
        target = target.squeeze().data.cpu().numpy()

        # flip y (forward is negative in our waypoints)
        waypoints[:, 1] *= -1
        target[1] *= -1

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            # magnitude of vectors, used for speed
            desired_speed += (
                np.linalg.norm(waypoints[i + 1] - waypoints[i]) * 2.0 / num_pairs
            )

            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((waypoints[i + 1] + waypoints[i]) / 2.0)
            if abs(self.config.aim_dist - best_norm) > abs(self.config.aim_dist - norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # choice of point to aim for steering, removing outlier predictions
        # use target point if it has a smaller angle or if error is large
        # predicted point otherwise
        # (reduces noise in eg. straight roads, helps with sudden turn commands)
        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        use_target_to_aim = use_target_to_aim or (
            np.abs(angle_target - angle_last) > self.config.angle_thresh
            and target[1] < self.config.dist_thresh
        )
        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        speed = velocity[0].data.cpu().numpy()
        brake = (
            desired_speed < self.config.brake_speed
            or (speed / desired_speed) > self.config.brake_ratio
        )

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            "speed": float(speed.astype(np.float64)),
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "wp_4": tuple(waypoints[3].astype(np.float64)),
            "wp_3": tuple(waypoints[2].astype(np.float64)),
            "wp_2": tuple(waypoints[1].astype(np.float64)),
            "wp_1": tuple(waypoints[0].astype(np.float64)),
            "aim": tuple(aim.astype(np.float64)),
            "target": tuple(target.astype(np.float64)),
            "desired_speed": float(desired_speed.astype(np.float64)),
            "angle": float(angle.astype(np.float64)),
            "angle_last": float(angle_last.astype(np.float64)),
            "angle_target": float(angle_target.astype(np.float64)),
            "angle_final": float(angle_final.astype(np.float64)),
            "delta": float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata
