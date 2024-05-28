import os

import numpy as np
import torch
import torch.nn.functional as F
from e2e_driving.augment import hard as augmenter
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CARLA_Data(Dataset):
    def __init__(self, root: str, data_folders: str, img_aug: bool = False):
        self.root = root
        self.img_aug = img_aug
        self._batch_read_number = 0
        self.front_img = []

        # current x, y position
        self.x = []
        self.y = []
        self.command = []
        self.target_command = []
        self.target_gps = []
        self.theta = []
        self.speed = []

        self.value = []
        self.feature = []
        self.action = []
        self.action_mu = []
        self.action_sigma = []

        # Future positions (expert demonstrations)
        self.future_x = []
        self.future_y = []

        self.x_command = []
        self.y_command = []
        self.command = []
        self.only_ap_brake = []

        self.future_feature = []
        self.future_action = []
        self.future_action_mu = []
        self.future_action_sigma = []
        self.future_only_ap_brake = []

        self.x_command = []
        self.y_command = []
        self.command = []
        self.only_ap_brake = []

        for sub_root in data_folders:
            data = np.load(
                os.path.join(sub_root, "packed_data.npy"), allow_pickle=True
            ).item()

            self.x_command += data["x_target"]
            self.y_command += data["y_target"]
            self.command += data["target_command"]

            self.front_img += data["front_img"]
            self.x += data["input_x"]
            self.y += data["input_y"]
            self.theta += data["input_theta"]
            self.speed += data["speed"]

            self.future_x += data["future_x"]
            self.future_y += data["future_y"]

            self.future_feature += data["future_feature"]
            self.future_action += data["future_action"]
            self.future_action_mu += data["future_action_mu"]
            self.future_action_sigma += data["future_action_sigma"]
            self.future_only_ap_brake += data["future_only_ap_brake"]

            self.value += data["value"]
            self.feature += data["feature"]
            self.action += data["action"]
            self.action_mu += data["action_mu"]
            self.action_sigma += data["action_sigma"]
            self.only_ap_brake += data["only_ap_brake"]

        self.img_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.front_img)

    def __getitem__(self, index):
        data = dict()

        if self.img_aug:
            data["front_img"] = self.img_transform(
                augmenter(self._batch_read_number).augment_image(
                    np.array(
                        Image.open(self.root + self.front_img[index][0]).convert("RGB")
                    )
                )
            )
        else:
            data["front_img"] = self.img_transform(
                np.array(
                    Image.open(self.root + self.front_img[index][0]).convert("RGB")
                )
            )

        if np.isnan(self.theta[index][0]):
            self.theta[index][0] = 0.0

        ego_x = self.x[index][0]
        ego_y = self.y[index][0]
        ego_theta = self.theta[index][0]

        """
        Setup ground-truth future waypoints.
        :R: matrix that converts the future (x, y) into ego's coordinate
        :local_command_point: centering the waypoint around the ego
        """
        waypoints = []

        R = np.array(
            [
                [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
                [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
            ]
        )
        for i in range(4):
            local_command_point = np.array(
                [self.future_y[index][i] - ego_y, self.future_x[index][i] - ego_x]
            )
            local_command_point = R.T.dot(local_command_point)
            waypoints.append(local_command_point)
        data["waypoints"] = np.array(waypoints)

        data["action"] = self.action[index]
        data["action_mu"] = self.action_mu[index]
        data["action_sigma"] = self.action_sigma[index]

        future_only_ap_brake = self.future_only_ap_brake[index]
        future_action_mu = self.future_action_mu[index]
        future_action_sigma = self.future_action_sigma[index]

        for i in range(len(future_only_ap_brake)):
            if future_only_ap_brake[i]:
                future_action_mu[i][0] = 0.8
                future_action_sigma[i][0] = 5.5

        data["future_action_mu"] = future_action_mu
        data["future_action_sigma"] = future_action_sigma
        data["future_feature"] = self.future_feature[index]

        only_ap_brake = self.only_ap_brake[index]
        if only_ap_brake:
            data["action_mu"][0] = 0.8
            data["action_sigma"][0] = 5.5

        local_command_point_aim = np.array(
            [(self.y_command[index] - ego_y), self.x_command[index] - ego_x]
        )
        local_command_point_aim = R.T.dot(local_command_point_aim)
        data["target_point_aim"] = local_command_point_aim[:2]

        data["target_point"] = local_command_point_aim[:2]

        data["speed"] = self.speed[index]
        data["feature"] = self.feature[index]
        data["value"] = self.value[index]

        """
        Create an one hot vector of high-level command.
        Ex: command = 3 (straight) -> cmd_one_got = [0,0,1,0,0,0]
        VOID = -1
        LEFT = 1
        RIGHT = 2
        STRAIGHT = 3
        LANEFOLLOW = 4
        CHANGELANELEFT = 5
        CHANGELANERIGHT = 6
        """
        command = self.command[index]
        if command < 0:
            command = 4
        data["target_command"] = F.one_hot(
            torch.tensor(command) - 1, num_classes=6
        ).float()

        self._batch_read_number += 1
        return data
