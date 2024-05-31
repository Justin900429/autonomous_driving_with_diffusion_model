import glob
import json
import os
from typing import Callable

import cv2
import torch

from .augment import img_augment_func


class TrajDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, img_transforms=None, use_img_augmentor=False):
        self.img_transforms = img_transforms
        self.use_img_augmentor = use_img_augmentor
        if use_img_augmentor:
            self.augmentor_func = img_augment_func
            self.count_access = 0
        self.root_path = root_path
        self.front_image = list(glob.glob(os.path.join(root_path, "front/*.png")))
        
        # Transform to torch to prevent memory issue
        waypoint_path = os.path.join(root_path, "waypoints.json")
        with open(waypoint_path, "r") as f:
            waypoints = json.load(f)
        waypoints.sort(key=lambda x: int(x["image"].split(".")[0]))
        self.waypoints = torch.Tensor([x["traj"] for x in waypoints])

    def __len__(self):
        return len(self.front_image)

    def __getitem__(self, idx):
        img_name = self.front_image[idx]
        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        if self.use_img_augmentor:
            self.count_access += 1
            img = self.augmentor_func(self.count_access).augment_image(img)
        img = self.img_transforms(img)
        traj = self.waypoints[idx].clip(-1, 1)
        return img, traj


def get_loader(cfg, train: bool, img_transforms: Callable) -> torch.utils.data.DataLoader:
    dataset = TrajDataset(
        cfg.TRAIN.ROOT,
        img_transforms=img_transforms,
        use_img_augmentor=cfg.TRAIN.USE_IMG_AUGMENTOR,
    )
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=train,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )