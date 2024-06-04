import glob
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
        self.front_image = sorted(list(glob.glob(os.path.join(root_path, "front/*.png"))))

    def __len__(self):
        return len(self.front_image)

    def __getitem__(self, idx):
        img_name = self.front_image[idx]
        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        if self.use_img_augmentor:
            self.count_access += 1
            img = self.augmentor_func(self.count_access).augment_image(img)
        img = self.img_transforms(img)

        waypoint_name = os.path.join(self.root_path, "waypoints", f"{idx:06d}.txt")
        with open(waypoint_name, "r") as f:
            waypoints = f.readlines()

        waypoints = torch.tensor(
            [list(map(float, line.strip().split())) for line in waypoints if len(line.strip()) != 0]
        )
        waypoints = waypoints.clip(-1, 1)
        assert len(waypoints) == 16
        return img, waypoints


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
