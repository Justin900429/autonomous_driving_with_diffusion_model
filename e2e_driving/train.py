import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from e2e_driving.config import GlobalConfig
from e2e_driving.data import CARLA_Data
from e2e_driving.model import Planner


class VA_planner(pl.LightningModule):
    def __init__(self, config, lr):
        super().__init__()
        self.lr = lr
        self.config = config
        self.model = Planner(config)

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        front_img = batch["front_img"]
        speed = batch["speed"].to(dtype=torch.float32).view(-1, 1) / 12.0
        target_point = batch["target_point"].to(dtype=torch.float32)
        command = batch["target_command"]

        state = torch.cat([speed, target_point, command], 1)
        value = batch["value"].view(-1, 1)

        gt_waypoints = batch["waypoints"]

        pred = self.model(front_img, state, target_point)
        dist_sup = MultivariateNormal(batch["action_mu"], batch["action_sigma"])
        dist_pred = MultivariateNormal(pred["mu_branches"], pred["sigma_branches"])
        kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
        action_loss = torch.mean(kl_div[:, 0]) * 0.5 + torch.mean(kl_div[:, 1]) * 0.5
        speed_loss = F.l1_loss(pred["pred_speed"], speed) * self.config.speed_weight
        value_loss = (
            F.mse_loss(pred["pred_value_traj"], value)
            + F.mse_loss(pred["pred_value_ctrl"], value)
        ) * self.config.value_weight
        wp_loss = F.l1_loss(pred["pred_wp"], gt_waypoints, reduction="none").mean()

        loss = action_loss + speed_loss + value_loss + wp_loss

        self.log("train_action_loss", action_loss.item())
        self.log("train_wp_loss_loss", wp_loss.item())
        self.log("train_speed_loss", speed_loss.item())
        self.log("train_value_loss", value_loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-6)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        front_img = batch["front_img"]
        speed = batch["speed"].to(dtype=torch.float32).view(-1, 1) / 12.0
        target_point = batch["target_point"].to(dtype=torch.float32)
        command = batch["target_command"]
        state = torch.cat([speed, target_point, command], 1)

        gt_waypoints = batch["waypoints"]

        pred = self.model(front_img, state, target_point)

        speed_loss = F.l1_loss(pred["pred_speed"], speed) * self.config.speed_weight

        wp_loss = F.l1_loss(pred["pred_wp"], gt_waypoints, reduction="none").mean()

        val_loss = wp_loss + speed_loss

        self.log("val_speed_loss", speed_loss.item(), sync_dist=True)
        self.log("val_wp_loss_loss", wp_loss.item(), sync_dist=True)
        self.log("val_loss", val_loss.item(), sync_dist=True)

        return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id", type=str, default="VA", help="Unique experiment identifier."
    )
    parser.add_argument(
        "--epochs", type=int, default=60, help="Number of train epochs."
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument(
        "--val_every", type=int, default=3, help="Validation frequency (epochs)."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--logdir", type=str, default="log", help="Directory to log data to."
    )
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus")

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    # Config
    config = GlobalConfig()

    # Data
    train_set = CARLA_Data(
        root=config.root_dir_all,
        data_folders=config.train_data,
        img_aug=config.img_aug,
        num_future=config.pred_len,
    )
    print(len(train_set))
    val_set = CARLA_Data(
        root=config.root_dir_all,
        data_folders=config.val_data,
        num_future=config.pred_len,
    )
    print(len(val_set))

    dataloader_train = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    dataloader_val = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    VA_model = VA_planner(config, args.lr)

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False,
        mode="min",
        monitor="val_loss",
        save_top_k=2,
        save_last=True,
        dirpath=args.logdir,
        filename="best_{epoch:02d}-{val_loss:.3f}",
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.logdir,
        gpus=args.gpus,
        accelerator="ddp",
        sync_batchnorm=True,
        plugins=DDPPlugin(find_unused_parameters=False),
        profiler="simple",
        benchmark=True,
        log_every_n_steps=1,
        flush_logs_every_n_steps=5,
        callbacks=[
            checkpoint_callback,
        ],
        check_val_every_n_epoch=args.val_every,
        max_epochs=args.epochs,
    )

    trainer.fit(VA_model, dataloader_train, dataloader_val)
