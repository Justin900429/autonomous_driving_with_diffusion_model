import argparse
import datetime
import gc
import glob
import math
import os
import os.path as osp
import random
import time

import accelerate
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from diffusers.optimization import get_constant_schedule_with_warmup
from diffusers.schedulers import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import make_image_grid, numpy_to_pil
from loguru import logger
from PIL import Image
from tqdm import tqdm

from config import create_cfg, merge_possible_with_base, show_config
from dataset import get_loader
from misc import AverageMeter, MetricMeter
from modeling import build_model


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None, type=str)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(
    cfg,
    unet,
    noise_scheduler,
    device,
    labels=None,
    conditional_imgs=None,
    conditional_img_transforms=None,
    filename=None,
):
    unet.eval()
    num_images = cfg.EVAL.BATCH_SIZE
    image_shape = (
        num_images,
        cfg.MODEL.HORIZEON,
        cfg.MODEL.TRANSITION_DIM,
    )
    images = torch.randn(image_shape, device=device)
    cond_river_images = glob.glob(os.path.join(cfg.TEST.ROOT, "*RI*.png"))
    cond_road_images = glob.glob(os.path.join(cfg.TEST.ROOT, "*RO*.png"))

    if labels is None:
        labels = [0] * (num_images // 2) + [1] * (num_images - num_images // 2)
        labels = torch.nn.functional.one_hot(
            torch.LongTensor(labels), num_classes=2
        ).to(device)
        labels = torch.cat(
            [torch.zeros_like(labels), torch.zeros_like(labels), labels], dim=0
        )

    road_images = random.choices(cond_road_images, k=num_images - num_images // 2)
    river_images = random.choices(cond_river_images, k=num_images // 2)
    cond_transform_images = torch.stack(
        [
            conditional_img_transforms(Image.open(img).convert("L"))
            for img in (road_images + river_images)
        ]
    ).to(device)
    cond_transform_images = torch.cat(
        [
            torch.zeros_like(cond_transform_images),
            cond_transform_images,
            torch.zeros_like(cond_transform_images),
        ],
        dim=0,
    )

    noise_scheduler.set_timesteps(cfg.EVAL.SAMPLE_STEPS, device=device)
    for t in tqdm(noise_scheduler.timesteps):
        stack_images = torch.cat([images, images, images], dim=0)  # Add sketch guidance
        stack_images = torch.cat((stack_images, cond_transform_images), dim=1)
        model_output_uncond, model_output_cond_img, model_output_cond_label = (
            unet(
                stack_images,
                t.reshape(-1),
                class_labels=labels,
                conditional_imgs=conditional_imgs,
            )[:, :-1]
            .float()
            .chunk(3, dim=0)
        )
        model_output = (
            model_output_uncond
            + cfg.EVAL.GUIDANCE * (model_output_cond_img - model_output_uncond) / 2
            + cfg.EVAL.GUIDANCE * (model_output_cond_label - model_output_uncond) / 2
        )
        if cfg.EVAL.SCHEDULER == "ddim":
            images = noise_scheduler.step(
                model_output, t, images, use_clipped_model_output=True, eta=cfg.EVAL.ETA
            ).prev_sample
        else:
            images = noise_scheduler.step(model_output, t, images).prev_sample

    # Save only the first three channels
    images = (images.to(torch.float32).clamp(-1, 1) + 1) / 2
    images = numpy_to_pil(images.cpu().permute(0, 2, 3, 1).numpy())

    if filename is not None:
        square_size = int(math.sqrt(cfg.EVAL.BATCH_SIZE))
        make_image_grid(images, rows=square_size, cols=square_size).save(filename)
        logger.info(f"Save generated samples to {filename}...")
    else:
        return images


def main(args):
    cfg = create_cfg()

    if args.config is not None:
        merge_possible_with_base(cfg, args.config)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    configuration = accelerate.utils.ProjectConfiguration(
        project_dir=cfg.PROJECT_DIR,
    )
    kwargs = accelerate.InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=3600))
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[kwargs],
        gradient_accumulation_steps=cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS,
        log_with=["aim"],
        project_config=configuration,
        mixed_precision=cfg.TRAIN.MIXED_PRECISION,
    )

    accelerator.init_trackers(project_name=cfg.PROJECT_NAME)
    if accelerator.is_main_process:
        show_config(cfg)

    device = accelerator.device
    if accelerator.is_main_process:
        os.makedirs(osp.join(cfg.PROJECT_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(osp.join(cfg.PROJECT_DIR, "generate"), exist_ok=True)

    # Build model
    model = build_model(cfg)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.TRAIN.SAMPLE_STEPS,
        prediction_type=cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE,
        beta_schedule=cfg.TRAIN.NOISE_SCHEDULER.TYPE,
        # For linear only
        beta_start=cfg.TRAIN.NOISE_SCHEDULER.BETA_START,
        beta_end=cfg.TRAIN.NOISE_SCHEDULER.BETA_END,
    )

    ema_model = EMAModel(
        model.parameters(),
        update_after_step=5000,
        decay=cfg.TRAIN.EMA_MAX_DECAY,
        use_ema_warmup=True,
        inv_gamma=cfg.TRAIN.EMA_INV_GAMMA,
        power=cfg.TRAIN.EMA_POWER,
    )

    # Build data loader
    img_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataloader = get_loader(
        cfg,
        train=True,
        img_transforms=img_transforms,
    )

    # Build optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.TRAIN.LR, betas=(0.95, 0.999), eps=1e-7
    )
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.LR_WARMUP,
    )

    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, dataloader
    )
    ema_model.to(accelerator.device)

    start_iter = 0
    if cfg.TRAIN.RESUME is not None:
        assert osp.exists(cfg.TRAIN.RESUME), "Resume file not found"
        if accelerator.is_main_process:
            logger.info(f"Resume checkpoint from {cfg.TRAIN.RESUME}...")
        with accelerator.main_process_first():
            state_dict = torch.load(cfg.TRAIN.RESUME, map_location=device)
        ema_model.load_state_dict(state_dict["ema_state_dict"])
        accelerator.unwrap_model(model).load_state_dict(state_dict["state_dict"])
        optimizer.optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.scheduler.load_state_dict(state_dict["lr_scheduler"])
        start_iter = state_dict["iter"] + 1
        del state_dict
        torch.cuda.empty_cache()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    loss_meter = MetricMeter()
    iter_time = AverageMeter()

    max_iter = cfg.TRAIN.MAX_ITER
    loader = iter(dataloader)

    cur_iter = start_iter
    start = time.time()
    while True:
        model.train()
        try:
            imgs, trajs = next(loader)
        except StopIteration:
            loader = iter(dataloader)
            imgs, trajs = next(loader)
        imgs = imgs.to(weight_dtype)
        trajs = trajs.to(weight_dtype)

        t = torch.randint(
            0, cfg.TRAIN.TIME_STEPS, (trajs.shape[0],), device=device
        ).long()
        noise = torch.randn_like(trajs, dtype=weight_dtype)
        noise_data = noise_scheduler.add_noise(trajs, noise, t)

        with accelerator.accumulate(model):
            pred = model(noise_data, imgs, t)

            if cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE == "epsilon":
                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            else:
                raise ValueError("Not supported prediction type.")

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                for param in model.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(
                            param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                        )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            ema_model.step(model.parameters())

        if (
            (cur_iter + 1) % cfg.TRAIN.LOG_INTERVAL == 0
            and accelerator.is_main_process
            and accelerator.sync_gradients
        ):
            iter_time.update(time.time() - start)
            loss_meter.update({"loss": loss.item()})
            nb_future_iters = max_iter - (cur_iter + 1)
            eta_seconds = iter_time.avg * nb_future_iters
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                f"iter: [{cur_iter + 1}/{max_iter}]\t"
                f"time: {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                f"eta: {eta_str}\t"
                f"lr: {optimizer.param_groups[-1]['lr']:.2e}\t"
                f"{loss_meter}"
            )
            accelerator.log(loss_meter.get_log_dict(), step=cur_iter + 1)
            start = time.time()

        if (
            (
                ((cur_iter + 1) % cfg.TRAIN.SAVE_INTERVAL == 0)
                or (cur_iter + 1 == max_iter)
            )
            and accelerator.is_main_process
            and accelerator.sync_gradients
        ):
            state_dict = {
                "state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.scheduler.state_dict(),
                "iter": cur_iter + 1,
                "ema_state_dict": ema_model.state_dict(),
            }
            save_name = (
                f"checkpoint_{cur_iter + 1}.pth"
                if cur_iter + 1 != max_iter
                else "final.pth"
            )
            torch.save(state_dict, osp.join(cfg.PROJECT_DIR, "checkpoints", save_name))
            logger.info(f"Save checkpoint to {save_name}...")

        if (
            accelerator.is_main_process
            and (
                ((cur_iter + 1) % cfg.TRAIN.SAMPLE_INTERVAL == 0)
                or (cur_iter + 1 == max_iter)
            )
            and accelerator.sync_gradients
        ):
            filename = osp.join(
                cfg.PROJECT_DIR, "generate", f"iter_{cur_iter + 1:03d}.png"
            )
            unet = accelerator.unwrap_model(model)
            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())
            evaluate(
                cfg,
                unet,
                noise_scheduler,
                device,
                filename=filename,
            )
            ema_model.restore(unet.parameters())

        if accelerator.sync_gradients:
            cur_iter += 1

        if cur_iter == max_iter:
            break
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    # same_seeds(29383)
    args = parse_args()
    main(args)
