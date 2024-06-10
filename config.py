import os.path as osp
import pprint

from colorama import Fore, Style
from tabulate import tabulate
from yacs.config import CfgNode as CN


def create_cfg():
    cfg = CN()
    cfg._BASE_ = None
    cfg.PROJECT_NAME = "carla_diffusion"
    cfg.PROJECT_DIR = None

    cfg.ENV = CN()
    cfg.ENV.CONFIG_PATH = "data_collect"
    cfg.ENV.AGENT_WARMUP = 1

    # ======= Model setup =======
    cfg.MODEL = CN()
    cfg.MODEL.HORIZON = 16
    cfg.MODEL.TRANSITION_DIM = 7
    cfg.MODEL.USE_ATTN = False
    cfg.MODEL.DIM = 64
    cfg.MODEL.DIM_MULTS = (1, 2, 4, 8)
    cfg.MODEL.DIFFUSER_BUILDING_BLOCK = "concat"

    # ======== Training set =======
    cfg.TRAIN = CN()
    cfg.TRAIN.RESUME = None
    cfg.TRAIN.USE_COND = "NO_GUIDANCE"
    cfg.TRAIN.USE_FREE_COND_PROB = 0.7
    cfg.TRAIN.LOG_INTERVAL = 20
    cfg.TRAIN.SAVE_INTERVAL = 3000
    cfg.TRAIN.SAMPLE_INTERVAL = 3000
    cfg.TRAIN.USE_IMG_AUGMENTOR = True
    cfg.TRAIN.ROOT = None
    cfg.TRAIN.IMAGE_HEIGHT = 256
    cfg.TRAIN.IMAGE_WIDTH = 900

    # Training iteration
    cfg.TRAIN.BATCH_SIZE = 32
    cfg.TRAIN.NUM_WORKERS = 4
    cfg.TRAIN.MAX_ITER = 100000
    cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS = 1
    cfg.TRAIN.GRAD_NORM = 1.0

    # EMA setup
    cfg.TRAIN.EMA_MAX_DECAY = 0.9999
    cfg.TRAIN.EMA_INV_GAMMA = 1.0
    cfg.TRAIN.EMA_POWER = 0.75

    # Optimizer
    cfg.TRAIN.LR = 0.0001
    cfg.TRAIN.LR_WARMUP = 1000

    # Diffusion setup
    cfg.TRAIN.TIME_STEPS = 100
    cfg.TRAIN.SAMPLE_STEPS = cfg.TRAIN.TIME_STEPS
    cfg.TRAIN.NOISE_SCHEDULER = CN()
    # Below two lines are only useful when setting the scheduler type to `linear`
    cfg.TRAIN.NOISE_SCHEDULER.BETA_START = 1e-4
    cfg.TRAIN.NOISE_SCHEDULER.BETA_END = 0.02
    cfg.TRAIN.NOISE_SCHEDULER.TYPE = "squaredcos_cap_v2"
    cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE = "sample"

    # ======= PID setup =======
    cfg.PID = CN()
    cfg.PID.TURN_KP = 1
    cfg.PID.TURN_KI = 0.5
    cfg.PID.TURN_KD = 1.0
    cfg.PID.TURN_N = 40
    cfg.PID.SPEED_KP = 5
    cfg.PID.SPEED_KI = 0.5
    cfg.PID.SPEED_KD = 1.0
    cfg.PID.SPEED_N = 40

    # ====== Control setup ======
    cfg.CONTROL = CN()
    cfg.CONTROL.AIM_DIST = 4.0
    cfg.CONTROL.ANGLE_THRESH = 0.3
    cfg.CONTROL.DIST_THRESH = 10
    cfg.CONTROL.BRAKE_SPEED = 0.4
    cfg.CONTROL.BRAKE_RATIO = 1.1
    cfg.CONTROL.CLIP_DELTA = 0.25
    cfg.CONTROL.MAX_THROTTLE = 9

    # ====== Guidance setup ======
    cfg.GUIDANCE = CN()
    cfg.GUIDANCE.USE_COND = "NO_GUIDANCE"
    cfg.GUIDANCE.LOSS_LIST = None
    cfg.GUIDANCE.STEP = 1
    cfg.GUIDANCE.CLASSIFIER_SCALE = 0.1
    cfg.GUIDANCE.FREE_SCALE = 5.5

    # ======= Evaluation set =======
    cfg.EVAL = CN()
    cfg.EVAL.BATCH_SIZE = 4
    cfg.EVAL.ETA = 0
    cfg.EVAL.CHECKPOINT = None
    cfg.EVAL.SCHEDULER = "ddim"
    cfg.EVAL.SAMPLE_STEPS = 10
    return cfg


def merge_possible_with_base(cfg: CN, config_path):
    with open(config_path, "r") as f:
        new_cfg = cfg.load_cfg(f)
    if "_BASE_" in new_cfg:
        cfg.merge_from_file(osp.join(osp.dirname(config_path), new_cfg._BASE_))
    cfg.merge_from_other_cfg(new_cfg)


def split_into(v):
    res = "(\n"
    for item in v:
        res += f"    {item},\n"
    res += ")"
    return res


def pretty_print_cfg(cfg):
    def _indent(s_, num_spaces):
        s = s_.split("\n")
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    r = ""
    s = []
    for k, v in sorted(cfg.items()):
        seperator = "\n" if isinstance(v, CN) else " "
        attr_str = "{}:{}{}".format(
            str(k),
            seperator,
            pretty_print_cfg(v) if isinstance(v, CN) else pprint.pformat(v),
        )
        attr_str = _indent(attr_str, 2)
        s.append(attr_str)
    r += "\n".join(s)
    return r


def show_config(cfg):
    table = tabulate(
        {"Configuration": [pretty_print_cfg(cfg)]},
        headers="keys",
        tablefmt="fancy_grid",
    )
    print(f"{Fore.BLUE}", end="")
    print(table)
    print(f"{Style.RESET_ALL}", end="")
