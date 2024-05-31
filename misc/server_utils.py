import logging
import os
import subprocess
import time

from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def kill_carla():
    kill_process = subprocess.Popen("killall -9 -r CarlaUE4-Linux", shell=True)
    kill_process.wait()
    time.sleep(1)
    log.info("Kill Carla Servers!")


class CarlaServerManager:
    def __init__(self, carla_sh_str, port=2000, configs=None, t_sleep=5):
        self._carla_sh_str = carla_sh_str
        # self._root_save_dir = root_save_dir
        self._t_sleep = t_sleep
        self.env_configs = []

        with open(os.path.join(os.path.dirname(carla_sh_str), "VERSION"), "r") as f:
            carla_version = f.read().strip()
        self.larger_than_0_9_12 = carla_version >= "0.9.12"

        if configs is None:
            cfg = {
                "gpu": 0,
                "port": port,
            }
            self.env_configs.append(cfg)
        else:
            for cfg in configs:
                for gpu in cfg["gpu"]:
                    single_env_cfg = OmegaConf.to_container(cfg)
                    single_env_cfg["gpu"] = gpu
                    single_env_cfg["port"] = port
                    self.env_configs.append(single_env_cfg)
                    port += 5

    def start(self, off_screen=False):
        kill_carla()
        for cfg in self.env_configs:
            cmd = (
                f"bash {self._carla_sh_str} " f'-fps=10 -carla-server -carla-rpc-port={cfg["port"]}'
            )
            if off_screen:
                if self.larger_than_0_9_12:
                    cmd = f"{cmd} -RenderOffScreen"
                else:
                    cmd = f"DISPLAY= {cmd} -opengl"
            log.info(cmd)
            subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        time.sleep(self._t_sleep)

    def stop(self):
        kill_carla()
        time.sleep(self._t_sleep)
        log.info("Kill Carla Servers!")
