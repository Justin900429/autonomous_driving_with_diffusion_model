import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import server_utils
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from carla_gym.utils import config_utils


def create_env(cfg: DictConfig, off_screen=False, seed=None):
    set_random_seed(cfg.seed if seed is None else seed, using_cuda=True)

    os.environ["CARLA_API_PATH"] = os.path.join(
        os.path.dirname(cfg.carla_sh_path), "PythonAPI/carla/"
    )
    # start carla servers
    server_manager = server_utils.CarlaServerManager(
        cfg.carla_sh_path, configs=cfg.train_envs
    )
    server_manager.start(off_screen=off_screen)

    # prepare agent
    agent_name = cfg.actors[cfg.ev_id].agent
    OmegaConf.save(config=cfg.agent[agent_name], f="config_agent.yaml")
    cfg_agent = OmegaConf.load("config_agent.yaml")

    obs_configs = {cfg.ev_id: OmegaConf.to_container(cfg_agent.obs_configs)}
    reward_configs = {cfg.ev_id: OmegaConf.to_container(cfg.actors[cfg.ev_id].reward)}
    terminal_configs = {
        cfg.ev_id: OmegaConf.to_container(cfg.actors[cfg.ev_id].terminal)
    }

    # env wrapper
    EnvWrapper = config_utils.load_entry_point(cfg_agent.env_wrapper.entry_point)
    wrapper_kargs = cfg_agent.env_wrapper.kwargs

    config_utils.check_h5_maps(cfg.train_envs, obs_configs, cfg.carla_sh_path)

    def env_maker(config, seed=None):
        print(f'making port {config["port"]}')
        env = gym.make(
            config["env_id"],
            obs_configs=obs_configs,
            reward_configs=reward_configs,
            terminal_configs=terminal_configs,
            host="localhost",
            port=config["port"],
            seed=cfg.seed if seed is None else seed,
            no_rendering=False,
            **config["env_configs"],
        )
        env = EnvWrapper(env, **wrapper_kargs)
        return env

    if cfg.dummy or len(server_manager.env_configs) == 1:
        env = DummyVecEnv(
            [
                lambda config=config: env_maker(config, seed)
                for config in server_manager.env_configs
            ]
        )
    else:
        env = SubprocVecEnv(
            [
                lambda config=config: env_maker(config, seed)
                for config in server_manager.env_configs
            ]
        )

    return env, server_manager
