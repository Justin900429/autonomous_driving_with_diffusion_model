import os

import carla
from loguru import logger


class CarlaAgent:
    def __init__(self, host, port, carla_map):
        self.client = carla.Client(host, port)
        self.client.set_timeout(60.0)
        self.world = self.client.load_world(carla_map)
        self.tm = self.client.get_trafficmanager(port + 6000)
        self.set_sync_mode(True)
        self.world.tick()
        
    def set_sync_mode(self, sync):
        settings = self.world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 0.1
        settings.deterministic_ragdolls = True
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(sync)
        
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        logger.debug("env __exit__!")

    def close(self):
        self.clean()
        self.set_sync_mode(False)
        self.client = None
        self.world = None
        self.tm = None

    def clean(self):
        self.world.tick()