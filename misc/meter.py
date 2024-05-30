"""Taken from
https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/avgmeter.py
"""

from collections import defaultdict

import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError("Input to MetricMeter.update() must be a dictionary")

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(f"{name} {meter.val:.4f} ({meter.avg:.4f})")
        return self.delimiter.join(output_str)

    def get_log_dict(self):
        log_dict = {}
        for name, meter in self.meters.items():
            log_dict[name] = meter.val
            log_dict[f"avg_{name}"] = meter.avg
        return log_dict
