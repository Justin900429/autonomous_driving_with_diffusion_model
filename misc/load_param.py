import torch


def copy_parameters(from_parameters: torch.nn.Parameter, to_parameters: torch.nn.Parameter):
    to_parameters = list(to_parameters)
    assert len(from_parameters) == len(to_parameters)
    for s_param, param in zip(from_parameters, to_parameters):
        param.data.copy_(s_param.to(param.device).data)
