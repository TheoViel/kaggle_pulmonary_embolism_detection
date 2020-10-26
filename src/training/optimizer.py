import torch

def define_optimizer(name, params, lr=1e-3):
    try:
        optimizer = getattr(torch.optim, name)(params, lr=lr)
    except:
        raise NotImplementedError

    return optimizer
