import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import OneCycle, CosineWarmupLR


def build_optimizer(model, optim_cfg):
    if optim_cfg.type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    elif optim_cfg.type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay,
            momentum=optim_cfg.momentum
        )
    elif optim_cfg.type == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, optim_cfg.lr, get_layer_groups(model), wd=optim_cfg.weight_decay, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg, lr_cfg):

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs

    if lr_cfg.policy == 'onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.lr, list(lr_cfg.moms), lr_cfg.div_factor, lr_cfg.pct_start
        )

    elif lr_cfg.policy == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, last_epoch=last_epoch)

    elif lr_cfg.policy == 'step':

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_cfg.step, last_epoch=last_epoch)

    else:
        raise NotImplementedError

    if 'warmup' in lr_cfg:
          lr_warmup_scheduler = CosineWarmupLR(
              optimizer, T_max=lr_cfg.warmup_iters,
              eta_min=optim_cfg.lr * lr_cfg.warmup_ratio
          )

    return lr_scheduler, lr_warmup_scheduler
