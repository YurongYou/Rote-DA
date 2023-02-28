from functools import partial
from colorlog import warnings

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle, CosineAnneal

# no update keys consists of names of the model components that are not updated with pretrained weights
def build_optimizer(model, optim_cfg, no_update_keys=[]):
    def build_optimizer_helper(model, lr, optim_cfg):
        if optim_cfg.OPTIMIZER == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=optim_cfg.WEIGHT_DECAY)
        elif optim_cfg.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), lr=lr, weight_decay=optim_cfg.WEIGHT_DECAY,
                momentum=optim_cfg.MOMENTUM
            )
        elif optim_cfg.OPTIMIZER in ['adam_onecycle', 'adam_cosine_anneal']:
            def children(m: nn.Module):
                return list(m.children())

            def num_children(m: nn.Module) -> int:
                return len(children(m))

            flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
            get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

            optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
            optimizer = OptimWrapper.create(
                optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
            )
        else:
            raise NotImplementedError
        return optimizer 


    if optim_cfg.get('HISTORY_QUERY', None) is not None and hasattr(model, "history_query"):
        o1 = build_optimizer_helper(model.history_query, optim_cfg.HISTORY_QUERY.LR, optim_cfg)
        # for n, m in model.named_children():
        #     if n != 'history_query':
        #         print(n)
        det = nn.Sequential(*[m for n, m in model.named_children() if n != 'history_query'])
        o2 = build_optimizer_helper(det, optim_cfg.LR, optim_cfg)
        optimizer = double_optimizer(o1, o2)
    else:
        optimizer = build_optimizer_helper(model, optim_cfg.LR, optim_cfg)
    

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):

    def build_scheduler_helper(optimizer, lr, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
        decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in decay_steps:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * optim_cfg.LR_DECAY
            return max(cur_decay, optim_cfg.LR_CLIP / lr)

        lr_warmup_scheduler = None
        total_steps = total_iters_each_epoch * total_epochs
        if optim_cfg.OPTIMIZER == 'adam_onecycle':
            lr_scheduler = OneCycle(
                optimizer, total_steps, lr, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
            )
        elif optim_cfg.OPTIMIZER == 'adam_cosine_anneal':
            lr_scheduler = CosineAnneal(
                optimizer, total_steps, lr, list(
                    optim_cfg.MOMS), optim_cfg.LR_MIN
            )
        else:
            lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

            if optim_cfg.LR_WARMUP:
                lr_warmup_scheduler = CosineWarmupLR(
                    optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                    eta_min=lr / optim_cfg.DIV_FACTOR
                )

        return lr_scheduler, lr_warmup_scheduler
    
    if optim_cfg.get('HISTORY_QUERY', None) is not None and isinstance(optimizer, double_optimizer):
        lr_scheduler1, lr_warmup_scheduler1 = build_scheduler_helper(optimizer.optimizer1, optim_cfg.HISTORY_QUERY.LR, \
            total_iters_each_epoch, total_epochs, last_epoch, optim_cfg)
        lr_scheduler2, lr_warmup_scheduler2 = build_scheduler_helper(optimizer.optimizer2, optim_cfg.LR, \
            total_iters_each_epoch, total_epochs, last_epoch, optim_cfg)

        lr_scheduler = double_lr_scheduler(lr_scheduler1, lr_scheduler2)
        if lr_warmup_scheduler1 is None:
            lr_warmup_scheduler = None
        else:
            lr_warmup_scheduler = double_lr_scheduler(lr_warmup_scheduler1, lr_warmup_scheduler2)
    else:
        lr_scheduler, lr_warmup_scheduler = build_scheduler_helper(optimizer, optim_cfg.LR, \
            total_iters_each_epoch, total_epochs, last_epoch, optim_cfg)
    return lr_scheduler, lr_warmup_scheduler

class double_optimizer:
    def __init__(self, optimizer1, optimizer2):
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        
        
    def step(self):
        self.optimizer1.step()
        self.optimizer2.step()
        
    def state_dict(self):
        return {
            'optimizer1': self.optimizer1.state_dict(),
            'optimizer2': self.optimizer2.state_dict()
        }
    
    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        
    def load_state_dict(self, state_dict):
        self.optimizer1.load_state_dict(state_dict['optimizer1'])
        self.optimizer2.load_state_dict(state_dict['optimizer2'])


class double_lr_scheduler:
    def __init__(self, lr_scheduler1, lr_scheduler2):
        self.lr_scheduler1 = lr_scheduler1
        self.lr_scheduler2 = lr_scheduler2

    def step(self, step):
        self.lr_scheduler1.step(step)
        self.lr_scheduler2.step(step)