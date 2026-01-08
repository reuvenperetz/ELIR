import torch


def get_optimizer(train_cfg, model):
    lr = train_cfg.get("lr", 0.0001)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    optimizer_params = train_cfg.get("optimizer_params", {})
    optimizer_params['lr'] = lr
    optimizer_params['weight_decay'] = weight_decay
    params = model.parameters()
    optimizer = train_cfg.get("optimizer", None)
    if optimizer:
        return optimizer(params, **optimizer_params)
    else:
        return torch.optim.Adam(params, **optimizer_params)

def get_scheduler(train_cfg, optimizer):
    scheduler_params = train_cfg.get("scheduler_params", {})
    scheduler = train_cfg.get("scheduler", None)
    if scheduler:
        scheduler = scheduler(optimizer, **scheduler_params)
    return scheduler

def get_opt_sched(train_cfg, model):
    # Optimizer
    optimizer = get_optimizer(train_cfg, model)
    # Scheduler
    scheduler = get_scheduler(train_cfg, optimizer)
    return optimizer, scheduler

