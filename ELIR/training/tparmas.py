import torch


def print_model_summary(model, optimizer=None):
    """Print a summary of model parameters and their training status."""

    total_params = 0
    trainable_params = 0
    frozen_params = 0

    # Get optimizer param ids if provided
    optimized_ids = set()
    if optimizer:
        for group in optimizer.param_groups:
            for p in group['params']:
                optimized_ids.add(id(p))

    print(f"{'Layer':<50} {'Shape':<20} {'Params':>12} {'Trainable':>10} {'Optimized':>10}")
    print("=" * 110)

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        trainable = param.requires_grad
        if trainable:
            trainable_params += num_params
        else:
            frozen_params += num_params

        optimized = "✓" if id(param) in optimized_ids else "-"
        trainable_str = "✓" if trainable else "✗"

        # Truncate long names
        display_name = name if len(name) <= 48 else "..." + name[-45:]

        print(f"{display_name:<50} {str(list(param.shape)):<20} {num_params:>12,} {trainable_str:>10} {optimized:>10}")

    print("=" * 110)
    print(f"Total parameters:     {total_params:>15,}")
    print(f"Trainable parameters: {trainable_params:>15,} ({100 * trainable_params / total_params:.1f}%)")
    print(f"Frozen parameters:    {frozen_params:>15,} ({100 * frozen_params / total_params:.1f}%)")

    if optimizer:
        optimized_count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        print(f"Optimized parameters: {optimized_count:>15,}")


def get_optimizer(train_cfg, model):
    lr = train_cfg.get("lr", 0.0001)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    optimizer_params = train_cfg.get("optimizer_params", {})
    optimizer_params['lr'] = lr
    optimizer_params['weight_decay'] = weight_decay
    params = model.parameters()
    optimizer = train_cfg.get("optimizer", None)
    if optimizer:
        opt = optimizer(params, **optimizer_params)
        print_model_summary(model, opt)
        return opt
    else:
        opt = torch.optim.Adam(params, **optimizer_params)
        print_model_summary(model, opt)
        return opt

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

