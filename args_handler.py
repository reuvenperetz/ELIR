import argparse


def argument_handler():
    parser = argparse.ArgumentParser()

    parser.add_argument("--yaml_path", '-y', type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--seed", '-s', type=int)
    parser.add_argument("--epochs", '-e', type=int)
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--weight_decay", type=float, help="weight decay")
    parser.add_argument("--ema_decay", type=float, help="EMA decay")
    parser.add_argument("--mlflow_tracking_uri", type=str, help="MLflow tracking URI")
    # Generic override for any nested config key using dot notation
    # Example: --override dataset_cfg.train_dataset.random_crop=true
    parser.add_argument("--override", "-O", action="append", default=[],
                        help="Override config value using dot notation: key.subkey=value")

    args_dict = vars(parser.parse_args())
    yaml_path = args_dict.pop("yaml_path")
    dot_overrides = args_dict.pop("override")
    overides = {k: v for k, v in args_dict.items() if v is not None}
    return yaml_path, overides, dot_overrides


def replace_key(conf, k, v):
    if k in conf:
        conf[k] = v
        return
    else:
        for sub_conf in conf:
            if isinstance(conf[sub_conf], dict):
                replace_key(conf[sub_conf], k, v)


def set_nested_value(conf, key_path, value):
    """
    Set a value in a nested dict using dot notation.
    Example: set_nested_value(conf, "dataset_cfg.train_dataset.random_crop", True)
    """
    keys = key_path.split(".")
    d = conf
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]

    # Convert string values to appropriate types
    if isinstance(value, str):
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        # Check for list syntax: [item1,item2,...]
        elif value.startswith("[") and value.endswith("]"):
            import ast
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # Keep as string if parsing fails
        elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string

    d[keys[-1]] = value


def set_overides(conf, overides, dot_overrides=None):
    # Apply simple key overrides (searches recursively)
    for k, v in overides.items():
        replace_key(conf, k, v)

    # Apply dot-notation overrides (precise path)
    if dot_overrides:
        for override in dot_overrides:
            if "=" in override:
                key_path, value = override.split("=", 1)
                set_nested_value(conf, key_path, value)
            else:
                print(f"Warning: Invalid override format '{override}'. Use key.path=value")

