import os
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str, overrides: list[str] | None = None) -> DictConfig:
    """
    Load a v2 config YAML, merging it on top of default.yaml.

    Args:
        config_path: Path to an experiment YAML (absolute or relative to
                     ssmer_v2/config/).  Pass "default" to load only the
                     default config.
        overrides:   Optional list of "key=value" strings that override any
                     field after merging, e.g. ["training.lr=0.01"].

    Returns:
        Merged OmegaConf DictConfig (immutable struct after this call).

    Example:
        cfg = load_config("ablation_A4_lambda05.yaml", ["training.batch_size=64"])
    """
    config_dir = os.path.dirname(__file__)
    default_path = os.path.join(config_dir, "default.yaml")
    cfg = OmegaConf.load(default_path)

    if config_path and config_path != "default":
        if not os.path.isabs(config_path):
            config_path = os.path.join(config_dir, config_path)
        experiment_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, experiment_cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.set_struct(cfg, True)
    return cfg
