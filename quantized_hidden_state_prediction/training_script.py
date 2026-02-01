import os
import multiprocessing
import sys
from pathlib import Path

from omegaconf import OmegaConf
from training import SelfPredictionTrainingRecipeDistributed


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"

    multiprocessing.set_start_method("spawn", force=True)

    # override the cfg with cli parameters
    cli_cfg = OmegaConf.from_dotlist(sys.argv[1:])
    cli_debug_flag = cli_cfg.pop("debug", False)
    cli_log_experiment_to_disk_flag = cli_cfg.pop("log_experiement_to_disk", False)
    cli_cfg_file = cli_cfg.pop("config_file", None)
    if cli_cfg_file is None:
        raise Exception ("No config file")
    print(f"Loading config file from command line: {cli_cfg_file}")
    cfg = OmegaConf.load(cli_cfg_file)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    cfg.evaluate_every_n_steps = 1000
    cfg.checkpoint_every_n_steps = 1000
    if cfg.get("model", {}).get("self_prediction_module", None) is not None:
        cfg.model.self_prediction_module.save_every_n_steps = 500

    if cli_debug_flag:
        print('Set to debug mode')
        cfg.evaluate_every_n_steps = 5
        cfg.evaluate_n_datapoints = 50
        cfg.checkpoint_every_n_steps = 100000
        cfg.log_every_n_steps = 5
        cfg.metric_logger.mode="disabled"

    assert os.path.exists(Path(cfg.checkpointer.output_dir).parents[0]), f"Checkpoint dir {Path(cfg.checkpointer.output_dir).parents[0]} does not exist"
    assert os.path.exists(cfg.output_dir), f"Output dir {cfg.output_dir} does not exist"

    cfg.dataset.packed_sequence_length = 2048

    cfg.compile = False
    cfg.metric_logger._component_ = "torchtune.training.metric_logging.WandBLogger"
    if cli_log_experiment_to_disk_flag:
        cfg.metric_logger._component_ = "torchtune.training.metric_logging.DiskLogger"
        cfg.metric_logger.log_dir = f"/home/woody/iwbi/iwbi106h/suuraj/codes/predicting-hidden-states/predicting_hidden_states/experiments/{cli_log_experiment_to_disk_flag}"
        cfg.checkpointer.output_dir = cfg.metric_logger.log_dir
        cfg.metric_logger.pop("project")
        cfg.metric_logger.pop("mode")
    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    main()
