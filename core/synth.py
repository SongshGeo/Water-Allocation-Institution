#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import logging

import pandas as pd
from omegaconf import DictConfig
from src.multi_synth import MultiSynth

logger = logging.getLogger("Synth")


# @hydra.main(version_base=None, config_path="../config", config_name="synth")
def run_multi_synth(cfg: DictConfig, policy: str) -> MultiSynth:
    # retrieve configuration
    policy = cfg.get(policy)
    start, end, treated = policy.start, policy.end, policy.treat_year
    data = pd.read_csv(cfg.db.pca85, index_col=0)

    # setup model
    model = MultiSynth(
        dataset=data,
        outcome_var=cfg.outcome_var,
        time_var=cfg.time_var,
        id_var=cfg.id_var,
        treated_units=cfg.province_include,
        excluded_units=cfg.province_exclude,
        features=cfg.features,
        start=start,
        end=end,
        pen=cfg.pen,
        n_optim=cfg.n_optim,
        random_seed=cfg.random_seed,
    )
    logger.info(f"开始合成控制法：{policy.name}")
    model.run_models(
        treated_time=treated,
        differenced=cfg.differenced,
        time_placebo=policy.placebo_time,
        space_placebo=cfg.space_placebo,
        time_placebo_optim=cfg.time_placebo_optim,
        space_placebo_optim=cfg.space_placebo_optim,
    )
    logger.info(f"运算结束，导出合成控制结果：{policy.name}")
    return model


if __name__ == "__main__":
    model = run_multi_synth()
