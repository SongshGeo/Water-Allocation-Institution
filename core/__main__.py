#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import logging
import os

import hydra
from overall_analysis import add_statistic_items, get_compare_df

from core.plots import comparison_plot, plot_comprehensive_fig1
from core.synth import run_multi_synth

logger = logging.getLogger("Main")


def output(filename: str) -> str:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    return os.path.join(output_dir, filename)


def policy_exp(code, cfg):
    # 计算1987年的制度
    exp = run_multi_synth(cfg, f"policy_{code}")
    exp.export_to_pickle(output(f"policy_{code}.pkl"))
    exp.plot_panels("trend", figsize=(12, 6), save=cfg.figs.get(f"trend_{code}"))
    exp.plot_panels("rmse", figsize=(12, 6), save=cfg.figs.get(f"rmse_{code}"))
    return exp


@hydra.main(version_base=None, config_path="../config", config_name="synth")
def main(cfg):
    logger.info(f"开始实验：{cfg.description}")
    exp87 = policy_exp(87, cfg=cfg)
    exp98 = policy_exp(98, cfg=cfg)

    logger.info("开始分析")
    statistic_87 = add_statistic_items(exp87, cfg)
    statistic_98 = add_statistic_items(exp98, cfg)

    plot_comprehensive_fig1(exp87, exp98, cfg, save=cfg.figs.results)
    compare_df = get_compare_df(statistic_87, statistic_98)
    comparison_plot(compare_df, save=cfg.figs.compare)

    logger.info(f"实验结束，共绘制了{len(cfg.figs)}张图像。")


if __name__ == "__main__":
    main()
