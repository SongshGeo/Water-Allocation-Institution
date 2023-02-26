#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import logging

import pandas as pd
import scipy
from src.multi_synth import MultiSynth

logger = logging.getLogger("Analysis")


def analysis_exp(exp: MultiSynth):
    # speed
    k = exp.plot_pre_post()
    observation = k["k_obs"]
    estimation = k["k_syn"]
    ratio = 100 * (observation - estimation) / estimation

    # total water use
    res = exp.agg_results("sum").loc[exp.treated_time + 1 :]
    yrs = len(res)
    obs, syn = res["Origin"].sum(), res["Synth"].sum()

    # T-test
    _, p_value_before = exp.pair_t_test("before")
    _, p_value_after = exp.pair_t_test("after")

    # report
    msg = rf"""
    1. 政策实施后用水量的变化速度：观测值{observation:.2f}$km^3/yrs$，预测值{estimation:.2f}$km^3/yrs$，观测值比预测值高出{ratio:.2f}\%。
    2. 从{exp.treated_time}年到{exp.end}年，在政策实施后{yrs}年间的总用水量：观测值{obs:.2f}$km^3$，预测值{syn:.2f}$km^3$，观测值比预测值高出{(obs-syn)/syn*100:.2f}\%。
    3. 观测-预测之间用水量差异的T检验结果，在政策实施前：{p_value_before:.2f}；在政策实施之后：{p_value_after:.3f}。
    """
    logger.debug("msg")
    return msg


def analysis_drought(drought):
    droughts_p1 = drought["m1"].loc[1988:1998]
    droughts_p2 = drought["m1"].loc[1998:2008]
    t, pval = scipy.stats.ttest_ind(droughts_p1, droughts_p2)

    rep = f"""
    两个时段干旱的差距; 1998~2008:
    - 1988~1998: {droughts_p1.mean()}
    - 1998~2008: {droughts_p2.mean()}

    T 检验：
    两者的 t-value: {t}, p-value: {pval}
    """
    logger.info(rep)
    return rep


def add_statistic_items(exp, cfg):
    def extract_mean_std(data, start, end):
        data = data.loc[start:end, :]
        return data.mean(), data.std()

    # read datasets
    wu_yr = pd.read_csv(cfg.db.wu_yr, index_col=0)
    wu_total = pd.read_csv(cfg.db.wu_all, index_col=0)
    ratio = pd.read_csv(cfg.db.ratio, index_col=0)
    plan = pd.read_csv(cfg.db.plan, index_col=0)
    quota = pd.read_csv(cfg.db.quota, index_col=0)

    statistic = exp.get_statistic_df()
    start, end = exp.treated_time, exp.end
    statistic["Total_WU"] = extract_mean_std(wu_total, start, end)[0]
    statistic["YR_WU"] = extract_mean_std(wu_yr, start, end)[0]
    statistic["ratio"] = extract_mean_std(ratio, start, end)[0]

    statistic["scheme83"] = quota.loc[1983, :]
    statistic["scheme87"] = quota.loc[1987, :]
    statistic["plan"] = plan.loc["Sum", :]
    statistic["satisfied"] = statistic["scheme87"] / statistic["plan"]
    statistic["unsatisfied"] = 1 - statistic["satisfied"]
    statistic["stress"] = statistic["unsatisfied"] * statistic["YR_WU"]

    statistic["diff_ratio"] = statistic["diff_sum"] / abs(statistic["diff_sum"].sum())
    statistic["punished"] = statistic["scheme83"] - statistic["scheme87"]

    statistic = statistic.sort_values("stress", ascending=False)
    return statistic


def get_compare_df(statistic_87, statistic_98):
    compare_df = pd.DataFrame()
    compare_df["87_ratio"] = statistic_87["diff_ratio"]
    compare_df["98_ratio"] = statistic_98["diff_ratio"]
    compare_df["wu_ratio_87"] = statistic_87["YR_WU"] / statistic_87["YR_WU"].sum()
    compare_df["wu_ratio_98"] = statistic_98["YR_WU"] / statistic_98["YR_WU"].sum()
    compare_df["YR_WU"] = (compare_df["wu_ratio_87"] + compare_df["wu_ratio_98"]) / 2
    compare_df = compare_df.sort_values("YR_WU", ascending=False)
    compare_df["base"] = statistic_87["scheme87"] / statistic_87["scheme87"].sum()
    return compare_df
