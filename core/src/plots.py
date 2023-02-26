#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import functools
from typing import Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg

from .tools import get_optimal_fit_linear


def save_plot(func):
    @functools.wraps(func)
    def wrapper(*args, save: Union[str, Iterable[str]] = None, **kwargs):
        result = func(*args, **kwargs)
        if save is not None:
            if isinstance(save, str):
                save = [save]
            for path in save:
                plt.gcf().savefig(path, dpi=300)
        return result

    return wrapper


# 自定义配色
NATURE_PALETTE = {
    "NS": "#c83c1c",
    "Nature": "#29303c",
    "NCC": "#0889a6",
    "NC": "#f1801f",
    "NG": "#006c43",
    "NHB": "#1951A0",
    "NEE": "#C7D530",
}


def plot_pre_post(
    data,
    treat,
    ylabel,
    ax=None,
    figsize=(4, 3),
    axvline=True,
):
    synth, actual = data["Synth"], data["Origin"]
    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    # prepare data
    obs_pre = actual.loc[:treat]
    obs_post = actual.loc[treat:]
    syn_pre = synth.loc[:treat]
    syn_post = synth.loc[treat:]

    # plots
    ax.plot(
        obs_pre.index,
        obs_pre,
        color=NATURE_PALETTE["NCC"],
        marker="o",
        lw=2,
        zorder=2,
        label="Observation",
    )
    ax.plot(
        syn_pre.index,
        syn_pre,
        color="lightgray",
        marker="o",
        lw=2,
        zorder=1,
        label="Prediction",
    )
    ax.scatter(
        syn_post.index,
        syn_post,
        color="gray",
        edgecolor=NATURE_PALETTE["Nature"],
        alpha=0.4,
        s=50,
    )
    ax.scatter(
        obs_post.index,
        obs_post,
        color=NATURE_PALETTE["NCC"],
        edgecolor=NATURE_PALETTE["NCC"],
        s=50,
        alpha=0.4,
    )
    y_sim_obs, k_obs = get_optimal_fit_linear(obs_post.index, obs_post.values)
    ax.plot(
        obs_post.index,
        y_sim_obs,
        color=NATURE_PALETTE["NCC"],
        lw=2,
        ls="--",
    )
    y_sim_syn, k_syn = get_optimal_fit_linear(syn_post.index, syn_post.values)
    ax.plot(
        syn_post.index,
        y_sim_syn,
        color="gray",
        lw=2,
        ls="--",
    )

    # ticks
    x_min, x_max = ax.get_xlim()
    # ax.axvspan(x_min, treat, color=NATURE_PALETTE['NG'], alpha=0.4)
    ax.set_xticks([(treat + x_min) / 2, (x_max + treat) / 2])
    ax.set_xticklabels(["Before", "After"])
    if axvline:
        ax.axvline(
            treat,
            ls=":",
            lw=4,
            color=NATURE_PALETTE["NG"],
            label=f"Policy: {treat}",
        )
    ax.set_xlabel("")

    # spines visibility
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.set_ylabel(ylabel=ylabel)
    ax.set_xlabel("")
    return {"k_obs": k_obs, "k_syn": k_syn}


def correlation_analysis(data, xs, y, ax=None, covar=True, method="pearson", **kwargs):
    if not ax:
        _, ax = plt.subplots()
    p_val = []
    r_results = []
    for x in xs:
        if covar:
            covar = xs.copy()
            covar.remove(x)
            result = pg.partial_corr(
                data=data, x=x, y=y, covar=covar, method=method, **kwargs
            )
        else:
            result = pg.corr(data[x], data[y], method=method)
        r_results.append(result.loc[method, "r"])
        p_val.append(result.loc[method, "p-val"])
    ax.bar(x=np.arange(len(r_results)), height=r_results)
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.arange(len(r_results)))
    ax.set_xticklabels(xs)
    ax.set_ylabel(f"Partial Corr to {y}")
    return r_results, p_val
