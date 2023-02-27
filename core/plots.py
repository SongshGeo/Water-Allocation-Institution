#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from src.plots import NATURE_PALETTE, save_plot
from src.tools import get_position_by_ratio

logger = logging.getLogger("Exp_plots")


@save_plot
def plot_outage(outage: pd.DataFrame, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(9, 2))
    bubbles = ax.scatter(
        x=outage["年份"],
        y=outage["断流天数"],
        s=outage["断流长度"] * 0.6,
        color="lightgray",
        edgecolors=NATURE_PALETTE["Nature"],
        linewidth=1.5,
        alpha=0.9,
        label="Drying-up",
        zorder=1,
    )
    ax.set_xlim(1970, 2010)
    ax.set_ylim(-10, 170)
    # ax.axvline(1987, ls=':', color=NATURE_PALETTE['NS'], lw=3, label='Policy 1', zorder=0)
    # ax.axvline(1998, ls=':', color=NATURE_PALETTE['NG'], lw=3, label='Policy 2', zorder=0)
    # ax.axvline(1978, ls='-.', color='gray', lw=1, label='Study period division')
    # ax.axvline(2008, ls='-.', color='gray', lw=1)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    # ax.axvspan(1975, 1987, alpha=0.1, label='Structure 1')
    # ax.axvspan(1987, 1998, alpha=0.1, color='red', label='Mismatched institution')
    # ax.axvspan(1998, 2008, alpha=0.1, color='green', label='Structure 3')

    ax.set_xlabel("Year")
    ax.set_ylabel("Drying-up / days")
    return bubbles


@save_plot
def plot_drought(drought, col, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(9, 2))
    mean_10 = drought.rolling(window=10, min_periods=10, center=True).mean()[col]
    # drought[col].plot.bar(ax=ax)
    ax.bar(
        x=drought[col].index,
        height=drought[col].values,
        color="#e0a418",
        alpha=0.8,
        label="Drought",
        zorder=0,
    )
    ax.plot(
        mean_10.index,
        mean_10.values,
        ls="-.",
        lw=2,
        color=NATURE_PALETTE["NS"],
        label="10yrs-Avg. drought index",
    )
    ax.set_xlim(1970, 2010)
    ax.axhline()
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)


@save_plot
def plot_comprehensive_fig1(exp87, exp98, cfg: DictConfig):
    fig = plt.figure(figsize=(9, 5), constrained_layout=True)

    # prepare plotting
    gs = fig.add_gridspec(
        ncols=3, nrows=2, hspace=0.0, height_ratios=[1.2, 1], width_ratios=[7, 10, 10]
    )

    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    ax3 = ax4.twinx()

    # plots
    exp87.plot_pre_post(ax=ax1, axvline=False)
    exp98.plot_pre_post(ax=ax2, axvline=False)

    outage = pd.read_excel(cfg.db.outage)
    plot_outage(outage, ax=ax3)

    drought = pd.read_excel(cfg.db.drought, index_col=0)
    drought.columns = [f"m{i}" for i in (1, 3, 6, 12)]
    plot_drought(drought, "m1", ax4)
    ax1.legend_.remove()
    ax2.legend_.remove()
    # beauty & annotation

    ax1.axvline(1987, ls="--", color=NATURE_PALETTE["NS"], lw=3, zorder=0)
    ax2.axvline(1998, ls="--", color=NATURE_PALETTE["NG"], lw=3, zorder=0)

    ax4.set_ylabel("Drought index")
    ax3.annotate(
        "",
        xy=(1978, ax3.get_ylim()[1]),
        xycoords="data",
        xytext=(1998, ax3.get_ylim()[1]),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="<->",
            connectionstyle="arc3",
            color=NATURE_PALETTE["NS"],
        ),
    )
    ax3.annotate(
        "",
        xy=(1987, 150),
        xycoords="data",
        xytext=(2008, 150),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="<->",
            connectionstyle="arc3",
            color=NATURE_PALETTE["NG"],
        ),
    )

    ax4.axvline(
        1987, ls="--", color=NATURE_PALETTE["NS"], lw=3, label="87-WAS", zorder=0
    )
    ax4.axvline(
        1998, ls="--", color=NATURE_PALETTE["NG"], lw=3, label="98-UBR", zorder=0
    )
    ax4.axvline(1978, ls="-.", color="gray", lw=1, label="Study period division")
    # ax3.text("")
    legend_handles = []
    legend_labels = []
    for handle, label in zip(*ax2.get_legend_handles_labels()):
        legend_handles.append(handle)
        legend_labels.append(label)
    for handle, label in zip(*ax3.get_legend_handles_labels()):
        legend_handles.append(handle)
        legend_labels.append(label)
    for handle, label in zip(*ax4.get_legend_handles_labels()):
        legend_handles.append(handle)
        legend_labels.append(label)

    ax3.text(
        1986.5,
        155,
        "IS1: 87-WAS",
        color=NATURE_PALETTE["NS"],
        horizontalalignment="right",
        weight="bold",
    )
    ax3.text(
        1998.5,
        155,
        "IS2: 98-UBR",
        color=NATURE_PALETTE["NG"],
        horizontalalignment="left",
        weight="bold",
    )

    for ax in [ax1, ax2]:
        ax.set_yticks(range(70, 111, 10))

    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.grid(True, color="lightgray", ls=":")
        ax.tick_params(direction="in")
        x, y = get_position_by_ratio(ax, 0.05, 0.95)
        ax.text(x, y, ("A", "B", "C")[i], weight="bold", fontsize=12)

    fig.legend(
        loc="upper left",
        handles=legend_handles,
        labels=legend_labels,
        handletextpad=1.5,
        handleheight=1.5,
        markerscale=0.7,
    )
    return fig


@save_plot
def comparison_plot(compare_df):
    from matplotlib import ticker

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    fig, ax1 = plt.subplots(figsize=(7, 4))

    width = 0.36
    x = np.arange(8)
    index = compare_df.index
    ax2 = ax1.twinx()

    ax2.bar(
        x - width / 2,
        width=width,
        height=compare_df["wu_ratio_87"],
        color="lightgray",
        alpha=0.4,
        edgecolor="white",
        hatch="xxx",
        label="WU ratio",
        zorder=0,
    )
    ax2.bar(
        x + width / 2,
        width=width,
        height=compare_df["wu_ratio_98"],
        color="lightgray",
        alpha=0.4,
        edgecolor="white",
        hatch="xxx",
        zorder=0,
    )

    ax2.scatter(
        range(8),
        compare_df["base"],
        marker="^",
        color="red",
        edgecolor="white",
        s=80,
        label="Quota",
    )

    ax1.bar(
        x - width / 2,
        width=width,
        height=compare_df["87_ratio"],
        color="#C83C1C",
        alpha=0.8,
        zorder=1,
        label="87-WAS",
        edgecolor="white",
    )
    ax1.bar(
        x + width / 2,
        width=width,
        height=compare_df["98_ratio"],
        color="#006C43",
        alpha=0.8,
        zorder=1,
        label="98-UBR",
        edgecolor="white",
    )

    # 坐标轴距离
    ax1.set_yticks(np.arange(-0.5, 0.51, 0.25))
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlim(-0.8, 7.8)
    ax2.set_yticks(np.arange(-0.4, 0.41, 0.1))
    ax2.set_ylim(-0.4, 0.4)
    ax2.set_yticklabels(["", "", "", "", "0%", "10%", "20%", "30%", "40%"])

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["bottom"].set_visible(False)

    xticklabels = index.to_list()
    ax1.set_xticks(range(len(xticklabels)))
    ax1.set_xticklabels(xticklabels)
    ax1.set_ylabel("Extra WU over the estimation")
    ax2.set_ylabel("WU ratio")
    #### 辅助线 ========
    ax1.axhline(0, ls="--", color="gray", lw=1, zorder=0)
    ax1.annotate(
        "",
        xy=(0 - width, -0.4),
        xycoords="data",
        xytext=(3 + width, -0.4),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="<->", connectionstyle="arc3", color="black", lw=1.5
        ),
    )
    ax1.text(
        1.5,
        -0.42,
        "Major water users",
        color="black",
        horizontalalignment="center",
        verticalalignment="top",
    )
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # -=====  图例

    legend_handles = []
    legend_labels = []
    for handle, label in zip(*ax1.get_legend_handles_labels()):
        legend_handles.append(handle)
        legend_labels.append(label)

    for handle, label in zip(*ax2.get_legend_handles_labels()):
        legend_handles.append(handle)
        legend_labels.append(label)

    fig.legend(
        loc=(0.7, 0.73),
        frameon=False,
        handles=legend_handles,
        labels=legend_labels,
        handletextpad=1.5,
    )
    return fig
