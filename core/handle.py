#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/4
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import os

import numpy as np
import pandas as pd
import seaborn as sns
from attrs import define, field
from experiment import Experiment
from matplotlib import pyplot as plt
from src.plots import NATURE_PALETTE


@define
class ExpResultsHandler(Experiment):
    """
    Experiment handler, for exploring results easier.
    """

    dfs: dict = field(factory=dict, repr=False)
    provinces: list = field(factory=list, repr=False)
    time_placebo_results: dict = field(factory=dict, repr=False)

    def __init__(self, experiment=None, yaml_file=None):
        if experiment:
            yaml_file = experiment.paths.get("yaml")
        if yaml_file:
            super().__init__(yaml_file)
        self.dfs = {}
        self.provinces = []
        self.time_placebo_results = {}

    def plot_grid(self, panel_df, save_path=None):
        grid = sns.FacetGrid(
            panel_df,
            col="Province",
            col_wrap=3,
            height=3,
        )

        grid.map(plt.axvline, x=self.parameters.get("treat_year"), ls=":", c=".5")
        grid.map(
            plt.plot,
            "Year",
            "Observation",
            marker=".",
            label="Observation",
            color="black",
        )
        grid.map(
            plt.plot,
            "Year",
            "Prediction",
            marker=".",
            label="Prediction",
            color="gray",
        )
        grid.add_legend()
        if save_path:
            plt.savefig(save_path, dpi=300)

    def original_plots(self, province, save=True):
        # TODO put this coretions into handle class.
        if save:
            # 储存图像路径
            figs_folder = os.path.join(self.exp_path, "figs")
            if not os.path.exists(figs_folder):
                os.mkdir(figs_folder)

            # 图像展示
            self.result[province].plot(
                ["original", "pointwise", "cumulative"],
                treated_label=province,
                synth_label=f"Synthetic {province}",
                treatment_label=f"Treatment in {self.parameters['treat_year']}",
                save_path=os.path.join(figs_folder, f"{province}.jpg"),
            )
        else:
            # 图像展示
            self.result[province].plot(
                ["original", "pointwise", "cumulative"],
                treated_label=province,
                synth_label=f"Synthetic {province}",
                treatment_label=f"Treatment in {self.parameters['treat_year']}",
            )

    def weight_df(self, province="all"):
        if province == "all":
            possible_provinces = self.dfs.get("merged_data").Province.unique()
            weights = pd.DataFrame(
                data=0.0, index=possible_provinces, columns=self.provinces
            )
            for p in self.provinces:
                weight_ser = self.result[p].original_data.weight_df["Weight"]
                weights.loc[weight_ser.index, p] = weight_ser
        else:
            weights = self.result[province].original_data.weight_df
        return weights

    def original_comparison(self, province="all"):
        if province == "all":
            comparison = pd.DataFrame()
            for p in self.result.keys():
                comparison_df = self.result[p].original_data.comparison_df
                # 把获取的结果储存到同一个表中
                for col in comparison_df:
                    # WMAPE is a indicator to stimulation of Synth Control.
                    if col == "WMAPE":
                        comparison[col + "_" + p] = comparison_df[col]
                    else:
                        comparison[col] = comparison_df[col]
        else:
            comparison = self.result[province].original_data.comparison_df
        return comparison

    def plot_upset(
        self,
        items,
        threshold=0.05,
        figszie=(8, 4),
        height_ratio=(3, 1.5),
        sort=True,
    ):
        fig = plt.figure(figsize=figszie, constrained_layout=False)
        plt.subplots_adjust(
            left=None,
            bottom=None,
            right=None,
            top=None,
            wspace=None,
            hspace=None,
        )
        gs = fig.add_gridspec(
            ncols=4,
            nrows=2,
            wspace=0.65,
            hspace=0.2,
            height_ratios=height_ratio,
        )
        ax2 = fig.add_subplot(gs[1, 1:])
        ax1 = fig.add_subplot(gs[0, 1:])
        ax3 = fig.add_subplot(gs[1, :1])

        for ax in [ax1, ax2, ax3]:
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax1.spines["left"].set_visible(True)
        ax1.tick_params(bottom=True, top=False, left=True, right=False)
        ax3.spines["bottom"].set_visible(True)
        ax3.tick_params(bottom=True, top=False, left=False, right=False)
        ax3.set_yticklabels("")
        ax1.set_ylabel("Increased ratio of WU")
        ax2.set_yticks(range(3))
        ax2.set_ylim(-0.5, 2.5)
        ax3.set_ylim(-0.5, 2.5)
        ax1.grid(True, axis="y", ls=":", color=NATURE_PALETTE["Nature"])
        ax3.set_xlim(0, 1)

        for i in range(3):
            ax2.axhspan(i - 0.3, i + 0.3, color=NATURE_PALETTE["Nature"], alpha=0.05)

        ax2.set_xlabel("")
        ax2.set_xticklabels("")
        ax3.set_title("Correlation", size=10)
        ax2.set_xlabel("Provinces in the YRB")

        ratio = self.dfs["statistic"]["diff_ratio"]
        if sort:
            ratio = ratio.sort_values(ascending=False)

        ax1.bar(
            range(len(ratio)),
            height=ratio,
            width=0.8,
            align="center",
            color=NATURE_PALETTE["NCC"],
            edgecolor=NATURE_PALETTE["Nature"],
        )
        ax1.set_xticks(range(len(ratio)))
        ax1.set_xticklabels(ratio.index, size=9)
        ax1.set_xlim(-0.5, len(ratio) - 0.5)
        ax1.set_ylim(ratio.min(), ratio.max())

        point_color = NATURE_PALETTE["NS"]
        labels = []
        for index, (label, bools, note, corr, p_val) in enumerate(items):
            bools = bools[ratio.index]
            true = []
            colors = [point_color if i else "gray" for i in bools]
            ax2.scatter(
                range(len(bools)),
                index * np.ones(shape=(len(bools),)),
                edgecolors="white",
                color=colors,
                s=160,
            )
            ax3.barh(
                y=index,
                width=corr,
                height=0.6,
                left=1 - corr,
                align="center",
                color=NATURE_PALETTE["Nature"],
            )
            if p_val < threshold:
                ax3.text(
                    s="*",
                    x=1 - corr - 0.1,
                    y=index - 0.15,
                    weight="bold",
                    color=NATURE_PALETTE["NS"],
                    size=10,
                )
            labels.append(label)
            for i, b in enumerate(bools):
                if b:
                    ax2.arrow(x=i, y=index, dx=0, dy=-0.3, color=point_color)
                    true.append(i)
            ax2.arrow(
                x=min(true),
                y=index - 0.3,
                dx=max(true) - min(true),
                dy=0,
                color=point_color,
            )
            ax2.text(
                s=note,
                x=(max(true) + min(true)) / 2,
                y=index - 0.5,
                fontstyle="italic",
                horizontalalignment="center",
                verticalalignment="center",
            )

        ax2.set_xlim(ax1.get_xlim())
        ax2.set_yticks(range(len(items)))
        ax2.set_ylim(-0.5, len(items) - 0.5)
        ax3.set_ylim(-0.5, len(items) - 0.5)
        ax2.set_yticklabels(labels)
        ax3.set_xticks(ticks=np.arange(0, 1.1, 0.5), labels=["1.0", "0.5", "0.0"])
        return ax1, ax2, ax3
