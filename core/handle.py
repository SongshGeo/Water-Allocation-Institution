#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/4
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9
import copy
import inspect
import os

import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from attrs import define, field
from matplotlib import pyplot as plt

from .experiment import Experiment
from .plots import NATURE_PALETTE, basic_plot
from .src.tools import extract_mean_std, get_optimal_fit_linear


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

    def load_processed_datasets(self):
        """Load processed datasets from disk."""
        for key in self.datasets.keys():
            data = pd.read_csv(self.datasets.get(key), index_col=0)
            self.dfs[key] = data
            self.log.info(f"{key} dataset loaded from disk.")
        return self.dfs.keys()

    def load_from_pickle(self):
        """Load exp results from pickle."""
        checks = [inspect.ismethod, inspect.isbuiltin]
        experiment = super().load_from_pickle()
        for attr in dir(experiment):
            obj = getattr(experiment, attr)
            if attr.startswith("__"):
                continue
            if any(check(obj) for check in checks):
                continue
            val = getattr(experiment, attr)
            setattr(self, attr, val)
        self.state = "loaded"
        self.load_processed_datasets()
        self.provinces = list(self.result.keys())
        return self

    def transfer_exp_pickle_to_data(self, save=False):
        """
        Transform a pickle experiment result to comparable csv data.
        """
        params = self.parameters
        years = np.arange(params["start"], params["end"] + 1)
        dataset = pd.DataFrame(index=years)
        for province, synth_result in self.result.items():
            data = synth_result.original_data
            synth_data = data.synth_outcome.T.ravel()  # Synth label
            actual_data = data.treated_outcome_all.ravel()  # original data

            dataset[f"{province}_synth"] = synth_data
            dataset[f"{province}_actual"] = actual_data
        if save:
            self._extracted_from_transfer_exp_pickle_to_data_16(dataset)
        return dataset

    # TODO Rename this here and in `transfer_exp_pickle_to_data`
    def _extracted_from_transfer_exp_pickle_to_data_16(self, dataset):
        path = os.path.join(
            self.paths.get("results"),
            f"{self.name}_diff.csv",
        )
        self.log.info("Saved diff dataset into csv.")
        dataset.to_csv(path)
        self.add_item("dfs", "diff", dataset)
        self.add_item("paths", "diff_csv", path)
        self.add_item("datasets", "diff_csv", path)

    def get_statistic_df(self, save=False):
        provinces = self.result.keys()
        data = self.transfer_exp_pickle_to_data()
        start, end = self.parameters.get("treat_year"), self.parameters.get("end")
        statistic = pd.DataFrame(index=provinces)
        for province in provinces:
            diff = data[f"{province}_actual"] - data[f"{province}_synth"]
            diff_sum = diff.loc[start:end].sum()
            statistic.loc[province, "diff_sum"] = diff_sum
            synth_sum = data[f"{province}_synth"].loc[start:end].sum()
            statistic.loc[province, "diff_ratio"] = diff_sum / abs(synth_sum)

        wu_total = self.dfs.get("wu_all")
        wu_yr = self.dfs.get("wu_yr")
        ratio = self.dfs.get("ratio")
        quota = self.dfs.get("quota")
        plan = self.dfs.get("plan")

        statistic["Total_WU"] = extract_mean_std(wu_total, start, end)[0]
        statistic["YR_WU"] = extract_mean_std(wu_yr, start, end)[0]
        statistic["ratio"] = extract_mean_std(ratio, start, end)[0]

        statistic["scheme83"] = quota.loc[1983, :]
        statistic["scheme87"] = quota.loc[1987, :]
        statistic["plan"] = plan.loc["Sum", :]
        statistic["satisfied"] = statistic["scheme87"] / statistic["plan"]
        statistic["unsatisfied"] = 1 - statistic["satisfied"]
        statistic["stress"] = statistic["unsatisfied"] * statistic["YR_WU"]

        statistic["diff_ratio"] = statistic["diff_sum"] / abs(
            statistic["diff_sum"].sum()
        )
        statistic["punished"] = statistic["scheme83"] - statistic["scheme87"]

        statistic = statistic.sort_values("stress", ascending=False)
        self.add_item("dfs", "statistic", statistic)
        if save:
            path = os.path.join(
                self.get_path("results", absolute=True), "statistic.csv"
            )
            self.add_item("paths", "statistic", path)
            self.add_item("datasets", "statistic", path)
            self.log.info("Statistic dataframe saved.")
        return statistic

    def correlation_analysis(
        self, xs, y, ax=None, covar=True, method="pearson", **kwargs
    ):
        data = self.dfs.get("statistic")
        if not ax:
            fig, ax = plt.subplots()
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
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(len(r_results)))
        ax.set_xticklabels(xs)
        ax.set_ylabel(f"Partial Corr to {y}")
        return r_results, p_val

    def plot(self, panels, province):
        sc = self.result.get(province)
        sc.plot(panels)

    def panel_plots(self, how="original", save_path=None, **kwargs):
        fig, (axs1, axs2) = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        axs = []
        axs.extend(axs1)
        axs.extend(axs2)
        for province, ax in zip(self.provinces, axs):
            sc = self.result.get(province)
            basic_plot(how=how, ax=ax, sc=sc, **kwargs)
            ax.set_xlabel(f"{province}")
        if save_path:
            self.log.info(f"Panel plots saved in {save_path}.")
            fig.save(save_path, dpi=300)

    def outcome_panel_data(self, outcome_var, save=False, plot=False):
        params = self.parameters
        datasets = []
        for province in self.provinces:
            sc = self.result.get(province)
            data = sc.original_data
            years = np.arange(params["start"], params["end"] + 1)
            synth_data = data.synth_outcome.T.ravel()  # Synth label
            actual_data = data.treated_outcome_all.ravel()  # original data
            df = pd.DataFrame(
                {
                    "Year": years,
                    "Prediction": synth_data,
                    "Observation": actual_data,
                    "Province": province,
                }
            )
            datasets.append(df)
        dataset = pd.concat(datasets)
        if save:
            self._extracted_from_outcome_panel_data_21(outcome_var, dataset)
        if plot:
            self.plot_grid(dataset)
        return dataset

    # TODO Rename this here and in `outcome_panel_data`
    def _extracted_from_outcome_panel_data_21(self, outcome_var, dataset):
        name = f"panel_{outcome_var}"
        path = os.path.join(self.get_path("results", absolute=True), f"{name}.csv")
        dataset.to_csv(path)
        self.add_item("paths", name, path)
        self.add_item("datasets", name, path)
        self.log.info(f"{name} dataframe saved.")

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

    def do_in_time_placebo(self, placebo_time=None, n_optim=None):
        if not n_optim:
            n_optim = self.parameters.get("placebo_optim")
        if not placebo_time:
            placebo_time = self.parameters.get("placebo_time")
        for province in self.provinces:
            self.result[province].in_time_placebo(placebo_time, n_optim=n_optim)
        return self.result

    def do_in_place_placebo(self, n_optim=None):
        if not n_optim:
            n_optim = self.parameters.get("placebo_optim")
        for province in self.provinces:
            self.result[province].in_space_placebo(n_optim=n_optim)
        return self.result

    def iter_time_placebo(self, span=1):
        original_time = self.parameters.get("treat_year")
        for year in range(original_time - span, original_time):
            exp_copy = copy.deepcopy(self)
            result = exp_copy.do_in_time_placebo(placebo_time=year)
            self.time_placebo_results[year] = result
            self.log.info(f"In time placebo done in {year}.")

    def summarize_analysis(self):
        df_synth = pd.DataFrame()
        df_actual = pd.DataFrame()
        diff_data = self.transfer_exp_pickle_to_data(save=True)
        for col in diff_data:
            if "synth" in col:
                df_synth[col] = diff_data[col]
            if "actual" in col:
                df_actual[col] = diff_data[col]
        yr_synth = df_synth.sum(axis=1)
        yr_actual = df_actual.sum(axis=1)
        return yr_synth, yr_actual

    def rmspe_analysis(self, two_bars=False, plot=True, ax=None):
        rmspe_others = []
        rmspe_provinces = []
        for province in self.provinces:
            rmspe_df = self.result.get(province).original_data.rmspe_df
            others = rmspe_df[rmspe_df["unit"] != province]
            mean = others["post/pre"].mean()
            rmspe = rmspe_df.set_index("unit").loc[province, "post/pre"]
            rmspe_provinces.append(rmspe)
            rmspe_others.append(mean)
        if two_bars:
            rmspe = pd.Series(np.mean(rmspe_provinces), index=["provinces"])
        else:
            rmspe = pd.Series(rmspe_provinces, index=self.provinces)
        rmspe["others"] = np.mean(rmspe_others)
        if plot:
            if not ax:
                _, ax = plt.subplots()
            rmspe.plot.bar(ax=ax)
        return rmspe

    def weight_gdp_analysis(self):
        # TODO finish this
        pass

    def plot_pre_post(
        self,
        ylabel=None,
        actual=None,
        synth=None,
        ax=None,
        figsize=(4, 3),
        axvline=True,
    ):
        if not actual and not synth:
            synth, actual = self.summarize_analysis()
            ylabel = self.parameters.get("Y_inputs")
        treat = self.parameters.get("treat_year")
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


if __name__ == "__main__":
    pass
