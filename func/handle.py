#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/4
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import inspect
import os

import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from attrs import define, field
from experiment import Experiment
from matplotlib import pyplot as plt
from plots import basic_plot
from tools import extract_mean_std


@define
class ExpResultsHandler(Experiment):
    """
    Experiment handler, for exploring results easier.
    """

    dfs: dict = field(factory=dict, repr=False)
    provinces: list = field(factory=list, repr=False)

    def __init__(self, yaml_file):
        super().__init__(yaml_file)
        self.dfs = {}
        self.provinces = []

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
            if any([check(obj) for check in checks]):
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
            path = os.path.join(
                self.paths.get("results"),
                f"{self.name}_diff.csv",
            )
            self.log.info("Saved diff dataset into csv.")
            dataset.to_csv(path)
            self.add_item("dfs", "diff", dataset)
            self.add_item("paths", "diff_csv", path)
            self.add_item("datasets", "diff_csv", path)
        return dataset

    def get_statistic_df(self, save=False):
        provinces = self.result.keys()
        data = self.transfer_exp_pickle_to_data()
        start, end = self.parameters.get("start"), self.parameters.get("end")
        statistic = pd.DataFrame(index=provinces)
        for province in provinces:
            diff = data[f"{province}_actual"] - data[f"{province}_synth"]
            diff_sum = diff.loc[start:end].sum()
            statistic.loc[province, "diff_sum"] = diff_sum
            synth_sum = data[f"{province}_synth"].loc[start:end].sum()
            statistic.loc[province, "diff_ratio"] = diff_sum / synth_sum

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

        statistic["diff_ratio"] = (
            statistic["diff_sum"] / statistic["diff_sum"].sum()
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

    def correlation_analysis(self, xs, y, ax=None, method="pearson", **kwargs):
        data = self.dfs.get("statistic")
        if not ax:
            fig, ax = plt.subplots()
        p_val = []
        r_results = []
        for x in xs:
            covar = xs.copy()
            covar.remove(x)
            result = pg.partial_corr(
                data=data, x=x, y=y, covar=covar, method=method, **kwargs
            )
            # result = pg.corr(x=data[x], y=data[y], **kwargs)
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

    def panel_plots(self, how="original", save_path=None):
        fig, (axs1, axs2) = plt.subplots(2, 4, figsize=(16, 8))
        axs = []
        axs.extend(axs1)
        axs.extend(axs2)
        for province, ax in zip(self.provinces, axs):
            sc = self.result.get(province)
            basic_plot(how=how, ax=ax, sc=sc)
        pass

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
            name = f"panel_{outcome_var}"
            path = os.path.join(
                self.get_path("results", absolute=True), f"{name}.csv"
            )
            dataset.to_csv(path)
            self.add_item("paths", name, path)
            self.add_item("datasets", name, path)
            self.log.info(f"{name} dataframe saved.")
        if plot:
            self.plot_grid(dataset)
        return dataset

    def plot_grid(self, panel_df, save_path=None):
        grid = sns.FacetGrid(
            panel_df,
            col="Province",
            col_wrap=3,
            height=3,
        )

        grid.map(
            plt.axvline, x=self.parameters.get("treat_year"), ls=":", c=".5"
        )
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
        # TODO put this functions into handle class.
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
            self.result[province].in_time_placebo(
                placebo_time, n_optim=n_optim
            )

    def do_in_place_placebo(self, n_optim=None):
        if not n_optim:
            n_optim = self.parameters.get("placebo_optim")
        for province in self.provinces:
            self.result[province].in_space_placebo(n_optim=n_optim)
        pass

    def drop_compared_datasets(self):
        pass

    pass
