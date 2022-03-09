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
from attrs import define, field
from experiment import Experiment
from matplotlib import pyplot as plt
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

        # save datasets
        dataset = pd.DataFrame(
            index=np.arange(params["start"], params["end"] + 1)
        )  # time
        for province, synth_result in self.result.items():
            synth_data = (
                synth_result.original_data.synth_outcome.T
            )  # Synth label
            actual_data = (
                synth_result.original_data.treated_outcome_all
            )  # original data
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
            weights = {}
            for p in self.result.keys():
                weights[p] = self.result[p].original_data.weight_df
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

    def do_in_time_placebo(self, placebo_time, province, n_optim=100):
        self.result[province].in_time_placebo(placebo_time, n_optim=n_optim)
        self.result[province].plot(
            ["in-time placebo"],
            treated_label=province,
            synth_label=f"Synthetic {province}",
        )

    def drop_compared_datasets(self):
        pass

    pass
