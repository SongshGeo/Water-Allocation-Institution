#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import pickle
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cos import notify_me_finished
from mksci_font import mksci_font
from src import PROVINCES_CHN2ENG
from src.plots import plot_pre_post, save_plot
from SyntheticControlMethods import DiffSynth, Synth


def subset_dataset(
    dataset: pd.DataFrame,
    time_var: str,
    id_var: str,
    treated_unit: str,
    outcome_var: str,
    features: Optional[Iterable[str]] = None,
    control_units: Optional[Iterable[str]] = None,
    start: int = None,
    end: int = None,
):
    """Subsets a pandas dataset based on time, id, and features.

    Args:
        dataset (pandas.DataFrame): The input dataset to subset.
        time_var (str): The name of the time variable column in the dataset.
        id_var (str): The name of the id variable column in the dataset.
        treated_unit (str): The treated unit to include in the subset.
        outcome_var (str): The name of the outcome variable column in the dataset.
        features (List[str]): The list of features to include in the subset.
            If None, use all columns except time_var, id_var, and outcome_var.
        control_units (Iterable): The control units to include in the subset.
            If None, all unique values of id_var except the treated_unit will be used.
        start (int): The start time (inclusive) to subset the dataset by.
            If None, the minimum value of time_var will be used.
        end (int): The end time (exclusive) to subset the dataset by.
            If None, the maximum value of time_var will be used.

    Returns:
        pandas.DataFrame: A subset of the input dataset that satisfies the specified conditions.
    """
    # Determine default features if not provided
    if features is None:
        features = dataset.columns.to_list()
    else:
        features = [time_var, id_var, outcome_var, *features]

    # Determine default control_units if not provided
    if control_units is None:
        control_units = dataset[id_var].unique().tolist()
        control_units.remove(treated_unit)

    # Determine default start and end times if not provided
    if start is None:
        start = dataset[time_var].min()
    if end is None:
        end = dataset[time_var].max()

    # Subset the dataset by time, id, and features
    time_mask = (dataset[time_var] >= start) & (dataset[time_var] < end)
    id_mask = dataset[id_var].isin(control_units) | (dataset[id_var] == treated_unit)
    feature_mask = dataset.columns.isin(features)
    return dataset.loc[time_mask & id_mask, feature_mask]


class OneSynth:
    def __init__(
        self,
        dataset,
        outcome_var,
        id_var,
        time_var,
        treated_unit,
        start=None,
        end=None,
        features=None,
        control_units=None,
        n_optim=10,
        pen=0,
        random_seed=0,
    ):
        # Subset the dataset by time, id, and features
        self.data = subset_dataset(
            dataset,
            time_var,
            start=start,
            end=end,
            id_var=id_var,
            treated_unit=treated_unit,
            outcome_var=outcome_var,
            features=features,
            control_units=control_units,
        )

        # Initialize other attributes with the given parameters
        self.outcome_var = outcome_var
        self.id_var = id_var
        self.time_var = time_var
        self.treated_unit = treated_unit
        self.features = features
        self.control_units = control_units
        self.model = None
        self.treated_time = None
        self.parameters = {"n_optim": n_optim, "pen": pen, "random_seed": random_seed}

    def __repr__(self):
        return f"<OneSynth[{len(self.control_units)}]: {self.treated_unit}({self.start}~{self.end})>"

    @property
    def time(self):
        return np.array(sorted(self.data[self.time_var].unique()))

    @property
    def start(self):
        return self.time.min()

    @property
    def end(self):
        return self.time.max()

    @property
    def result(self):
        data = self.model.original_data
        synth = data.synth_outcome.ravel()
        origin = data.treated_outcome_all.ravel()
        return pd.DataFrame({"Synth": synth, "Origin": origin}, index=self.time)

    def do_synth_model(
        self,
        treated_time: int,
        differenced: Optional[bool] = True,
        time_placebo: Optional[int] = None,
        space_placebo: Optional[bool] = False,
        time_placebo_optim: Optional[int] = 5,
        space_placebo_optim: Optional[int] = 3,
    ) -> Synth:
        """
        Fit a Synth or DiffSynth model to the subset of data.

        Args:
            treated_time: A scalar integer representing the time of treatment.
            differenced: A boolean indicating whether to use DiffSynth or Synth.
            time_placebo: An integer representing the time at which to create a time placebo.
            space_placebo: A boolean indicating whether to create a space placebo.
            time_placebo_optim: An integer representing the number of optimization attempts for the time placebo.
            space_placebo_optim: An integer representing the number of optimization attempts for the space placebo.

        Returns:
            A fitted Synth or DiffSynth model object.

        """
        synth_method = DiffSynth if differenced else Synth
        self.treated_time = treated_time
        model = synth_method(
            self.data,
            self.outcome_var,
            self.id_var,
            self.time_var,
            treated_time,
            self.treated_unit,
            **self.parameters,
        )
        self.model = model
        if time_placebo is not None:
            self.time_placebo(time_placebo, n_optim=time_placebo_optim)
        if space_placebo:
            self.space_placebo(n_optim=space_placebo_optim)
        return model

    def time_placebo(self, time: int, n_optim: int = 5) -> None:
        n_optim = self.parameters.get("time_placebo_optim", 5)
        self.model.in_time_placebo(time, n_optim=n_optim)

    def space_placebo(self, n_optim: int = 3) -> None:
        n_optim = self.parameters.get("space_placebo_optim", 3)
        self.model.in_space_placebo(n_optim=n_optim)

    def rmse(self, confident_level=0.99) -> pd.Series:
        from scipy.stats import ttest_1samp

        def judge_sig(col, cl=confident_level):
            placebo = data.loc[1:, col]
            treated = data.loc[0, col]
            res = ttest_1samp(placebo, treated)
            ci = res.confidence_interval(confidence_level=cl)
            return ci

        data = self.model.original_data.rmspe_df
        result = data.loc[0, :].copy()
        # 原假设：在处理之前，安慰剂对照的均值 大于 收到处理的组
        ci = judge_sig("post/pre")
        result["low"], result["high"] = ci.low, ci.high
        # result['pre/post_p-value'] = judge_sig(, 'greater')
        return result

    # @mksci_font(xlabel="均方根误差", ylabel="合成控制实验")
    def plot_rmse(self, cl=0.95, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 4))
        data = self.model.original_data.rmspe_df.sort_values("post/pre")
        units = data["unit"]
        value = data["post/pre"]
        res = self.rmse(cl)
        colors = ["black" if unit == self.treated_unit else "gray" for unit in units]
        ax.barh(units, value, edgecolor="white", color=colors)
        ax.axvspan(res["low"], res["high"], color="red", alpha=0.4)
        ax.set_xlabel("Root Mean Square Error (RMSE)")
        ax.set_ylabel("Synth Control tests")
        ax.set_yticklabels([])
        return ax

    def plot_trend(self, exclusion_multiple=5, ax=None):
        """
        in_space_exclusion_multiple : float, default: 5
        used only in 'in-space placebo' plot.
        excludes all placebos with PRE-treatment rmspe greater than
        the real synthetic control*in_space_exclusion_multiple
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 4))
        data = self.model.original_data
        # plot_pre_post(data, time, self.outcome_var, ax=ax)
        ax.axhline(0, ls="--", color="gray")
        ax.plot(self.time, data.in_space_placebos[0], "0.7")
        for i in range(1, data.n_controls):
            # If the pre rmspe is not more than
            # in_space_exclusion_multiple times larger than synth pre rmspe
            if exclusion_multiple is None:
                ax.plot(self.time, data.in_space_placebos[i], "0.7")

            elif (
                data.rmspe_df["pre_rmspe"].iloc[i]
                < exclusion_multiple * data.rmspe_df["pre_rmspe"].iloc[0]
            ):
                ax.plot(self.time, data.in_space_placebos[i], "0.9")
        ax.axvline(data.treatment_period, linestyle=":", color="gray")
        normalized_treated_outcome = data.treated_outcome_all - data.synth_outcome.T
        ax.plot(self.time, normalized_treated_outcome, "b-")
        ax.set_ylabel(f"Normalized {self.outcome_var}")
        return ax


class MultiSynth:
    def __init__(
        self,
        dataset,
        outcome_var,
        id_var,
        time_var,
        treated_units,
        start=None,
        end=None,
        features=None,
        control_units=None,
        excluded_units=None,
        n_optim=10,
        pen=0,
        random_seed=0,
    ):
        if excluded_units is None:
            excluded_units = []
        if control_units is None:
            control_units = dataset[id_var].unique()
            control_units = control_units[
                ~np.isin(control_units, [*treated_units, *excluded_units])
            ]
        self.units = {}
        self.parameters = {"n_optim": n_optim, "pen": pen, "random_seed": random_seed}
        # Determine default start and end times if not provided
        if start is None:
            start = dataset[time_var].min()
        if end is None:
            end = dataset[time_var].max()
        self.start = start
        self.end = end
        self.treated_time = None
        self.outcome_var = outcome_var

        for unit in treated_units:
            model = OneSynth(
                dataset=dataset,
                outcome_var=outcome_var,
                id_var=id_var,
                time_var=time_var,
                treated_unit=unit,
                start=start,
                end=end,
                features=features,
                control_units=control_units,
                **self.parameters,
            )
            self.units[unit] = model

    def __repr__(self):
        return f"<MultiSynth: {list(self.units)}>"

    @notify_me_finished
    def run_models(
        self,
        treated_time,
        differenced=True,
        time_placebo=None,
        space_placebo=False,
        time_placebo_optim=5,
        space_placebo_optim=3,
    ):
        self.treated_time = treated_time
        for _, model in self.units.items():
            model.do_synth_model(
                treated_time,
                differenced=differenced,
                time_placebo=time_placebo,
                space_placebo=space_placebo,
                time_placebo_optim=time_placebo_optim,
                space_placebo_optim=space_placebo_optim,
            )

    def plot(self, unit, plot_types: Iterable[str]):
        return self.units[unit].model.plot(plot_types)

    @property
    def result(self):
        """
        Transform a pickle experiment result to comparable csv data.
        """
        dataset = []
        for province, model in self.units.items():
            data = model.result.rename(
                {"Synth": f"{province}_synth", "Origin": f"{province}_origin"}, axis=1
            )
            dataset.append(data)
        return pd.concat(dataset, axis=1)

    def outcome(self, item="Synth") -> pd.DataFrame:
        data = {unit: model.result[item] for unit, model in self.units.items()}
        return pd.DataFrame(data)

    def agg_results(self, how):
        def agg_data(data):
            np_func = getattr(np, how)
            return np_func(data, axis=1)

        synth = agg_data(self.outcome("Synth")).rename("Synth")
        actual = agg_data(self.outcome("Origin")).rename("Origin")
        return pd.concat([synth, actual], axis=1)

    def difference(self, how="sum"):
        data = self.agg_results(how).loc[self.treated_time :, :]
        return 100 * (data["Origin"] - data["Synth"]) / data["Synth"]

    @save_plot
    def plot_pre_post(self, how="sum", ax=None, save=None, **kwargs):
        return plot_pre_post(
            self.agg_results(how), self.treated_time, self.outcome_var, ax=ax, **kwargs
        )

    def export_to_pickle(self, file_path):
        """Export the object to a pickle file."""

        with open(file_path, "wb") as model_file:
            pickle.dump(self, model_file)

    @classmethod
    def load_from_pickle(cls, file_path):
        """Load a MultiSynth object from a pickle file."""
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("The pickle file does not contain a MultiSynth object")
        return obj

    def get_statistic_df(self):
        statistic = pd.DataFrame(index=self.units.keys())
        data = self.result
        for unit in self.units:
            diff = data[f"{unit}_origin"] - data[f"{unit}_synth"]
            diff_sum = diff.loc[self.treated_time :].sum()
            statistic.loc[unit, "diff_sum"] = diff_sum
            synth_sum = data[f"{unit}_synth"].loc[self.treated_time :].sum()
            statistic.loc[unit, "diff_ratio"] = diff_sum / abs(synth_sum)
        return statistic

    def pair_t_test(self, when="after", how="sum"):
        from scipy.stats import ttest_rel

        data = self.agg_results(how=how)
        if when == "after":
            slc = slice(self.treated_time, self.end)
        elif when == "before":
            slc = slice(self.start, self.treated_time)
        elif when == "all":
            slc = slice(self.start, self.end)
        synth = data["Synth"].loc[slc].values
        actual = data["Origin"].loc[slc].values
        return ttest_rel(synth, actual)

    def rmse_report(self, confident_level: float = 0.95):
        result = {unit: sc.rmse(confident_level) for unit, sc in self.units.items()}
        result = pd.DataFrame(result).T
        return result

    @save_plot
    def plot_panels(self, how="rmse", figsize=(12, 8), **kwargs):
        supported = ("rmse", "trend")
        if how not in supported:
            raise KeyError(f"Unsupported plot type: {how}.")
        fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
        for i, (unit, sc) in enumerate(self.units.items()):
            ax = axes.flatten()[i]
            ax.set_title(unit)
            ax = getattr(sc, f"plot_{how}")(ax=ax, **kwargs)
            if i not in [0, 4]:
                ax.set_ylabel("")
        return fig
