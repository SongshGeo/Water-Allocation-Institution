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
import pingouin as pg
from cos import notify_me_finished
from SyntheticControlMethods import DiffSynth, Synth

from .plots import basic_plot, plot_pre_post


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

    # def diff_plot(self, ax, bp):
    #     data = self.original_data
    #     time = data.dataset[data.time].unique()
    #     treat = data.treatment_period
    #     synthetic = data.synth_outcome.T  # Synthetic
    #     original = data.treated_outcome_all  # Original
    #     temp_data = pd.DataFrame(
    #         data={
    #             "time": time,
    #             "origin": original,
    #             "synth": synthetic,
    #         }
    #     )
    #     before = temp_data[temp_data["time"] <= treat]
    #     after = temp_data[temp_data["time"] > treat]
    #     bp.add_element(
    #         data=(before["time"], before["origin"]),
    #         ax=ax,
    #         how="plot",
    #         label="before_observation",
    #     )
    #     bp.add_element(
    #         data=(before["time"], before["synth"]),
    #         ax=ax,
    #         how="plot",
    #         label="before_prediction",
    #     )
    #     bp.add_element(
    #         data=(after["time"], after["origin"]),
    #         ax=ax,
    #         how="linear_fit",
    #         label="after_observation",
    #     )
    #     bp.add_element(
    #         data=(after["time"], after["synth"]),
    #         ax=ax,
    #         how="linear_fit",
    #         label="after_prediction",
    #     )
    #     bp.add_element(
    #         data=[treat], ax=ax, how="axvline", label=f"IS: {treat}"
    #     )
    #     return bp


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

    def plot_pre_post(self, how="sum", ax=None, **kwargs):
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

    def panel_plots(self, how="original", save_path=None, **kwargs):
        fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        axs = axs.flatten()
        for unit, ax in zip(self.units, axs):
            sc = self.units.get(unit).model
            basic_plot(how=how, ax=ax, sc=sc, **kwargs)
            ax.set_xlabel(f"{unit}")
        return fig

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
