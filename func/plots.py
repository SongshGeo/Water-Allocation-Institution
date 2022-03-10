#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/4
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9


import numpy as np
from attrs import define, field

from func.tools import get_optimal_fit_linear

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


@define
class BeautyPlot(object):
    elements: dict = field(factory=dict, repr=True)
    beauty_dicts: dict = field(factory=dict, repr=False)

    def add_beauty_dict(self, label, beauty_dict):
        self.beauty_dicts[label] = beauty_dict
        return f"{beauty_dict} added."

    def get_beauty_dict(self, label):
        return self.beauty_dicts.get(label)

    def add_element(self, data, ax, how, label):
        beauty_dict = self.get_beauty_dict(label)
        if hasattr(self, how):
            func_ = getattr(self, how)
            new_elements = func_(ax=ax, data=data, label=label)
        elif not hasattr(ax, how):
            raise "Incorrect plotting way."
        else:
            func_ = getattr(ax, how)
            element = func_(*data, **beauty_dict)
            new_elements = (label, how)
            self.elements[label] = element
        return new_elements

    def linear_fit(self, ax, data, label):
        beauty_points = self.get_beauty_dict(f"{label}_points")
        beauty_line = self.get_beauty_dict(f"{label}_line")
        scatter = ax.scatter(*data, **beauty_points)
        y_sim = get_optimal_fit_linear(*data)
        line = ax.plot(data[0], y_sim, **beauty_line, label=label)
        self.elements[label] = (scatter, line)
        return label, "linear_fit", line


def basic_plot(
    how,
    sc,
    ax,
    treated_label="Treated Unit",
    synth_label="Synthetic Treated Unit",
    treatment_label="Treatment",
    in_space_exclusion_multiple=5,
):
    """
    Supported plots:
      original:
        Outcome of the treated unit and the synthetic control unit for all time periods

      pointwise:
        Difference between the outcome of the treated and synthetic control unit
        I.e. same as original but normalized wrt. the treated unit

      cumulative:
        Sum of pointwise differences in the post treatment period
        I.e. the cumulative treatment effect

      in-space placebo:
        Pointwise plot of synthetic control and in-space placebos

        Procedure:
        Fits a synthetic control to every control unit.
        These synthetic controls are referred to "in-space placebos"

      pre/post rmspe:
        Histogram showing
        (post-treatment rmspe) / (pre-treatment rmspe)
        for real synthetic control and all placebos

        Extreme values indicates small difference in pre-period with
        large difference (estimated treatment effect) in the post-period
        Treated unit should be more extreme than placebos to indicate significance


    Arguments:
      panels : list of strings
        list of the plots to be generated

      figsize : tuple (int, int)
        argument to plt.figure
        First value indicated desired width of plot, second the height
        The height height is divided evenly between each subplot, whereas each subplot has full width
        E.g. three plots: each subplot will have figure size (width, height/3)

      treated_label : str
        Label for treated unit in plot title and legend

      synth_label : str
        Label for synthetic control unit in plot title and legend

      in_space_exclusion_multiple : float
        default: 5
        used only in 'in-space placebo' plot.
        excludes all placebos with PRE-treatment rmspe greater than
        the real synthetic control*in_space_exclusion_multiple

    Returns:
    :param sc:
    :param how:
    :param data:
    :param ax:
    :param in_space_exclusion_multiple:
    :param treated_label:
    :param synth_label:
    :param treatment_label:
    """
    data = sc.original_data
    # Extract Synthetic Control
    synth = data.synth_outcome
    time = data.dataset[data.time].unique()
    valid_panels = [
        "original",
        "pointwise",
        "cumulative",
        "in-space placebo",
        "rmspe ratio",
        "in-time placebo",
    ]
    solo_panels = ["rmspe ratio"]  # plots with different axes
    if not any([how in valid_panels, how in solo_panels]):
        raise ValueError("Wrong plot type.")

    if "original" in how:
        # Determine appropriate limits for y-axis
        max_value = max(
            np.max(data.treated_outcome_all), np.max(data.synth_outcome)
        )
        min_value = min(
            np.min(data.treated_outcome_all), np.min(data.synth_outcome)
        )

        # Make plot
        ax.set_title("{} vs. {}".format(treated_label, synth_label))
        ax.plot(time, synth.T, "r--", label=synth_label)
        ax.plot(time, data.treated_outcome_all, "b-", label=treated_label)
        ax.axvline(data.treatment_period - 1, linestyle=":", color="gray")
        ax.set_ylim(
            -1.2 * abs(min_value), 1.2 * abs(max_value)
        )  # Do abs() in case min is positive, or max is negative
        # ax.annotate(treatment_label,
        #             # Put label below outcome if pre-treatment trajectory is decreasing, else above
        #             xy=(data.treatment_period - 1, data.treated_outcome[-1] * (
        #                         1 + 0.2 * np.sign(data.treated_outcome[-1] - data.treated_outcome[0]))),
        #             xytext=(-160, -4),
        #             xycoords='data',
        #             textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->"))
        ax.set_ylabel(data.outcome_var)
        ax.set_xlabel(data.time)
        ax.legend()

    if "pointwise" in how:
        # Subtract outcome of synth from both synth and treated outcome
        normalized_treated_outcome = data.treated_outcome_all - synth.T
        normalized_synth = np.zeros(data.periods_all)
        most_extreme_value = np.max(np.absolute(normalized_treated_outcome))

        ax.set_title("Pointwise Effects")
        ax.plot(time, normalized_synth, "r--", label=synth_label)
        ax.plot(time, normalized_treated_outcome, "b-", label=treated_label)
        ax.axvline(data.treatment_period - 1, linestyle=":", color="gray")
        ax.set_ylim(-1.2 * most_extreme_value, 1.2 * most_extreme_value)
        # ax.annotate(treatment_label,
        #             xy=(data.treatment_period - 1, 0.5 * most_extreme_value),
        #             xycoords='data',
        #             xytext=(-160, -4),
        #             textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->"))
        ax.set_ylabel(data.outcome_var)
        ax.set_xlabel(data.time)
        ax.legend()

    if "cumulative" in how:
        normalized_treated_outcome = data.treated_outcome_all - synth.T
        # Compute cumulative treatment effect as cumulative sum of pointwise effects
        cumulative_effect = np.cumsum(
            normalized_treated_outcome[data.periods_pre_treatment :]
        )
        cummulative_treated_outcome = np.concatenate(
            (np.zeros(data.periods_pre_treatment), cumulative_effect),
            axis=None,
        )
        normalized_synth = np.zeros(data.periods_all)

        ax.set_title("Cumulative Effects")
        ax.plot(time, normalized_synth, "r--", label=synth_label)
        ax.plot(time, cummulative_treated_outcome, "b-", label=treated_label)
        ax.axvline(data.treatment_period - 1, linestyle=":", color="gray")
        # ax.set_ylim(-1.1*most_extreme_value, 1.1*most_extreme_value)
        # ax.annotate(treatment_label,
        #             xy=(data.treatment_period - 1, cummulative_treated_outcome[-1] * 0.3),
        #             xycoords='data',
        #             xytext=(-160, -4),
        #             textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->"))
        ax.set_ylabel(data.outcome_var)
        ax.set_xlabel(data.time)
        ax.legend()

    if "in-space placebo" in how:
        # assert self.in_space_placebos != None, "Must run in_space_placebo() before you can plot!"

        zero_line = np.zeros(data.periods_all)
        normalized_treated_outcome = data.treated_outcome_all - synth.T

        ax.set_title("In-space placebo's")
        ax.plot(time, zero_line, "k--")

        # Plot each placebo
        ax.plot(time, data.in_space_placebos[0], "0.7", label="Placebos")
        for i in range(1, data.n_controls):

            # If the pre rmspe is not more than
            # in_space_exclusion_multiple times larger than synth pre rmspe
            if in_space_exclusion_multiple is not None:
                if (
                    data.rmspe_df["pre_rmspe"].iloc[i]
                    < in_space_exclusion_multiple
                    * data.rmspe_df["pre_rmspe"].iloc[0]
                ):
                    ax.plot(time, data.in_space_placebos[i], "0.7")
            else:
                ax.plot(time, data.in_space_placebos[i], "0.7")

        ax.axvline(data.treatment_period - 1, linestyle=":", color="gray")
        ax.plot(time, normalized_treated_outcome, "b-", label=treated_label)

        ax.set_ylabel(data.outcome_var)
        ax.set_xlabel(data.time)
        ax.legend()

    if "rmspe ratio" in how:
        assert (
            data.rmspe_df.shape[0] != 1
        ), "Must run in_space_placebo() before you can plot 'rmspe ratio'!"

        # Sort by post/pre rmspe ratio, high
        sorted_rmspe_df = data.rmspe_df.sort_values(
            by=["post/pre"], axis=0, ascending=True
        )
        ax.set_title("Postperiod RMSPE / Preperiod RMSPE")

        # Create horizontal barplot, one bar per unit
        y_pos = np.arange(data.n_controls + 1)  # Number of units
        ax.barh(
            y_pos, sorted_rmspe_df["post/pre"], color="#3F5D7D", ec="black"
        )

        # Label bars with unit names
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_rmspe_df["unit"])

        # Label x-axis
        ax.set_xlabel("Postperiod RMSPE / Preperiod RMSPE")

    if "in-time placebo" in how:
        ax.set_title(
            "In-time placebo: {} vs. {}".format(treated_label, synth_label)
        )

        ax.plot(time, data.in_time_placebo_outcome.T, "r--", label=synth_label)
        ax.plot(time, data.treated_outcome_all, "b-", label=treated_label)

        ax.axvline(data.placebo_treatment_period, linestyle=":", color="gray")
        # ax.annotate('Placebo Treatment',
        #             xy=(data.placebo_treatment_period,
        #                 data.treated_outcome_all[data.placebo_periods_pre_treatment] * 1.2),
        #             xytext=(-160, -4),
        #             xycoords='data',
        #             textcoords='offset points',
        #
        #             arrowprops=dict(arrowstyle="->"))
        ax.set_ylabel(data.outcome_var)
        ax.set_xlabel(data.time)
        ax.legend()
