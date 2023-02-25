#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg

from .tools import get_optimal_fit_linear

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
        raise ValueError(f"Wrong plot type, valid input: {valid_panels}.")

    if "original" in how:
        # Determine appropriate limits for y-axis
        max_value = max(np.max(data.treated_outcome_all), np.max(data.synth_outcome))
        min_value = min(np.min(data.treated_outcome_all), np.min(data.synth_outcome))

        # Make plot
        ax.set_title(f"{treated_label} vs. {synth_label}")
        ax.plot(time, synth.T, "r--", label=synth_label)
        ax.plot(time, data.treated_outcome_all, "b-", label=treated_label)
        ax.axvline(data.treatment_period - 1, linestyle=":", color="gray")
        ax.set_ylim(
            -1.2 * abs(min_value), 1.2 * abs(max_value)
        )  # Do abs() in case min is positive, or max is negative
        _better_labels(ax, data)
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
        _better_labels(ax, data)
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
        _better_labels(ax, data)
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
            if in_space_exclusion_multiple is None:
                ax.plot(time, data.in_space_placebos[i], "0.7")

            elif (
                data.rmspe_df["pre_rmspe"].iloc[i]
                < in_space_exclusion_multiple * data.rmspe_df["pre_rmspe"].iloc[0]
            ):
                ax.plot(time, data.in_space_placebos[i], "0.7")
        ax.axvline(data.treatment_period - 1, linestyle=":", color="gray")
        ax.plot(time, normalized_treated_outcome, "b-", label=treated_label)

        _better_labels(ax, data)
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
        colors = []
        for p in sorted_rmspe_df["unit"]:
            if p == data.treated_unit:
                colors.append("black")
            else:
                colors.append("gray")
        ax.barh(y_pos, sorted_rmspe_df["post/pre"], color=colors, ec="black")

        # Label bars with unit names
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_rmspe_df["unit"])

        # Label x-axis
        ax.set_xlabel("Postperiod RMSPE / Preperiod RMSPE")

    if "in-time placebo" in how:
        ax.set_title(f"In-time placebo: {treated_label} vs. {synth_label}")

        ax.plot(time, data.in_time_placebo_outcome.T, "r--", label=synth_label)
        ax.plot(time, data.treated_outcome_all, "b-", label=treated_label)

        ax.axvline(data.placebo_treatment_period, linestyle=":", color="gray")
        _better_labels(ax, data)


def _better_labels(ax, data):
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
