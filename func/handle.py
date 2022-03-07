#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/4
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import os
import pickle

import numpy as np
import pandas as pd
import yaml
from attrs import define


@define
class ExpResultsHandler(object):
    """
    Experiment handler, for exploring results easier.
    """

    def correlation_stimulation(self):
        pass

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
        # for province in self.result.keys():
        # self.result
        pass

    pass


def transfer_exp_pickle_to_data(yaml_path):
    """
    Transform a pickle experiment result to comparable csv data.
    :param yaml_path: YAML parameters data as a file path.
    :return: transformed dataset.
    """
    # open parameters
    with open(yaml_path, "r", encoding="utf-8") as file:
        params = yaml.load(file.read(), Loader=yaml.FullLoader)
        file.close()

    # read pickle file experiment results
    with open(
        os.path.join(os.path.dirname(yaml_path), params["name"] + ".pkl"), "rb"
    ) as pkl:
        experiment = pickle.load(pkl)
        pkl.close()

    # save datasets
    dataset = pd.DataFrame(
        index=np.arange(params["start"], params["end"] + 1)
    )  # time
    for province, synth_result in experiment.result.items():
        synth_data = synth_result.original_data.synth_outcome.T  # Synth label
        actual_data = (
            synth_result.original_data.treated_outcome_all
        )  # original data
        dataset[f"{province}_synth"] = synth_data
        dataset[f"{province}_actual"] = actual_data

    dataset.to_csv(
        os.path.join(
            os.path.dirname(yaml_path),
            f"{params['name']}_{params['treat_year']}.csv",
        )
    )
    return dataset
