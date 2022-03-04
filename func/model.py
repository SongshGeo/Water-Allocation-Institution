#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# Created date: 2022-02-10
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import os
import pickle

import numpy as np
import pandas as pd
import yaml
from SyntheticControlMethods import Synth
from tqdm import tqdm


def do_synth_model(p):
    """
    Do a synth control modelling.
    :param p: parameters import from html
    :return: results of experiment, dict of results
    """
    # 加载数据，排除所有的省份
    data = pd.read_csv(p["data_input"], index_col=0)
    if "province_exclude" in p:
        for province in p["province_exclude"]:
            data = data[data["Province"] != province]
    outside_data = data.copy()
    for province in p["province_include"]:
        outside_data = outside_data[outside_data["Province"] != province]

    # 获取变量
    used_variables = p["X_inputs"]

    # 计算结果
    sc_results = {}
    for province in tqdm(p["province_include"]):
        print(f"solving {province}!...")
        used_province = outside_data.append(data[data["Province"] == province])
        used_period = used_province[
            (used_province["Year"] >= p["start"])
            & (used_province["Year"] <= p["end"])
        ]
        used_data = used_period[used_variables]

        sc = Synth(
            used_data,
            p["Y_inputs"],
            "Province",
            "Year",
            p["treat_year"],
            province,
            n_optim=p["n_optim"],
            pen=p["pen"],
            random_seed=p["random_seed"],
        )

        sc_results[province] = sc
    return sc_results


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
        exp = pickle.load(pkl)
        pkl.close()

    # save datasets
    dataset = pd.DataFrame(
        index=np.arange(params["start"], params["end"] + 1)
    )  # time
    for province, synth_result in exp.result.items():
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


if __name__ == "__main__":
    pass
