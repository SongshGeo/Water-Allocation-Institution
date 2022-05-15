#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# Created date: 2022-02-10
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import pandas as pd
from attrs import define, field
from SyntheticControlMethods import DiffSynth, Synth
from tqdm import tqdm

from .unit_base import ItemBase, UnitBase


class MethodItem(ItemBase):
    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)
        self._parameters = parameters

    pass

    @property
    def parameters(self):
        return self._parameters

    def str_parameters(self):
        str_parameters = []
        for i, k in enumerate(self.parameters):
            str_parameters.append(f"({i+1}) {k}: {self.parameters[k]}.")
        return "".join(str_parameters)


class Method(UnitBase):
    def __init__(self, name="model"):
        super().__init__(unit_name=name)

    def add_function_item(
        self, function, name, parameters=None, description=""
    ):
        if not parameters:
            parameters = {}
        item = MethodItem(
            name=name,
            obj=function,
            parameters=parameters,
            description=description,
        )
        item.update_metadata("parameters", parameters)
        self.add_item(item)
        pass

    def add_item_from_yaml(self, yaml_file):
        pass

    def check_dataset(self, dataset):
        self.dataset = dataset
        pass

    def report(self, show=True, show_notes=False, max_width=30):
        table = super().report(show=False, show_notes=show_notes)
        funcs, parameters = [], []
        for item in self.items:
            item = self.get_item(item)
            funcs.append(item.obj.__name__)
            parameters.append(item.str_parameters())
        table.add_column("Func", funcs)
        table.add_column("Params", parameters)
        table.max_width = max_width
        if show:
            print(table)
        else:
            return table


@define
class ProvinceSynth(Synth):
    """Data input"""

    province: str = field(default=None, repr=True)

    def __init__(
        self,
        dataset,
        outcome_var,
        id_var,
        time_var,
        treatment_period,
        treated_unit,
        n_optim=10,
        pen=0,
        exclude_columns=None,
        random_seed=0,
        **kwargs,
    ):
        if exclude_columns is None:
            exclude_columns = []
        super().__init__(
            dataset,
            outcome_var,
            id_var,
            time_var,
            treatment_period,
            treated_unit,
            n_optim,
            pen,
            exclude_columns,
            random_seed,
            verbose=False,
            **kwargs,
        )
        self.province = treated_unit

    def diff_plot(self, ax, bp):
        data = self.original_data
        time = data.dataset[data.time].unique()
        treat = data.treatment_period
        synthetic = data.synth_outcome.T  # Synthetic
        original = data.treated_outcome_all  # Original
        temp_data = pd.DataFrame(
            data={
                "time": time,
                "origin": original,
                "synth": synthetic,
            }
        )
        before = temp_data[temp_data["time"] <= treat]
        after = temp_data[temp_data["time"] > treat]
        bp.add_element(
            data=(before["time"], before["origin"]),
            ax=ax,
            how="plot",
            label="before_observation",
        )
        bp.add_element(
            data=(before["time"], before["synth"]),
            ax=ax,
            how="plot",
            label="before_prediction",
        )
        bp.add_element(
            data=(after["time"], after["origin"]),
            ax=ax,
            how="linear_fit",
            label="after_observation",
        )
        bp.add_element(
            data=(after["time"], after["synth"]),
            ax=ax,
            how="linear_fit",
            label="after_prediction",
        )
        bp.add_element(
            data=[treat], ax=ax, how="axvline", label=f"IS: {treat}"
        )
        return bp

    pass


def do_synth_once(province, data, outside_data, used_variables, parameters):
    used_province = outside_data.append(data[data["Province"] == province])
    used_period = used_province[
        (used_province["Year"] >= parameters["start"])
        & (used_province["Year"] <= parameters["end"])
    ]
    used_data = used_period[used_variables]

    if parameters.get("synth") == "DSC":
        # DSC = Diff Synth Control
        synth = DiffSynth
    else:
        synth = Synth
    sc = synth(
        used_data,
        parameters["Y_inputs"],
        "Province",
        "Year",
        parameters["treat_year"],
        province,
        n_optim=parameters["n_optim"],
        pen=parameters["pen"],
        random_seed=parameters["random_seed"],
    )
    return sc


def do_synth_model(datasets, parameters):
    """
    Do a synth control modelling.
    :param datasets: {name: path} dictionary of necessary input datasets.
    :param parameters: necessary parameters import from html.
    :return: results of experiment, dict of results.
    """
    # 加载数据，排除所有的省份
    data = pd.read_csv(datasets["merged_data"], index_col=0)
    if "province_exclude" in parameters:
        for province in parameters["province_exclude"]:
            data = data[data["Province"] != province]
    outside_data = data.copy()
    for province in parameters["province_include"]:
        outside_data = outside_data[outside_data["Province"] != province]

    # 获取变量
    used_variables = parameters["X_inputs"]

    # 计算结果
    sc_results = {}
    for province in tqdm(parameters["province_include"]):
        print(f"solving {province}!...")
        sc = do_synth_once(
            province=province,
            data=data,
            outside_data=outside_data,
            used_variables=used_variables,
            parameters=parameters,
        )
        sc_results[province] = sc
    return sc_results


def do_synth_analysis(exp, parameters, analysis):
    if analysis["placebo"]:
        exp.do_in_place_placebo()
        exp.do_in_time_placebo()
    if analysis["iter_placebo"]:
        span = parameters.get("placebo_iter_span")
        exp.iter_time_placebo(span=span)
    return exp.result


if __name__ == "__main__":
    pass
