#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/4
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import inspect

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
