#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/4
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9


# 使用图片的比例来定位
def get_position_by_ratio(ax, x_ratio, y_ratio):
    """
    使用图片的比例来返回定位，从而更好的控制说明文字的位置
    ax: 一个 matplotlib 的画图轴对象
    x_ratio: 横坐标的比例位置
    y_ratio: 纵坐标的比例位置
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x = (x_max - x_min) * x_ratio + x_min
    y = (y_max - y_min) * y_ratio + y_min
    return x, y


def get_optimal_fit_linear(x_arr, y_arr):
    from scipy import optimize

    def linear(x, slope, intercept):
        return slope * x + intercept

    k, b = optimize.curve_fit(linear, x_arr, y_arr)[0]  # optimize
    y_sim = linear(x_arr, k, b)  # simulated y
    return y_sim
