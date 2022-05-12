#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9


import numpy as np
import pandas as pd
import pca
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

from func.tools import ROOT


def calculate_vif(df):
    vif = pd.DataFrame()
    vif["index"] = df.columns
    vif["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif


def filter_features_by_vif(df):
    vif = calculate_vif(df)
    while (vif["VIF"] > 10).any():
        remove = vif.sort_values(by="VIF", ascending=False)["index"][
            :1
        ].values[0]
        df.drop(remove, axis=1, inplace=True)
        vif = calculate_vif(df)
