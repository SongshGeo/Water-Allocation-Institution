#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import pandas as pd
import pca
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def fit_pca(pca_data, params):
    features = params['features']
    n_components = params['n_components']
    normalize = params['normalize']
    model = pca(n_components=n_components, normalize=normalize)
    data = pca_data[features]
    results = model.fit_transform(data.values, col_labels=data.columns, row_labels=data.index)
    return model, results


def transform_features(model, params):
    pass


def calculate_vif(df):
    vif = pd.DataFrame()
    vif["index"] = df.columns
    vif["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif


def filter_features_by_vif(df, params):
    threshold = params['threshold']
    vif = calculate_vif(df)
    while (vif["VIF"] > threshold).any():
        remove = vif.sort_values(by="VIF", ascending=False)["index"][
            :1
        ].values[0]
        df.drop(remove, axis=1, inplace=True)
        vif = calculate_vif(df)
