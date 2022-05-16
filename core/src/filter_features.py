#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import pandas as pd
import statsmodels.api as sm
from pca import pca
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor


def fit_pca(pca_data, params):
    features = params["features"]
    n_components = params["n_components"]
    normalize = params["normalize"]
    model = pca(n_components=n_components, normalize=normalize)
    data = pca_data[features]
    results = model.fit_transform(
        data.values, col_labels=data.columns, row_labels=data.index
    )
    return model, results


def transform_features(transform_data, params, fitted_model):
    transform_data = transform_data.dropna(how="any")
    features = params["features"]
    if params["normalize"]:
        scaled_features = preprocessing.scale(transform_data[features])
    else:
        scaled_features = transform_data[features].values
    other_data = transform_data.copy().reset_index()
    other_data = other_data.drop(features, axis=1)
    transformed_data = fitted_model.transform(scaled_features).reset_index()
    result = pd.concat([other_data, transformed_data], axis=1)
    result = result.drop("index", axis=1)
    result.index = transformed_data.index
    return result


def filter_features_by_vif(df, params):
    def calculate_vif(df):
        vif = pd.DataFrame()
        vif["index"] = df.columns
        vif["VIF"] = [
            variance_inflation_factor(df.values, i) for i in range(df.shape[1])
        ]
        return vif

    threshold = params["threshold"]
    if params["normalize"]:
        df = pd.DataFrame(
            preprocessing.scale(df), index=df.index, columns=df.columns
        )
    vif = calculate_vif(df)
    while (vif["VIF"] > threshold).any():
        remove = vif.sort_values(by="VIF", ascending=False)["index"][
            :1
        ].values[0]
        df.drop(remove, axis=1, inplace=True)
        vif = calculate_vif(df)