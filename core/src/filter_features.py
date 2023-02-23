#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import numpy as np
import pandas as pd
from pca import pca
from statsmodels.stats.outliers_influence import variance_inflation_factor


def my_scale(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def check_data_align(data):
    province_counting = data.groupby("Province").count().iloc[:, 0]
    if province_counting.max() != province_counting.min():
        raise ValueError(
            f"NOT aligned dataset: max year: {province_counting.max()} max year:{province_counting.min()}"
        )
    if data["Year"].unique().__len__() != province_counting.min():
        raise ValueError()
    return data


def generate_pca_data(data, params):
    features = params["features"]
    variables = ["Province"]
    variables.extend(features)
    data = data[variables].dropna(how="any")
    return data


def fit_pca(pca_data, features, n_components=0.95, normalize=True):
    model = pca(n_components=n_components, normalize=normalize)
    data = pca_data[features]
    results = model.fit_transform(
        data.values, col_labels=data.columns, row_labels=data.index
    )
    return model, results


def transform_features(transform_data, features, fitted_model, normalize=True):
    transform_data = transform_data.dropna(how="any", subset=features)
    if normalize:
        data = my_scale(transform_data[features])
    else:
        data = transform_data[features].values
    other_data = transform_data.copy().reset_index()
    other_data = other_data.drop(features, axis=1)
    transformed_data = fitted_model.transform(data).reset_index()
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
        df = pd.DataFrame(my_scale(df), index=df.index, columns=df.columns)
    vif = calculate_vif(df)
    while (vif["VIF"] > threshold).any():
        remove = vif.sort_values(by="VIF", ascending=False)["index"][:1].values[0]
        df.drop(remove, axis=1, inplace=True)
        vif = calculate_vif(df)
