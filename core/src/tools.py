#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/4
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import os
from collections import OrderedDict

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from affine import Affine
from matplotlib import pyplot as plt
from qcloudsms_py import SmsSingleSender
from qcloudsms_py.httpclient import HTTPError
from rasterio import features


def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict to yaml"""
    with open(save_path, "w") as file:
        file.write(yaml.dump(dict_value, allow_unicode=True, sort_keys=False))


def send_finish_message(num):
    """
    Send message to myself as notification.
    :param num: experiment num
    :return: dict, sending result from the network.
    """
    # 短信应用SDK AppID
    appid = 1400630042  # SDK AppID是1400开头
    # 短信应用SDK AppKey
    app_key = "ad30ec46aa617263813ca8996e1a0113"
    # 需要发送短信的手机号码
    phone_numbers = ["18500685922"]
    # 短信模板ID，需要在短信应用中申请
    template_id = 1299444
    # 签名
    sms_sign = "隅地公众号"

    s_sender = SmsSingleSender(appid, app_key)
    params = [num]  # 当模板没有参数时，`params = []`
    try:
        result = s_sender.send_with_param(
            86,
            phone_numbers[0],
            template_id,
            params,
            sign=sms_sign,
            extend="",
            ext="",
        )  # 签名参数不允许为空串
        print(result)
        return result
    except HTTPError as e:
        print(e)
    except Exception as e:
        print(e)


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
    return y_sim, k


def extract_mean_std(data, start, end):
    data = data.loc[start:end, :]
    return data.mean(), data.std()


def transform_from_latlon(lat, lon):
    """input 1D array of lat / lon and output an Affine transformation"""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(
    shapes,
    coords,
    latitude="latitude",
    longitude="longitude",
    fill=np.nan,
    **kwargs,
):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.

    usage:
    -----
    1. read shapefile to geopandas.GeoDataFrame
          `states = gpd.read_file(shp_dir+shp_file)`
    2. encode the different shapefiles that capture those lat-lons as different
        numbers i.e. 0.0, 1.0 ... and otherwise np.nan
          `shapes = (zip(states.geometry, range(len(states))))`
    3. Assign this to a new coord in your original xarray.DataArray
          `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

    arguments:
    ---------
    : **kwargs (dict): passed to `rasterio.rasterize` function

    attrs:
    -----
    :transform (affine.Affine): how to translate from latlon to ...?
    :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
      outside the .shp file with np.nan
    :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
      with "X", "Y" as keys, and xr.DataArray as values

    returns:
    -------
    :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
      and coords `Y` = latitude, 'X' = longitude.


    """
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(
        shapes,
        out_shape=out_shape,
        fill=fill,
        transform=transform,
        dtype=float,
        **kwargs,
    )
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(
        raster, coords=spatial_coords, dims=(latitude, longitude)
    )


def add_shape_coord_from_data_array(xr_da, shp_path, coord_name):
    """Create a new coord for the xr_da indicating whether or not it
     is inside the shapefile

    Creates a new coord - "coord_name" which will have integer values
     used to subset xr_da for plotting / analysis/

    Usage:
    -----
    precip_da = add_shape_coord_from_data_array(precip_da, "awash.shp", "awash")
    awash_da = precip_da.where(precip_da.awash==0, other=np.nan)
    """
    # 1. read in shapefile
    shp_gpd = gpd.read_file(shp_path)

    # 2. create a list of tuples (shapely.geometry, id)
    #    this allows for many different polygons within a .shp file (e.g. States of US)
    shapes = [(shape, n) for n, shape in enumerate(shp_gpd.geometry)]

    # 3. create a new coord in the xr_da which will be set to the id in `shapes`
    xr_da[coord_name] = rasterize(
        shapes, xr_da.coords, longitude="longitude", latitude="latitude"
    )

    return xr_da


def within_province_mask(provinces, ds, var, **kwargs):
    # TODO finish the doc
    """
    Transform NetCDF datasets (ds) masked by provinces.shp into Chinese map
    :param provinces:
    :param ds:
    :param kwargs:
    :return:
    """
    provinces_ids = {k: i for i, k in enumerate(provinces.NAME)}
    shapes = [(shape, n) for n, shape in enumerate(provinces.geometry)]
    ds["states"] = rasterize(shapes, ds.coords, **kwargs)
    result = pd.DataFrame()
    for province in provinces_ids:
        data = (
            ds[var]
            .where(ds.states == provinces_ids[province])
            .mean(dim=["lat", "lon"])
        )
        index = pd.DatetimeIndex(ds.time).strftime("%Y")
        result[province] = pd.Series(data, index=index)
    return result


def show_files(path, all_files=None, full_path=False, suffix=None):
    """
    All files under the folder.
    :param path: A folder.
    :param all_files: initial list, where files will be saved.
    :param full_path: Save full path or just file name? Default False, i.e., just file name.
    :param suffix: Filter by suffix.
    :return: all_files, a updated list where new files under the path-folder saved, besides of the original input.
    """
    # 首先遍历当前目录所有文件及文件夹
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} is not a folder.")
    if all_files is None:
        all_files = []
    if not suffix:
        suffix = []

    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        if isinstance(suffix, str):
            judge = not cur_path.endswith(suffix)
        else:
            judge = all([not cur_path.endswith(suf) for suf in suffix])
        if judge:
            continue
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files, full_path)
        else:
            if full_path:
                all_files.append(cur_path)
            else:
                all_files.append(file)
    return all_files


def plot_gam_and_interval(
    x,
    y,
    main_color="#29303C",
    err_color="gray",
    width=0.95,
    ax=None,
    err_space=True,
    alpha=0.1,
    scatter_alpha=0.8,
    y_label="Y",
):
    if not ax:
        fig, ax = plt.subplots()

    import pygam

    X = x.reshape(-1, 1)

    gam = pygam.LinearGAM(n_splines=25).gridsearch(X, y)
    XX = gam.generate_X_grid(term=0, n=500)

    ax.plot(
        XX,
        gam.predict(XX),
        color=main_color,
        label="{} GAM Fitted".format(y_label),
    )  # 拟合曲线
    ax.scatter(X, y, facecolor=main_color, alpha=scatter_alpha, label="")  # 散点

    if err_space:
        err = gam.prediction_intervals(XX, width=width)
        ax.plot(
            XX,
            gam.prediction_intervals(XX, width=width),
            color=err_color,
            ls="--",
        )  # 置信区间
        ax.fill_between(
            XX.reshape(
                -1,
            ),
            err[:, 0],
            err[:, 1],
            color=err_color,
            alpha=alpha,
            label="{} Confidence interval".format(y_label),
        )
