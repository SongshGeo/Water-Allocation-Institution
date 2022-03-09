#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/4
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

from qcloudsms_py import SmsSingleSender
from qcloudsms_py.httpclient import HTTPError


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
    return y_sim


def extract_mean_std(data, start, end):
    data = data.loc[start:end, :]
    return data.mean(), data.std()
