#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from bidict import bidict

from .filter_features import fit_pca, transform_features
from .plots import NATURE_PALETTE, plot_pre_post

__all__ = ["fit_pca", "transform_features", "NATURE_PALETTE", "plot_pre_post"]


VARS_EN2CH = {
    # "Province": '省份',
    # "Year": '年份',
    "Irrigated area: Rice": "水稻",
    "Irrigated area: Wheat": "小麦",
    "Irrigated area: Maize": "玉米",
    "Irrigated area: Vegetables and fruits": "果蔬",
    "Irrigated area: Others": "其它农业",
    "Industrial gross value added (GVA): Textile": "纺织",
    "Industrial gross value added (GVA): Papermaking": "造纸",
    "Industrial gross value added (GVA): Petrochemicals": "石化",
    "Industrial gross value added (GVA): Metallurgy": "冶金",
    "Industrial gross value added (GVA): Mining": "采矿",
    "Industrial gross value added (GVA): Food": "食品",
    "Industrial gross value added (GVA): Cements": "水泥",
    "Industrial gross value added (GVA): Machinery": "机械",
    "Industrial gross value added (GVA): Electronics": "电子",
    "Industrial gross value added (GVA): Thermal electrivity": "火电",
    "Industrial gross value added (GVA): Others": "其它工业",
    "Urban population": "城市人口",
    "Service GVA": "服务业",
    "Rural population": "农村人口",
    "Livestock population": "牲畜数量",
    # "Total water use": "总用水量",
    "WCI": "农业节水灌溉",
    "Ratio of industrial water recycling": "工业再利用率",
    "prec": "降水",
    "temp": "气温",
    # "wind": '风速',
}


PROVINCES_CHN2ENG = bidict(
    {
        "青海": "Qinghai",
        "四川": "Sichuan",
        "甘肃": "Gansu",
        "宁夏": "Ningxia",
        "内蒙古": "Neimeng",
        "陕西": "Shaanxi",
        "山西": "Shanxi",
        "河南": "Henan",
        "山东": "Shandong",
        "河北": "Hebei",
        "天津": "Tianjin",
        "北京": "Beijing",
        # "河北和天津": "Jinji",
        "津冀": "Jinji",
        "辽宁": "Liaoning",
        "吉林": "Jilin",
        "黑龙江": "Heilongjiang",
        "上海": "Shanghai",
        "江苏": "Jiangsu",
        "云南": "Yunan",
        "安徽": "Anhui",
        "广东": "Guangdong",
        "广西": "Guangxi",
        "新疆": "Xinjiang",
        "江西": "Jiangxi",
        "浙江": "Zhejiang",
        "海南": "Hainan",
        "湖北": "Hubei",
        "湖南": "Hunan",
        "福建": "Fujian",
        "西藏": "Tibet",
        "贵州": "Guizhou",
        "重庆": "Chongqing",
        "中国": "China",
        # "内蒙": "Neimeng",
        # "黑龙": "Heilongjiang",
    }
)
