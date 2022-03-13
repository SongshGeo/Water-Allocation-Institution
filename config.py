#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/2
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import logging
import os
import sys

from func.experiment import Experiment
from func.handle import ExpResultsHandler
from func.model import do_synth_analysis, do_synth_model

PROJECT_NAME = "WAInstitution_YRB_2021"
ROOT = "/Users/songshgeo/Documents/Pycharm/WAInstitution_YRB_2021"

SCHEME_87 = {
    "Qinghai": 1.41,
    # 'Sichuan': 0.04,
    "Gansu": 3.04,
    "Ningxia": 4.00,
    "Neimeng": 5.86,
    "Shanxi": 3.80,
    "Shaanxi": 4.31,
    "Henan": 5.54,
    "Shandong": 7,
    # 'Jinji': 2
}

PROVINCES_CHN2ENG = {
    "青海": "Qinghai",
    "四川": "Sichuan",
    "甘肃": "Gansu",
    "宁夏": "Ningxia",
    "内蒙古": "Neimeng",
    "陕西": "Shanxi",
    "山西": "Shaanxi",
    "河南": "Henan",
    "山东": "Shandong",
    "河北": "Hebei",
    "天津": "Tianjin",
    "北京": "Beijing",
    "河北和天津": "Jinji",
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
    "内蒙": "Neimeng",
    "黑龙": "Heilongjiang",
}

# BASIC logging settings.
LOG_FORMAT = "%(asctime)s %(levelname)s %(filename)s %(message)s "
DATE_FORMAT = "%Y-%m-%d  %H:%M:%S %a "
FILE_LEVEL = "debug"
CMD_LEVEL = "warn"


def set_logger(
    name,
    path=None,
    reset=False,
    log_format=None,
    file_level="debug",
    cmd_level="info",
):
    """
    Setup a log file for my project.
    :param name: logger's name
    :param path: sub-folder path of the log file
    :param reset: remove the old log file and refresh new one?
    :param log_format: format of file message.
    :param file_level: message save level in file.
    :param cmd_level: message level print in CMD
    :return: a logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_level = getattr(logging, file_level.upper())
    cmd_level = getattr(logging, cmd_level.upper())

    if path:
        file_path = os.path.join(ROOT, path, f"{name}.log")
    else:
        file_path = os.path.join(ROOT, f"{name}.log")
    if not log_format:
        log_format = LOG_FORMAT
    if reset:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Logging file not found in {file_path}, no need of reset."
            )
        else:
            os.remove(file_path)

    # 建立一个 FileHandler 来把日志记录在文件里
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    # 建立一个 StreamHandler 来把日志打在CMD窗口上
    cmd_handler = logging.StreamHandler()
    cmd_handler.setLevel(cmd_level)
    cmd_handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(file_handler)
    logger.addHandler(cmd_handler)
    return logger


log = set_logger(PROJECT_NAME, file_level=FILE_LEVEL, cmd_level=CMD_LEVEL)

if __name__ == "__main__":
    # Do experiment model and analysis from CMD.
    YAML_PATH = sys.argv[1]
    exp = ExpResultsHandler(yaml_file=YAML_PATH)

    # Change to a special logger.
    exp_rel_path = exp.get_path("experiment", absolute=False)
    exp_log = set_logger(exp.name, path=exp_rel_path)
    exp.change_experiment_logger(exp_log)
    # exp.load_from_pickle()

    # modelling
    sc_result = exp.do_experiment(model=do_synth_model, notification=True)
    exp.drop_exp_to_pickle()

    # analysis
    sc_analysis = exp.do_analysis(model=do_synth_analysis, notification=True)
    exp.drop_exp_to_pickle()
    pass
