#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# Created date: 2022-02-10
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import logging
import os
import pickle

import pandas as pd
import yaml
from attrs import define, field
from qcloudsms_py import SmsSingleSender
from qcloudsms_py.httpclient import HTTPError

# BASIC testing settings.
LOG_FORMAT = logging.Formatter(
    "%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s "
)
DATE_FORMAT = "%Y-%m-%d  %H:%M:%S %a "


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


@define
class Experiment:
    model = field(repr=False)
    yaml_path: str = field(repr=False)
    name: str = field(default=None, repr=True)
    other_path: str = field(default=None, repr=False)
    result: dict = field(default=None, repr=False)
    state: str = field(default="Initialized", repr=True)

    def __init__(
        self,
        model,
        yaml_path,
        other_path=False,
    ) -> None:
        """
        Initiate an experiment.
        :param model: how to calculate the result.
        :param yaml_path: parameters input.
        :param other_path: save to another output path? default False.
        """
        # 实验存储的路径
        if not other_path:
            experiment_path = os.path.dirname(yaml_path)  # 父级目录
        else:
            experiment_path = other_path

        # 从YAML文件中读取参数
        with open(yaml_path, "r", encoding="utf-8") as file:
            p = yaml.load(file.read(), Loader=yaml.FullLoader)
            file.close()

        # 实验类变量
        self.p = p
        self.path = experiment_path
        self.log = None
        if "name" in p:
            self.name = p["name"]
        else:
            self.name = None
        self.model = model
        self.state = "Initialized"
        self.result = None

    def set_log_file(
        self, file_level=logging.DEBUG, cmd_level=logging.WARNING
    ):
        """
        set up log file
        :param file_level: file logging level, default DEBUG (all).
        :param cmd_level: cmd logging level, default WARNING.
        :return:
        """
        # 配置日志
        log_path = os.path.join(self.path, f"{self.p['name']}.log")
        if os.path.exists(log_path):
            os.remove(log_path)
        logger = logging.getLogger(f"{self.name}")
        logger.setLevel(file_level)
        # 建立一个 FileHandler 来把日志记录在文件里，级别为debug以上
        fh = logging.FileHandler(log_path)
        fh.setLevel(file_level)
        # 建立一个 StreamHandler 来把日志打在CMD窗口上，级别为error以上
        ch = logging.StreamHandler()
        ch.setLevel(cmd_level)
        ch.setFormatter(LOG_FORMAT)
        fh.setFormatter(LOG_FORMAT)
        # 将相应的handler添加在logger对象中
        logger.addHandler(ch)
        logger.addHandler(fh)
        self.log = logger
        logger.info(f"Log file set, level {file_level}, cmd level {cmd_level}")
        return logger

    def do_experiment(self, notification=False):
        """
        Main func, do experiment.
        :param notification:
        :return:
        """
        logger = self.log
        logger.info("Start experiment.")
        result = self.model(self.p)
        self.state = "EXP finished!"
        logger.info("End experiment.")
        # Send a message to my phone for notification.
        if notification:
            sent = send_finish_message(self.name)
            logger.info(f"Message {sent} sent.")
        self.result = result
        return result

    def drop_exp_to_pickle(self):
        """
        Drop the experiment to pickle data for further uses.
        """
        file = f"{self.name}.pkl"
        with open(os.path.join(self.path, file), "wb") as pkl:
            pickle.dump(self, pkl)

    def original_plots(self, province, save=True):
        # TODO put this functions into handle class.
        if save:
            # 储存图像路径
            figs_folder = os.path.join(self.path, "figs")
            if not os.path.exists(figs_folder):
                os.mkdir(figs_folder)

            # 图像展示
            self.result[province].plot(
                ["original", "pointwise", "cumulative"],
                treated_label=province,
                synth_label=f"Synthetic {province}",
                treatment_label=f"Treatment in {self.p['treat_year']}",
                save_path=os.path.join(figs_folder, f"{province}.jpg"),
            )
        else:
            # 图像展示
            self.result[province].plot(
                ["original", "pointwise", "cumulative"],
                treated_label=province,
                synth_label=f"Synthetic {province}",
                treatment_label=f"Treatment in {self.p['treat_year']}",
            )

    def weight_df(self, province="all"):
        if province == "all":
            weights = {}
            for p in self.result.keys():
                weights[p] = self.result[p].original_data.weight_df
        else:
            weights = self.result[province].original_data.weight_df
        return weights

    def original_comparison(self, province="all"):
        if province == "all":
            comparison = pd.DataFrame()
            for p in self.result.keys():
                comparison_df = self.result[p].original_data.comparison_df
                # 把获取的结果储存到同一个表中
                for col in comparison_df:
                    # WMAPE is a indicator to stimulation of Synth Control.
                    if col == "WMAPE":
                        comparison[col + "_" + p] = comparison_df[col]
                    else:
                        comparison[col] = comparison_df[col]
        else:
            comparison = self.result[province].original_data.comparison_df
        return comparison

    def do_in_time_placebo(self, placebo_time, province, n_optim=100):
        self.result[province].in_time_placebo(placebo_time, n_optim=n_optim)
        self.result[province].plot(
            ["in-time placebo"],
            treated_label=province,
            synth_label=f"Synthetic {province}",
        )

    def drop_compared_datasets(self):
        # for province in self.result.keys():
        # self.result
        pass
