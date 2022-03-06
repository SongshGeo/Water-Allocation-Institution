#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# Created date: 2022-02-10
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import datetime
import os
import pickle
import pprint
import sys

import yaml
from attrs import define, field

from config import ROOT
from config import log as logger
from config import set_logger
from func.model import do_synth_model
from func.tools import send_finish_message


@define
class Experiment(object):
    model = field(repr=False)
    name: str = field(default=None, repr=True)
    paths = field(factory=dict, repr=False)
    result: dict = field(default=None, repr=False)
    state: str = field(default="Initialized", repr=True)
    datasets: dict = field(factory=dict, repr=False)
    parameters: dict = field(factory=dict, repr=False)
    analysis: dict = field(factory=dict, repr=False)
    log = field(default=None, repr=False)
    authors = field(factory=list, repr=False)
    description: str = field(default=None, repr=True)
    _updated_time = field(default=None, repr=True)
    _strftime = field(default="%Y-%m-%d, %H:%M:%S", repr=False)
    _run_time = field(default="", repr=False)

    def __init__(
        self,
        model,
        yaml_file,
        name="exp",
        experiment_path=None,
        results_path=None,
    ):
        """
        Initiate an experiment, as the following structure.

        ```
        root
        ├── example
        │   └── config.yaml
        └── experiment
            ├── name.log
            └── results
                ├── name.pkl
                └── name_results.pkl
        ```
        Read experiment config from config.yaml, run model and then export results.

        :param model: how to calculate the result.
        :param yaml_file: experimental parameters as a `.YAML` file input,
            Mandatory parameters:
                1. name: experiments' name, if none, setup when init.
                2. root: root path of the project.
                3. experiment_path: folder path of experiment.
                4. results_path: folder path of experimental results.
                5. datasets: all model needed datasets.
                6. parameters: other necessary parameters of the model.
                7. analysis: auto analysis pipelines.
            Metadata:
                1. Author...
                2. Description...
        """
        # Read parameters from yaml.
        with open(yaml_file, "r", encoding="utf-8") as file:
            parameters = yaml.load(file.read(), Loader=yaml.FullLoader)
            self.parameters = parameters.get("parameters")
            self.analysis = parameters.get("analysis")
            self.datasets = parameters.get("datasets")
            self.description = parameters.get("description")
            self.authors = parameters.get("author")

            # root path: from yaml > default project root
            root = parameters.get("root")
            if not root:
                root = ROOT
            os.chdir(root)
            if not results_path:
                results_path = os.path.join(
                    root, parameters.get("results_path")
                )
            if not experiment_path:
                experiment_path = os.path.join(
                    root, parameters.get("experiment_path")
                )
            file.close()

        # basic attributions
        self.name = name
        self.log = logger
        self.model = model
        self.state = "Initialized"
        self.paths = {}
        self._run_time = "Not done"
        self.result = None
        self._strftime = "%Y-%m-%d, %H:%M:%S"
        self._updated_time = datetime.datetime.now().strftime(self._strftime)

        # setup paths
        self.paths["root"] = root
        self.paths["yaml"] = os.path.abspath(yaml_file)
        self.paths["experiment"] = os.path.abspath(
            os.path.join(root, experiment_path)
        )
        self.paths["results"] = os.path.abspath(
            os.path.join(root, results_path)
        )
        self.paths["log"] = os.path.join(root, f"{logger.name}.log")

        for dataset in self.datasets.keys():
            self.paths[dataset] = os.path.join(
                root, self.datasets.get(dataset)
            )
        for key, path in self.paths.items():
            if not os.path.exists(path):
                os.mkdir(path)
                logger.warning(
                    f"{key} folder made in {os.path.dirname(path)}."
                )

    def get_path(self, key="experiment", absolute=True):
        path = self.paths.get(key)
        root = self.paths.get("root")
        if absolute:
            return path
        else:
            return os.path.relpath(path, root)

    def set_experiment_log(self, **kwargs):
        """
        Set up log file to this experiment and switch the default logger.
        :return: The new exp logger.
        """
        # Default project logger
        exp_path = self.get_path("experiment")
        exp_rel_path = self.get_path("experiment", absolute=False)
        logger.warning(
            f"Experiment {self.name} log file will be set under {exp_rel_path}."
        )
        # Change logger
        exp_log = set_logger(self.name, path=exp_rel_path, **kwargs)
        self.log = exp_log
        exp_log.info(f"Experiment {self.name} log file set.")
        self._updated_time = datetime.datetime.now().strftime(self._strftime)
        self.paths["log"] = os.path.join(exp_path, f"{self.name}.log")
        return exp_log

    def do_experiment(self, notification=False):
        """
        Main func, do experiment.
        :param notification: send message to user.
        :return: state of the experiment.
        """
        log = self.log
        # Do experiment
        log.info(f"Start experiment, model: {self.model.__name__}.")
        start_time = datetime.datetime.now()
        result = self.model(self.datasets, self.parameters)
        self.state = "finished"
        log.info("End experiment.")

        # Send a message to my phone for notification.
        if notification:
            sent = send_finish_message(self.name)
            log.info(f"Notification sending msg: {sent['errmsg']}.")

        # Save results.
        self.result = result
        self.drop_result_to_pickle()
        after_time = datetime.datetime.now()
        self._updated_time = after_time.strftime(self._strftime)
        self._run_time = str(after_time - start_time)
        return self.state

    def drop_result_to_pickle(self):
        """Drop the experimental results to pickle data."""
        result_pickle_path = os.path.join(
            self.get_path("results"), f"{self.name}_results.pkl"
        )
        with open(result_pickle_path, "wb") as pkl:
            pickle.dump(self.result, pkl)
            self.log.info(
                f"Saved results as pickle file {self.get_path('results', absolute=False)}"
            )
        self.paths["result_pickle"] = result_pickle_path

    def drop_exp_to_pickle(self):
        """Drop the experiment to pickle data."""
        file = f"{self.name}_experiment.pkl"
        result_pickle_path = self.get_path("results")
        file_path = os.path.join(result_pickle_path, file)
        with open(file_path, "wb") as pkl:
            pickle.dump(self, pkl)
            self.log.info(f"Saved as pickle file {file_path}")
        self.paths["experiment_pickle"] = file_path

    def do_analysis(self):
        pass


if __name__ == "__main__":
    YAML_PATH = sys.argv[1]
    exp = Experiment(do_synth_model, YAML_PATH)

    sc_result = exp.do_experiment(notification=True)
    exp.drop_exp_to_pickle()
    logger.info(exp.state)
    pprint.pprint(exp.paths)
