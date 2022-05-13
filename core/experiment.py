#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# Created date: 2022-02-10
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import datetime
import logging
import os
import pickle

import yaml
from attrs import define, field

from .src.tools import send_finish_message


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
        yaml_file,
        name="exp",
        model=None,
        experiment_path=None,
        results_path=None,
        log=None,
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
            name = parameters.get("name")

            # root path: from yaml > default project root
            root = parameters.get("root")
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
        if not log:
            log = logging.getLogger(__file__)
        self.log = log
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
        self.paths["log"] = os.path.join(root, f"{self.log.name}.log")

        for dataset in self.datasets.keys():
            full_dataset_path = os.path.join(root, self.datasets.get(dataset))
            self.paths[dataset] = full_dataset_path
            self.datasets[dataset] = full_dataset_path

        for key, path in self.paths.items():
            if key == "log":
                continue
            if not os.path.exists(path):
                os.mkdir(path)
                self.log.warning(
                    f"{key} folder made in {os.path.dirname(path)}."
                )

    def get_path(self, key="experiment", absolute=True):
        path = self.paths.get(key)
        root = self.paths.get("root")
        if absolute:
            return path
        else:
            return os.path.relpath(path, root)

    def change_experiment_logger(self, exp_log):
        """
        Set up log file to this experiment and switch the default logger.
        :return: The new exp logger.
        """
        # Default project logger
        exp_path = self.get_path("experiment")
        exp_rel_path = self.get_path("experiment", absolute=False)
        self.log.warning(
            f"Experiment {self.name} log file will be set under {exp_rel_path}."
        )
        # Change logger
        self.log = exp_log
        exp_log.info(f"Experiment {self.name} log file set.")
        self._updated_time = datetime.datetime.now().strftime(self._strftime)
        self.paths["log"] = os.path.join(exp_path, f"{self.name}.log")
        return exp_log

    def do_experiment(self, model=None, notification=False):
        """
        Main func, do experiment.
        :param model: how to do the experiment.
        :param notification: send message to user.
        :return: state of the experiment.
        """
        log = self.log
        if model:
            self.model = model
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

    def load_from_pickle(self):
        with open(self.paths.get("yaml"), "r", encoding="utf-8") as file:
            params = yaml.load(file.read(), Loader=yaml.FullLoader)
            root = params.get("root")
            result_path = params.get("results_path")
            file.close()
        path = os.path.join(root, result_path, f"{self.name}_experiment.pkl")
        with open(path, "rb") as pkl:
            obj = pickle.load(pkl)
        return obj

    def list_datasets(self):
        names = self.datasets.keys()
        categories = []
        for name in names:
            if "processed" in self.datasets.get(name):
                category = "processed"
            elif "result" in self.datasets.get(name):
                category = "result"
            else:
                category = "Unknown"
            categories.append(category)
        return [(n, c) for n, c in zip(names, categories)]

    def update(self, attr=None, val=None, msg="Nothing."):
        """Update something."""
        flag = False
        if not attr and not val:
            self._updated_time = datetime.datetime.now().strftime(
                self._strftime
            )
            self.log.info(f"Updated: {msg}")
            return flag
        old_val = getattr(self, attr)
        if val != old_val:
            flag = True
            setattr(self, attr, val)
            self._updated_time = datetime.datetime.now().strftime(
                self._strftime
            )
            self.log.info(f"{attr} changed to {val} at {self._updated_time}.")
        return flag

    def add_item(self, attr=None, label=None, val=None):
        """Add items"""
        flag = False
        if hasattr(self, attr):
            self.__getattribute__(attr)[label] = val
            self.update(msg=f"Add {label} in {attr}.")
            flag = True
        return flag

    def reload_from_yaml(self, yaml_file):
        with open(yaml_file, "r", encoding="utf-8") as file:
            parameters = yaml.load(file.read(), Loader=yaml.FullLoader)
            file.close()
        new = []
        for attr in ("parameters", "analysis", "datasets"):
            params_dict = parameters.get(attr)
            for label, val in params_dict.items():
                if label not in self.__getattribute__(attr):
                    new.append(label)
                self.add_item(attr, label, val)
        self.authors = parameters.get("author")
        self.description = parameters.get("description")
        return new

    def do_analysis(self, model, notification=False):
        self.update(msg=f"Start analysis {model.__name__}")
        result = model(self, self.parameters, self.analysis)
        self.update(msg=f"End analysis {model.__name__}.")
        self.result = result
        self.update("state", "analyzed")
        if notification:
            sent = send_finish_message(self.name)
            self.update(msg=f"Notification sending msg: {sent['errmsg']}.")
        return self.state


if __name__ == "__main__":
    pass
