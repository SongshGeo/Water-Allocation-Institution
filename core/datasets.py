#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import os
from collections import defaultdict

import pandas as pd

from .unit_base import ItemBase, UnitBase


class DataItem(ItemBase):
    pass


class Datasets(UnitBase):
    def __init__(self, name="dataset"):
        super().__init__(unit_name=name)

    def dt(self, name):
        return self.get_item(name).obj

    def add_item_from_dataframe(
        self, data, name, category="resource", save=False
    ):
        if save:
            path = self.path
            if isinstance(save, str):
                path = self.return_or_add_dir(save)
            abs_path = os.path.join(path, f"{name}.csv")
            data.to_csv(abs_path, index=False)
        else:
            abs_path = None

        item = DataItem(
            name=name,
            obj=data,
            abs_path=abs_path,
            description="Dataframe added from pd object.",
        )
        item.update_metadata("category", category)
        self.add_item(item)
        return abs_path

    def add_item_from_csv(self, rel_path, name, category="Source", save=False):
        path = os.path.join(self.path, rel_path)
        data = pd.read_csv(path, index_col=0)
        os.path.basename(path)
        self.add_item_from_dataframe(data, name)
        pass

    def load_from_pickle(self, filename):
        pass

    def dump_all(self, out_path):
        pass

    def process_dataset(self, data, how, name=None):
        pass

    def metadata(self, name=None):
        pass

    pass
