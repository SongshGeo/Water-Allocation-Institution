#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import os

import pandas as pd

from .unit_base import ItemBase, UnitBase


class DataItem(ItemBase):
    def __init__(self, abs_path, category, **kwargs):
        super().__init__(**kwargs)
        self._abs_path = abs_path
        self._category = category
        self.update_metadata("path", abs_path)

    pass

    @property
    def rel_path(self):
        if self.abs_path:
            return os.path.relpath(self._abs_path)
        else:
            return None

    @property
    def category(self):
        return self._category


class Datasets(UnitBase):
    def __init__(self, categories=None, name="dataset"):
        super().__init__(unit_name=name)
        if not categories:
            categories = {
                "assets": "Original datasets",
                "source": "Source of used data",
                "out": "Outputs",
            }
        self._category = categories

    @property
    def categories(self):
        return self._category

    def dt(self, name):
        return self.get_item(name).obj

    def in_category(self, category):
        if category not in self.categories:
            False
        else:
            return True

    def add_item_from_dataframe(
        self, data, name, category="assets", save=False, description=""
    ):
        if not self.in_category(category):
            raise ValueError(
                f"{category} not valid in {self.categories.keys()}."
            )
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
            category=category,
            description=description,
        )
        item.update_metadata("category", category)
        item.update_metadata("path", abs_path)
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
