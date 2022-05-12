#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9
import datetime
import os

from .unit_base import ItemBase, UnitBase


class DataItem(ItemBase):
    pass


class Datasets(UnitBase):
    def __init__(self, name="dataset"):
        super().__init__(unit_name=name)

    @property
    def dt(self):
        return self._dataset

    def add_item_from_dataframe(self, data, name, save=False):
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
            metadata={
                "ctime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        self._items[name] = item
        return abs_path

    def process_dataset(self, data, how, name=None):
        pass

    def metadata(self, name=None):
        pass

    pass
