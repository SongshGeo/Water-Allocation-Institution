#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

from core.unit_base import ItemBase, UnitBase


class AnalystItem(ItemBase):
    def __init__(self, data_item, method_item, **kwargs):
        super().__init__(**kwargs)
        self._data = data_item
        self._method = method_item

    @property
    def data(self):
        return self._data

    @property
    def method(self):
        return self._method


class Analyst(UnitBase):
    def __init__(self, name="analyst"):
        super().__init__(unit_name=name)

    def add_analyst_item(
        self, name, description, data_item, method_item, check=None
    ):
        item = AnalystItem(
            obj=check,
            data_item=data_item,
            method_item=method_item,
            name=name,
            description=description,
        )
        item.update_metadata("data", data_item.name)
        item.update_metadata("method", method_item.name)
        self.add_item(item)
        pass

    def report(self, show=True, show_notes=False, max_width=30):
        table = super().report(show=False, show_notes=show_notes)
        data_list, method_list = [], []
        for item in self.items:
            item = self.get_item(item)
            data = item.data.name
            method = item.method.name
            data_list.append(data)
            method_list.append(method)
        table.add_column("Data", data_list)
        table.add_column("Method", method_list)
        table.max_width = max_width
        if show:
            print(table)
        else:
            return table

    pass
