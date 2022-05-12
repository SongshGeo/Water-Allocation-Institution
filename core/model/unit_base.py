#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9
import logging
import os
from collections import defaultdict
from pprint import pprint


class ItemBase(object):
    def __init__(self, name, obj, abs_path, metadata=None):
        if not metadata:
            metadata = defaultdict(str)
        self._name = name
        self._obj = obj
        self._abs_path = abs_path
        self._metadata = metadata

    @property
    def name(self):
        return self._name

    @property
    def abs_path(self):
        return self._abs_path

    @property
    def metadata(self):
        return self._metadata

    pass

    @property
    def obj(self):
        return self._obj

    def add_notes(self, note):
        if not isinstance(note, str):
            raise TypeError("Note must be a string.")
        self._metadata["notes"] += f"{note}\n"

    def rel_path(self):
        return os.path.realpath(self._abs_path)


class UnitBase(object):
    def __init__(self, unit_name, unit_base=None, project_base=None, log=None):
        if not project_base:
            project_base = os.getcwd()
        if not unit_base:
            unit_base = unit_name
        if not log:
            log = logging.getLogger(__file__)
        self._name = unit_name
        self._items = {}
        self._root = project_base
        self._module = unit_base
        self._log = log
        self._unit_path = os.path.join(project_base, unit_base)
        pass

    @property
    def items(self):
        if self._items.__len__() == 0:
            return "No items"
        else:
            return [self.get_item(item).name for item in self._items]

    @property
    def path(self):
        return self._unit_path

    def add_item(self, item):
        """
        Add a new item to the items.

        Arguments:
            item -- An item object to add.

        Raises:
            ValueError: name cannot be empty or repeated.
        """
        name = item.name
        if name in self.items():
            raise ValueError(f"{name} already in items.")
        self._items[name] = item
        self.log.info(f"Added item {name} to {self.name}.")
        pass

    def get_item(self, name):
        return self._items.get(name)

    def add_notes(self, name, notes):
        item = self.get_item(name)
        item.add_notes(notes)
        self.log.info(f"Add note {notes} to item {name}.")
        pass

    def has_dir(self, dirname):
        return os.path.isdir(os.path.join(self.path, dirname))

    def add_dir(self, dirname):
        dir_path = os.path.join(self.path, dirname)
        if os.path.isdir(dir_path):
            raise FileExistsError(f"Directory {dirname} already exists.")
        os.mkdir(dir_path)

    def return_or_add_dir(self, dirname):
        if not self.has_dir(dirname):
            self.add_dir(dirname)
            self.log.info(f"Add a new directory {dirname} by {self.name}.")
        return os.path.join(self.path, dirname)

    def save_under_unit(self):
        pass

    def report(self):
        pprint(
            f"""{self._name}:\n
              {self.items}
              """
        )
        pass

    pass
