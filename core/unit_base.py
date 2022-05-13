#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9
import datetime
import logging
import os
from textwrap import fill

from prettytable import PrettyTable

from .src.tools import save_dict_to_yaml


class ItemBase(object):
    def __init__(self, name, obj, abs_path, description="", metadata=None):
        self._name = name
        self._obj = obj
        self._abs_path = abs_path
        self._description = description
        self._ctime = datetime.datetime.now()
        self._mtime = datetime.datetime.now()

        if not metadata:
            metadata = {}
            metadata["name"] = name
            metadata["path"] = abs_path
            metadata["description"] = description
            metadata["ctime"] = self._ctime.strftime("%Y-%m-%d %H:%M:%S")
            metadata["mtime"] = self._mtime.strftime("%Y-%m-%d %H:%M:%S")
            metadata["notes"] = {}
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

    @property
    def description(self):
        return self._description

    @property
    def obj(self):
        return self._obj

    @property
    def rel_path(self):
        return os.path.relpath(self._abs_path)

    @property
    def notes(self):
        return self._metadata.get("notes")

    @property
    def str_notes(self):
        str_notes = "".join(
            [f"({i+1}){self.notes[k]}. " for i, k in enumerate(self.notes)]
        )
        return str_notes

    def show_notes(self, show=True):
        notes = self.notes
        table = PrettyTable()
        table.field_names = ["Time", "Note"]
        for k, v in notes.items():
            table.add_row([k, v])
        if show:
            print(f"'{self.name}' has {len(notes)} notes:")
            print(table)
        return notes

    def update_mtime(self):
        self._mtime = datetime.datetime.now()
        self._metadata["mtime"] = self._mtime.strftime("%Y-%m-%d %H:%M:%S")

    def add_notes(self, note):
        if not isinstance(note, str):
            raise TypeError("Note must be a string.")
        notes = self.notes
        notes[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")] = note
        self.update_metadata("notes", notes)

    def update_metadata(self, k, value):
        self._metadata[k] = value
        self.update_mtime()
        pass

    def dump_metadata(self, save_path=""):
        if not save_path:
            save_path = os.path.join(save_path, f"{self.name}.yaml")
        save_dict_to_yaml(self.metadata, save_path)


class UnitBase(object):
    def __init__(self, unit_name, unit_base=None, project_base=None, log=None):
        if not project_base:
            project_base = os.getcwd()
        if not unit_base:
            unit_base = unit_name
        if not log:
            log = logging.getLogger(__file__)
        if unit_base not in os.listdir(project_base):
            os.mkdir(os.path.join(project_base, unit_base))
            log.warning(f"No {unit_base} under the project base, created.")
        self._name = unit_name
        self._items = {}
        self._root = project_base
        self._module = unit_base
        self.log = log
        self._unit_path = os.path.join(project_base, unit_base)

    @property
    def items(self):
        if self._items.__len__() == 0:
            return "No items"
        else:
            return [self.get_item(item).name for item in self._items]

    @property
    def path(self):
        return self._unit_path

    @property
    def name(self):
        return self._name

    def add_item(self, item, **kwargs):
        """
        Add a new item to the items.

        Arguments:
            item -- An item object to add.

        Raises:
            ValueError: name cannot be empty or repeated.
        """
        name = item.name
        item.update_metadata("unit", self.name)
        for k, v in kwargs.items():
            item.update_metadata(k, v)
        if name in self.items:
            raise ValueError(f"{name} already in items.")
        if hasattr(self, name):
            raise ValueError(f"Attribute {name} already in items.")
        self._items[name] = item
        self.log.info(f"Added item {name} to {self.name}.")
        setattr(self, name, item)
        pass

    def get_item(self, name):
        return self._items.get(name)

    def add_notes(self, name, notes):
        item = self.get_item(name)
        item.add_notes(notes)
        self.log.info(f"Add note to item {name}.")
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

    def report(self, show_notes=False, max_width=30):
        table = PrettyTable()
        table.field_names = ["Name", "Path", "Description"]
        for item in self.items:
            path = self.get_item(item).rel_path
            description = self.get_item(item).description
            table.add_row([item, path, description])
        if show_notes:
            str_notes = []
            for item in self.items:
                str_notes.append(self.get_item(item).str_notes)
            table.add_column("Notes", str_notes)
        table.max_width = max_width
        print(
            f"{self.__class__.__name__} '{self.name}' has {self.items.__len__()} items:"
        )
        print(table)

    def dump_metadata(self, save_path=None):
        if not save_path:
            save_path = os.path.join(self.path, f"{self.name}.yaml")
        metadata_dicts = {
            item: self.get_item(item).metadata for item in self.items
        }
        save_dict_to_yaml(metadata_dicts, save_path)

    pass
