#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import os
import sys

import numpy as np
import pandas as pd
import pytest

from core.model.datasets import Datasets

df = pd.DataFrame(np.random.random((4, 5)), columns=["A", "B", "C", "D", "E"])


class TestUnits:
    """
    测试基本单元的运行
    """

    def test_create_dir(self):
        dataset = Datasets(name="test")
        abs_path = dataset.add_item_from_dataframe(
            data=df, name="test_data", save="testing"
        )
        assert os.path.isfile(abs_path)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def test_report(self):
        dataset = Datasets(name="test")
        assert dataset.items == "No items"
        dataset.add_item_from_dataframe(
            data=df, name="test_data", save="testing"
        )
        assert "test_data" in dataset.items
        assert dataset.dt("test_data") is df
        assert dataset.test_data.obj is df

        dataset.test_data.add_notes("A testing note.")
        assert "testing" in dataset.test_data.notes
        dataset.report()
