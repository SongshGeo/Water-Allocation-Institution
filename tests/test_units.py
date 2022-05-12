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

from core.model.datasets import Datasets


class TestUnits:
    def test_create_dir(self):
        pass

    def test_report(self):
        dataset = Datasets(name="data")
        assert dataset.items == "No items"
        df = pd.DataFrame(
            np.random.random((4, 5)), columns=["A", "B", "C", "D", "E"]
        )
        dataset.add_item_from_dataframe(data=df, name="test", save="test")
