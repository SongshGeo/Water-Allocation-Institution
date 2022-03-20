#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Created date: 2022/3/20
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9
import sys

import yaml
import os

YAML_PATH = "metadata.yaml"
with open(YAML_PATH, "r", encoding="utf-8") as file:
    params = yaml.load(file.read(), Loader=yaml.FullLoader)
    file.close()


class SciPaper(dict):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    print(os.getcwd())
    os.chdir(os.getcwd())
    title = params.get("title")
    abstract = params.get("abstract")
    authors = params.get("authors").keys()
    keywords = params.get("keywords")
    keywords = ", ".join(keywords)
    print(title, abstract, authors, keywords)
    # for folder in ('00_cover_letter', '01_manuscript', '02_appendix'):
    #     os.system(f"cd {folder}")
    #     for name, element in zip(
    #             ['title', 'abstract', 'keyword'],
    #             [title, abstract, keywords]
    #     ):
    #         os.system(f"echo '{element}' > {name}.tex")
    #         os.system("cp ../*.sh .")
    #         os.system("zsh *.sh")
        # for name in ['title', 'abstract', 'keyword']:
        #     os.system(f"rm '{name}.tex'")
    for name, element in zip(
            ['title', 'abstract', 'keywords'],
            [title, abstract, keywords]
    ):
        os.system(f"echo '{element}' > 00_cover_letter/{name}.tex")
        os.system(f"echo '{element}' > 01_manuscript/{name}.tex")
        os.system(f"echo '{element}' > 02_appendix/{name}.tex")
        os.system(f"echo '{element}' > {name}.tex")
    # os.system(f'zsh make.sh')
    # for name in ['title', 'abstract', 'keyword']:
    #     os.system(f"rm '{name}.tex'")
    pass
