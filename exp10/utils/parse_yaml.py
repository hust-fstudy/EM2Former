# -*- coding: utf-8 -*-
# @Time: 2025/6/25
# @File: parse_yaml.py
# @Author: fwb
import yaml


def parse_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            cfgs = yaml.safe_load(f)
            return DotDict(cfgs)
    except FileNotFoundError:
        print(f"{file_path} does not exist!")
    except yaml.YAMLError as exc:
        print(f"{exc} format error!")


class DotDict(dict):
    def __getattr__(self, item):
        value = self.get(item)
        if value is None:
            raise KeyError(f"[{item}] key is not found!")
        else:
            return value
