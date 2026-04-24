# -*- coding: utf-8 -*-
# @Time: 2024/11/22
# @File: uniform_event.py
# @Author: fwb
from dataclasses import dataclass
import numpy as np


@dataclass
class Event:
    x: np.array
    y: np.array
    t: np.array
    p: np.array
    label: np.array

    def __int__(self, x, y, t, p, label):
        if (len(x) != len(y)) or (len(y) != len(t)) or (len(t) != len(p)):
            raise ValueError("The dimensions of x, y, t, p, and label must be the same")
        assert np.all(x >= 0) or np.all(y >= 0) or np.all(t >= 0) or np.all(label >= 0)
        self.x = x
        self.y = y
        self.t = t
        self.p = p
        self.label = label

    def to_uniform_format(self):
        """
        :return events: an event dictionary contains x, y, t, p, label
        """
        t_permutation = np.argsort(self.t)
        events = {'x': self.x[t_permutation],
                  'y': self.y[t_permutation],
                  't': self.t[t_permutation],
                  'p': self.p[t_permutation],
                  'label': self.label}
        return events
