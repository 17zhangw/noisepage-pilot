import numpy as np
import pandas as pd
import pickle
from enum import Enum


# If you change this indexing, change behavior/model_workload/models/buffer_page_model.py
class OpType(Enum):
    SELECT = 1
    INSERT = 2
    UPDATE = 3
    DELETE = 4


def keyspace_metadata_output(container, *args):
    with open(f"{container}/keyspaces.pickle", "wb") as f:
        for arg in args:
            pickle.dump(arg, f)


def keyspace_metadata_read(container):
    args = []
    with open(f"{container}/keyspaces.pickle", "rb") as f:
        while True:
            try:
                args.append(pickle.load(f))
            except:
                break

    return tuple(args)


class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]
