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
