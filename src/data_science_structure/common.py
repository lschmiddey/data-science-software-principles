import pandas as pd
import pickle
import re
from collections.abc import Iterable
from argparse import ArgumentParser

import numpy as np


def save_obj(obj: str, name: str):
    """Save object as a pickle file to a given path."""
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name: str):
    """Load object as a pickle file to a given path."""
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


def cat_transform(train_var: np.array, test_var: np.array):
    """remap number to categorical variable and save dictionaries.
    test_var is then mapped according to the train_var"""
    dict_list = []
    train_var_shape = train_var.shape
    test_var_shape = test_var.shape
    if len(train_var.shape)==1:
        train_var = train_var.reshape(-1,1)
        test_var = test_var.reshape(-1,1)
    for i in range(train_var.shape[1]):
        dict_ = {j: element for j, element in enumerate(set(train_var[:,i]))}
        dict_list.append(dict_)
    dict_inv_list = [{v: k for k, v in dict_list[i].items()}
                     for i, dict_ in enumerate(dict_list)]

    # map numpy arrays
    for i in range(train_var.shape[1]):
        train_var[:,i] = np.vectorize(dict_inv_list[i].get)(train_var[:,i])
    for i in range(test_var.shape[1]):
        test_var[:,i] = np.vectorize(dict_inv_list[i].get)(test_var[:,i])

    train_var = train_var.reshape(train_var_shape).astype(int)
    test_var = test_var.reshape(test_var_shape).astype(int)

    return train_var, test_var, dict_list, dict_inv_list


def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    #Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name:str)->str:
    "Change `name` from camel to snake style."
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


def parse_args(logger):
    parser = ArgumentParser()
    parser.add_argument("--model_name", "-model", help="which model to train, 03days, 14days, or 30days")

    args = parser.parse_args()

    if not args.model_name:
        logger.error('model_name is not provided - aborting')
        raise ValueError('model_name is not provided')

    return args