# -*- coding: utf-8 -*-

""" Helper functions for running NSCLC 2-years survival
"""
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from vantage6.tools.util import info
from typing import Tuple, Union, List

XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]


def coordinate_task(client, input: dict, ids: list) -> list:
    """ Coordinate tasks to be sent to data nodes, which includes dispatching
    the task, waiting for results to return and collect completed results

    Parameters
    ----------
    client
        Vantage6 user or mock client
    input
        Input parameters for the task, such as the method and its arguments
    ids
        List with organisation ids that will receive the task

    Returns
    -------
    results
        Collected partial results from all the nodes
    """

    # Create a new task for the desired organizations
    info('Dispatching node tasks')
    task = client.create_new_task(
        input_=input,
        organization_ids=ids
    )

    # Wait for nodes to return results
    info('Waiting for results')
    task_id = task.get('id')
    task = client.get_task(task_id)
    while not task.get('complete'):
        task = client.get_task(task_id)
        info('Waiting for results')
        time.sleep(1)

    # Collecting results
    info('Obtaining results')
    results = client.get_results(task_id=task.get('id'))

    return results


def set_initial_params(model: LogisticRegression):
    """
    Sets initial parameters as zeros
    """
    model.classes_ = np.array(['alive', 'dead'])
    model.coef_ = np.zeros((1, 3))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))
    return model


def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model
