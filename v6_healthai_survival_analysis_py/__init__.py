# -*- coding: utf-8 -*-

""" Federated algorithm for 2-years survival for TNM data of NSCLC patients
Adapted from:
https://flower.dev/blog/2021-07-21-federated-scikit-learn-using-flower/
"""
import re
import warnings

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from vantage6.tools.util import info
from v6_healthai_survival_analysis_py.helper import coordinate_task
from v6_healthai_survival_analysis_py.helper import set_initial_params
from v6_healthai_survival_analysis_py.helper import set_model_params
from v6_healthai_survival_analysis_py.helper import get_model_parameters


def master(
        client, data: pd.DataFrame, max_iter: int = 15, org_ids: list = None
) -> dict:
    """ Master algorithm that coordinates the tasks and performs averaging

    Parameters
    ----------
    client
        Vantage6 user or mock client
    data
        DataFrame with the input data
    max_iter
        Maximum number of iterations to perform
    org_ids
        List with organisation ids to be used

    Returns
    -------
    results
        Dictionary with the final averaged result
    """

    # Get all organization ids that are within the collaboration or
    # use the provided ones
    info('Collecting participating organizations')
    organizations = client.get_organizations_in_my_collaboration()
    ids = [organization.get('id') for organization in organizations
           if not org_ids or organization.get('id') in org_ids]

    # Initialise the weights for the logistic regression
    info('Initializing logistic regression weights')
    model = LogisticRegression()
    model = set_initial_params(model)
    parameters = get_model_parameters(model)

    # TODO: check convergence by looking at the losses
    # The next steps are run until the maximum number of iterations is reached
    iteration = 0
    while iteration < max_iter:
        # The input for the partial algorithm
        info('Defining input parameters')
        input_ = {
            'method': 'logistic_regression_partial',
            'kwargs': {'parameters': parameters}
        }

        # Send partial task and collect results
        results = coordinate_task(client, input_, ids)
        info(f'Results: {results}')

        # TODO: dimension of coefficients should not be hard coded
        # Average model weights with the federated average method
        info('Run global averaging for model weights')
        coefficients = np.zeros((1, 3))
        for i in range(coefficients.shape[1]):
            coefficients[0, i] = np.mean([
                result['model'].coef_[0, i] for result in results
            ])
        intercept = np.mean([result['model'].intercept_ for result in results])
        intercept = np.array([intercept])

        # TODO: how to average losses and accuracy?
        # Average loss and accuracy with a simple average
        loss = np.mean([result['loss'] for result in results])
        accuracy = np.mean([result['accuracy'] for result in results])

        # Re-define the global parameters and update iterations counter
        parameters = (coefficients, intercept)
        iteration += 1

    return {
        'model': model,
        'loss': loss,
        'accuracy': accuracy
    }


def RPC_logistic_regression_partial(
        df: pd.DataFrame, parameters
) -> list:
    """ Partial method for federated logistic regression

    Parameters
    ----------
    df
        DataFrame with input data
    parameters
        Model weigths of logistic regression

    Returns
    -------
    results
        Dictionary with local model, loss and accuracy
    """
    # TODO: how to run data preparation steps only once and save result?
    # Drop rows with NaNs
    df = df.dropna(how='any')

    # Convert from categorical to numerical TNM, if necessary, values such as
    # Tx, Nx, Mx are converted to -1
    columns = ['t', 'n', 'm']
    for col in columns:
        if is_string_dtype(df[col]):
            df[col] = df[col].apply(lambda x: re.compile(r'\d').findall(x))
            df[col] = df[col].apply(
                lambda x: int(x[0]) if len(x) != 0 else -1
            )

    # Make sure no NaNs are left after transformation
    df = df.dropna(how='any')

    # Get 2-years survival
    df['date_of_diagnosis'] = pd.to_datetime(df['date_of_diagnosis'])
    df['date_of_fu'] = pd.to_datetime(df['date_of_fu'])
    df['days'] = df.apply(
        lambda x: (x['date_of_fu'] - x['date_of_diagnosis']).days, axis=1
    )
    df['survival'] = df.apply(
        lambda x: 'dead' if ((x['days'] <= 2*365) and
                             (x['vital_status'] == 'dead'))
        else 'alive',
        axis=1
    )

    # Get features and outcomes
    X = df[columns].values
    y = df['survival'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty='l2',
        max_iter=1,       # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Fitting local model
    model = set_model_params(model, parameters)
    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_train, y_train)
        info('Training round finished')

    # Evaluate model
    loss = log_loss(y_test, model.predict_proba(X_test))
    accuracy = model.score(X_test, y_test)

    # Results
    results = {
        'model': model,
        'loss': loss,
        'accuracy': accuracy
    }

    return results
