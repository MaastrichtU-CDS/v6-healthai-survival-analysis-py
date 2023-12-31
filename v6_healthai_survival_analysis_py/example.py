# -*- coding: utf-8 -*-

""" Sample code to test the federated algorithm with a mock client
"""
import os
import numpy as np
from vantage6.tools.mock_client import ClientMockProtocol


# Start mock client
data_dir = os.path.join(
    os.getcwd(), 'v6_healthai_survival_analysis_py', 'local'
)
client = ClientMockProtocol(
    datasets=[
        os.path.join(data_dir, 'data1.csv'),
        os.path.join(data_dir, 'data2.csv')
    ],
    module='v6_healthai_survival_analysis_py'
)

# Get mock organisations
organizations = client.get_organizations_in_my_collaboration()
print(organizations)
ids = [organization['id'] for organization in organizations]

# Check master method
master_task = client.create_new_task(
    input_={
        'master': True,
        'method': 'master',
        'kwargs': {
            'org_ids': [0, 1],
            'max_iter': 5
        }
    },
    organization_ids=[0, 1]
)
results = client.get_results(master_task.get('id'))
model = results[0]['model']
print(model)

X = np.array([[0, 0, 0]])
print(model.predict_proba(X))
