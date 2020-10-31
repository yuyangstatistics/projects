# -*- coding: utf-8 -*-
"""
Unemployment Data Parsing

This script collects the unemployment statistics from
the 5 different states are saves a loadable dictionary
for easy access.
"""
import pandas as pd
import numpy as np

import pickle
import re

states = ['AZ', 'CO', 'IA', 'PA', 'VA']
csv_files = [state.lower() + '_unemp.csv' for state in states]

unemp_data = None

for csv_file, state in zip(csv_files, states):
    state_data = pd.read_csv(csv_file)
    state_data.drop(['Series ID', 'Period'], axis=1, inplace=True)
    
    state_data.rename(columns={'Year': 'year', 'Label': 'year_month', 
                               'Value': 'unemp_rate'},
                      inplace=True)
    state_data['state'] = state
    if unemp_data is None:
        unemp_data = state_data
    else:
        unemp_data = train_data = pd.concat([unemp_data, state_data], axis=0)


short_month_to_long = {
        'Jan': 'January',
        'Feb': 'February',
        'Mar': 'March',
        'Apr': 'April',
        'May': 'May',
        'Jun': 'June',
        'Jul': 'July',
        'Aug': 'August',
        'Sep': 'September',
        'Oct': 'October',
        'Nov': 'November',
        'Dec': 'December'
        }
month_data = unemp_data['year_month'].values
month_re_object = re.compile('\d+\s(\w{3})')
month_data = [short_month_to_long[re.match(rexp_object, x).group(1)]
 for x in month_data]
unemp_data['month'] = month_data
unemp_data.drop(['year_month'], inplace=True, axis=1)

unemp_data.to_pickle('unemp_data.pkl')

