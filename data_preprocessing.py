import numpy as np
import pandas as pd


def data_analysis(data):

    _data_exploration(data)

    data = _missing_values(data)

    print('\nAfter treating missing values\n')
    _data_exploration(data)

    if 'id' in list(data.columns.values):
        data = _del_id(data)

    if 'age' in list(data.columns.values):
        data = _age_discretization(data)

    print('\nAfter treating id and age\n')
    _data_exploration(data)

    return data


def _data_exploration(data):

    no_of_attributes = len(data.iloc[0])
    for attr in range(no_of_attributes):
        print('Attribute ' + str(attr) + ': ' + data.columns.values[attr])
        print(np.unique(np.array(data.iloc[:, attr])))

    return


def _missing_values(data):

    missing = '?'
    missing_value = 'missing'
    data = data.replace(to_replace=missing, value=missing_value)

    return data


def _del_id(data):

    label = 'id'
    data = data.drop(columns=label)

    return data


def _age_discretization(data):

    all_cases = range(len(data))
    age_column = data.columns.get_loc('age')

    for case in all_cases:
        age = data.iloc[case, age_column]

        if age == 'missing':
            continue
        else:
            age = int(age)

        if age < 10:
            data.iloc[case, age_column] = '0-9'
        elif age < 20:
            data.iloc[case, age_column] = '10-19'
        elif age < 30:
            data.iloc[case, age_column] = '20-29'
        elif age < 40:
            data.iloc[case, age_column] = '30-39'
        elif age < 50:
            data.iloc[case, age_column] = '40-49'
        elif age < 60:
            data.iloc[case, age_column] = '50-59'
        elif age < 70:
            data.iloc[case, age_column] = '60-69'
        elif age < 80:
            data.iloc[case, age_column] = '70-79'
        else:
            data.iloc[case, age_column] = '80-over'

    return data
