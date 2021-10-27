from common import *

import numpy as np

def transform_data(df_train:pd.DataFrame, df_test:pd.DataFrame):
    """Takes dataframe as input and transforms data"""
    # let's add a categorical variable
    countries = ['Germany', 'US']
    household_income = ['low', 'high']
    df_train["country"] = np.random.choice(countries, len(df_train))
    df_test["country"] = np.random.choice(countries, len(df_test))
    df_train["household_income"] = np.random.choice(household_income, len(df_train))
    df_test["household_income"] = np.random.choice(household_income, len(df_test))

    x_train = df_train.iloc[:, 1:-2].values.reshape(-1, 1, 24)
    x_test = df_test.iloc[:, 1:-2].values.reshape(-1, 1, 24)

    y_train = df_train.iloc[:, 0].values-1
    y_test = df_test.iloc[:, 0].values-1

    emb_vars_train = df_train.iloc[:, -2:].values
    emb_vars_test = df_test.iloc[:, -2:].values
    return x_train, x_test, y_train, y_test, emb_vars_train, emb_vars_test