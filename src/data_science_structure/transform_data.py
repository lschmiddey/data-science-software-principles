from common import *

import numpy as np

class BaseData():
    def __init__(self):
        pass

    def load_data(self, data_path:str):
        self.df_train = pd.read_csv(f'{data_path}/ItalyPowerDemand_TRAIN.txt', header=None,delim_whitespace=True)
        self.df_test = pd.read_csv(f'{data_path}/ItalyPowerDemand_TEST.txt', header=None, delim_whitespace=True)

    def make_fake_cat_vars(self):
        """Takes dataframe as input and transforms data"""
        # let's add a categorical variable
        countries = ['Germany', 'US']
        household_income = ['low', 'high']
        self.df_train["country"] = np.random.choice(countries, len(self.df_train))
        self.df_test["country"] = np.random.choice(countries, len(self.df_test))
        self.df_train["household_income"] = np.random.choice(household_income, len(self.df_train))
        self.df_test["household_income"] = np.random.choice(household_income, len(self.df_test))

    def transform_data(self):
        self.x_train = self.df_train.iloc[:, 1:-2].values.reshape(-1, 1, 24)
        self.x_test = self.df_test.iloc[:, 1:-2].values.reshape(-1, 1, 24)

        self.y_train = self.df_train.iloc[:, 0].values-1
        self.y_test = self.df_test.iloc[:, 0].values-1

        self.emb_vars_train = self.df_train.iloc[:, -2:].values
        self.emb_vars_test = self.df_test.iloc[:, -2:].values

    def categorize_variables(self):
        self.emb_vars_train, self.emb_vars_test, self.dict_embs, self.dict_inv_embs = \
            cat_transform(self.emb_vars_train, self.emb_vars_test)