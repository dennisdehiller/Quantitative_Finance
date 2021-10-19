import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statistics import NormalDist

data = pd.read_excel('data_Ass3_G2.xlsx', sheet_name='Returns', engine='openpyxl').drop('date', 1)

def ex1(data):
    cov_matrix = data.iloc[:, 2:].cov()
    mean_vector = data.iloc[:, 2:].mean()
    weight_vector = pd.DataFrame({'w': ([0.2] * 5)})


    portfolio_my = weight_vector.values.flatten().T.dot(mean_vector)
    portfolio_sigma = (((weight_vector.values.flatten().T.dot(cov_matrix)).dot(weight_vector)) ** 0.5)[0]

    portfolio_value = 50000

    inv_dist_98 = NormalDist(0, 1).inv_cdf(0.98)

    VaR = (-portfolio_my * portfolio_value) + (portfolio_sigma * portfolio_value * inv_dist_98)

    # print("Exercise 1")
    # # print ("My: ", portfolio_my)
    # # print ("Sigma: ", portfolio_sigma)
    # print("Model 1")
    # print('Value at risk:', VaR)
    # print(round(VaR / portfolio_value, 4) * 100, "%")

ex1(data)

def ex2(data):
    variables = []
    for i in range(5):
        return_data = data[['mkt']].copy()
        return_data['ret'] = data.iloc[:, 2 + i]

        model = smf.ols("ret ~ mkt", data=return_data)
        result = model.fit()
        residual_std = result.resid.std()
        variables.append([result.params[0], result.params[1], residual_std])

    var_matrix = pd.DataFrame(variables)
    var_matrix.columns = ['alpha', 'mkt_beta', 'sigma']
    var_matrix.index = data.iloc[:, 2:].columns
    print(var_matrix, "\n")

    print("Market MY:", round(data['mkt'].mean(), 5))
    print("Market SIGMA:", round(data['mkt'].std(), 5), "\n")

    predicted_returns = data[['mkt']].copy()
    predicted_returns = pd.concat([predicted_returns] * 5, axis=1, ignore_index=True)
    predicted_returns.columns = data.iloc[:, 2:].columns

    for i in range(predicted_returns.shape[0]):
        for j in range(predicted_returns.shape[1]):
            predicted_returns.iloc[i, j] = predicted_returns.iloc[i, j] * var_matrix.iloc[j, 1] + var_matrix.iloc[j, 0]

    cov_matrix_model_2 = predicted_returns.cov()

    print("CORRELATION MATRIX")
    pd.set_option('display.max_columns', None)
    print(cov_matrix_model_2)

ex2(data)


