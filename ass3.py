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
    resids = pd.DataFrame()
    for i in range(5):
        return_data = data[['mkt']].copy()
        return_data['ret'] = data.iloc[:, 2 + i]

        model = smf.ols("ret ~ mkt", data=return_data)
        result = model.fit()
        resids.append(result.resid)



ex2(data)