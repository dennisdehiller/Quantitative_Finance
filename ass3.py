import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

data = pd.read_excel('data_Ass3_G2.xlsx', sheet_name='Returns', engine='openpyxl').drop('date', 1)

cov_matrix = data.iloc[:, 2:].cov()
mean_vector = data.iloc[:, 2:].mean()
weight_vector = pd.DataFrame({'w': ([0.2] * 5)})

portfolio_my = weight_vector.values.flatten().T.dot(mean_vector)
portfolio_sigma = (((weight_vector.values.flatten().T.dot(cov_matrix)).dot(weight_vector)) ** 0.5)[0]

portfolio_value = 50000
alpha = 0.98



VaR = (-portfolio_my * portfolio_value) + (portfolio_sigma * portfolio_value * alpha)
