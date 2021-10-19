import pandas as pd
import numpy as np
from statistics import NormalDist
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

data = pd.read_excel('data_Ass3_G2.xlsx', sheet_name='Returns', engine='openpyxl').drop('date', 1)


def ex1(data):
    print("EXERCISE 1")
    cov_matrix = data.iloc[:, 2:].cov()
    mean_vector = data.iloc[:, 2:].mean()
    weight_vector = pd.DataFrame({'w': ([0.2] * 5)})
    portfolio_value = 50000

    portfolio_my = weight_vector.values.flatten().T.dot(mean_vector)
    portfolio_sigma = (((weight_vector.values.flatten().T.dot(cov_matrix)).dot(weight_vector)) ** 0.5)[0]

    inv_dist_98 = NormalDist(0, 1).inv_cdf(0.98)

    value_at_risk = (- portfolio_my * portfolio_value) + (portfolio_sigma * portfolio_value * inv_dist_98)

    print("VaR_0.98:", round(value_at_risk, 1))
    print(round(value_at_risk / portfolio_value, 4) * 100, "%")


ex1(data)


def ex2(data):
    print("EXERCISE 2")
    portfolio_value = 50000
    weight_vector = pd.DataFrame({'w': ([0.2] * 5)})

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

    market_my = data['mkt'].mean()
    market_sigma = data['mkt'].std()

    print("Market MY:", round(market_my, 5))
    print("Market SIGMA:", round(market_sigma, 5), "\n")

    predicted_returns = data[['mkt']].copy()
    predicted_returns = pd.concat([predicted_returns] * 5, axis=1, ignore_index=True)
    predicted_returns.columns = data.iloc[:, 2:].columns

    for i in range(predicted_returns.shape[0]):
        for j in range(predicted_returns.shape[1]):
            predicted_returns.iloc[i, j] = predicted_returns.iloc[i, j] * var_matrix.iloc[j, 1] + var_matrix.iloc[j, 0]

    cov_matrix_model_2 = predicted_returns.cov()

    print("CORRELATION MATRIX")
    pd.set_option('display.max_columns', None)
    print(cov_matrix_model_2, "\n")

    portfolio_beta = var_matrix['mkt_beta'].mean()
    portfolio_alpha = var_matrix['alpha'].mean()
    portfolio_sigma = (var_matrix['sigma'].pow(2).mean()) ** 0.5

    inv_dist_98 = NormalDist(0, 1).inv_cdf(0.98)

    var_1 = -(portfolio_alpha + (market_my * portfolio_beta)) + (market_sigma * portfolio_beta * inv_dist_98)

    var_2 = - (portfolio_sigma * inv_dist_98)

    val_at_risk = ((var_1 ** 2 + var_2 ** 2) ** 0.5) * 50000

    print(val_at_risk)

    # MONTE CARLO
    def day_returns(alpha, beta, sigma, market_my, market_sigma, runs):
        asset_returns = []
        for i in range(runs):
            asset_returns.append(alpha + beta * np.random.normal(market_my, market_sigma) + np.random.normal(0, sigma))
        return asset_returns

    monte_carlo_returns = np.array(day_returns(portfolio_alpha, portfolio_beta, portfolio_sigma, market_my, market_sigma, 10000000))

    print(np.percentile(monte_carlo_returns, 2) * 50000)


ex2(data)