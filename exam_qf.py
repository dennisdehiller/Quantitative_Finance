import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#            Uncomment # in order to run code. For benchmark portfolio line 84, for Rev1 line 158.


# Import daily data, replacing NaN with 0
daily_returns = pd.read_csv('data_daily_ret.csv', sep=',', header=None).replace(np.nan, 0)
daily_permno = pd.read_csv('data_daily_permno.csv', header=None).replace(np.nan, 0)
daily_date = pd.read_csv('data_daily_date.csv', header=None).replace(np.nan, 0)

# Import monthly data, replacing NaN with 0
monthly_ret = pd.read_csv('data_monthly_ret.csv', sep=',', header=None).replace(np.nan, 0)
monthly_cap = pd.read_csv('data_monthly_cap.csv', sep=',', header=None).replace(np.nan, 0)
monthly_permno = pd.read_csv('data_monthly_permno.csv', sep=',', header=None).replace(np.nan, 0)
month = pd.read_csv('data_monthly_month.csv', header=None).replace(np.nan, 0)
monthly_mkt = pd.read_csv('data_monthly_mkt.csv', header=None).replace(np.nan, 0)
monthly_rf = pd.read_csv('data_monthly_rf.csv', header=None).replace(np.nan, 0)


def benchmark_portfolio(monthly_cap, monthly_ret, monthly_rf):

    # Calculating cumulative returns

    cap_sum = []
    for j in (monthly_cap.values):
        cap_sum.append(np.sum(j))

    monthly_cap = monthly_cap.divide(cap_sum, axis=0)

    portfolio_ret = []
    weighted_ret = monthly_ret.multiply(monthly_cap)
    for j in (weighted_ret.values):
        portfolio_ret.append(np.sum(j))

    avg_ret = np.mean(portfolio_ret[61:])
    std_ret = np.std(portfolio_ret[61:])
    sharpe_ratio = (np.mean(portfolio_ret[61:]) - np.mean(monthly_rf[61:])) / np.std(portfolio_ret[61:])


    # Risk measures
    value_at_risk95 = avg_ret + (std_ret * 1.645)
    value_at_risk98 = avg_ret + (std_ret * stats.norm.ppf(0.99))
    ES_95 = avg_ret + ((std_ret * stats.norm.ppf(0.95)) / (0.95))

    print('BENCHMARK PORTFOLIO')
    print('Value at risk (95%):', value_at_risk95)
    print('Value at risk (99%):', value_at_risk98)
    print('Expected Shortfall:', ES_95)

    #Plotting
    portfolio_ret = pd.DataFrame(portfolio_ret[61:])

    plt.plot((1 + portfolio_ret).cumprod())
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.title("Benchmark portfolio")
    plt.savefig("benchmark.png", dpi=300)
    plt.show()

    #Testing distribution
    mean = np.mean(portfolio_ret[61:])
    #print(mean)
    std = np.std(portfolio_ret[61:])
    #print(std)
    skew = (portfolio_ret[61:]).skew()
    #print(skew)
    kurt = (portfolio_ret[61:]).kurt()
    #print(kurt)


    delta_p = (std) * np.cbrt((skew / 2))
    print('Benchmark Delta:', delta_p)
    eta_p = np.sqrt(((std ** 2) - (delta_p ** 2)))
    print('Benchmark Eta:', eta_p)
    omega_p = mean - delta_p
    print('Benchmark Omega:', omega_p)

    # Performance stat
    print('Benchmark Porfolio:', 'Avg Portfolio return:', round(avg_ret, 6), 'Portfolio sigma:', round(std_ret, 6), 'Sharpe Ratio:', round(sharpe_ratio, 6))

#benchmark_portfolio(monthly_cap, monthly_ret, monthly_rf)



def rev1_portfolio(monthly_ret, monthly_rf, monthly_cap):
    average_ret_47m = monthly_ret.rolling(47).mean()

    #Calculating weights and returns, which stocks to include

    caps = []
    portfolio = []
    for i in range(47, monthly_ret.shape[0] - 13):
        sorted_zipped = sorted(zip(average_ret_47m.iloc[i], monthly_ret.iloc[i + 13], monthly_cap.iloc[i]))
        sorted_zipped = sorted_zipped[: round(len(sorted_zipped) / 10)]
        portfolio.append(np.mean(list(zip(*sorted_zipped))[1]))
        caps.append(np.mean(list(zip(*sorted_zipped))[2]))


    # Plotting cumulative returns

    portfolio_returns = pd.DataFrame(portfolio)
    cumulative_return = np.cumprod(1 + portfolio_returns) - 1
    plt.plot(cumulative_return)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.title("Rev1 Portfolio")
    plt.savefig("rev1.png", dpi=300)
    plt.show()

    #Plotting distribution
    plt.hist(portfolio_returns, bins=40)
    plt.savefig("histogram.png", dpi=300)
    plt.show()

    # Testing distribution
    mean = portfolio_returns.mean()
    #print(mean)
    std = portfolio_returns.std()
    #print(std)
    skew = portfolio_returns.skew()
    #print(skew)
    kurt = portfolio_returns.kurt()
    #print(kurt)

    print('REV1 PORTFOLIO')

    delta = (std) * np.cbrt((skew/2))
    print('Rev1 Delta:', delta)
    eta = np.sqrt(((std**2)-(delta**2)))
    print('Rev1 Eta:', eta)
    omega = (mean - delta)
    print('Rev1 Omega:', omega)


    # Portfolio performance
    risk_free = monthly_rf.iloc[61:].mean()
    mean_portfolio_returns = portfolio_returns.mean()
    sigma_portfolio_returns = np.std(portfolio_returns)
    sharpe_ratio = ((mean_portfolio_returns - risk_free) / sigma_portfolio_returns)

    # Risk measures
    value_at_risk95 = mean_portfolio_returns + (sigma_portfolio_returns*1.645)
    value_at_risk99 = mean_portfolio_returns + (sigma_portfolio_returns * stats.norm.ppf(0.99))
    ES_95 = mean_portfolio_returns + ((sigma_portfolio_returns * stats.norm.ppf(0.95)) /(0.95))

    print('Avg returns:', round(mean_portfolio_returns, 6))
    print('Sigma_p:', round(sigma_portfolio_returns, 6))
    print('Sharpe ratio:', round(sharpe_ratio, 6))

    print('Value at risk (95%):', value_at_risk95)
    print('Value at risk (99 %):', value_at_risk99)
    print('Expected Shortfall (95%):', ES_95)


#rev1_portfolio(monthly_ret, monthly_rf, monthly_cap)

