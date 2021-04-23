import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
import seaborn as sns
import statsmodels.api as sm
from math import sqrt

from prophet import Prophet


import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.preprocessing import power_transform
from sklearn.metrics import mean_squared_error


## some datetime conversion warning
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
# import seaborn as sns

plt.style.use('seaborn')

def prophet_add_regressors(col_list):
    '''adds additional regressors to Prophet model'''
    model = Prophet(interval_width=0.95)
    for col in col_list: 
        model.add_regressor(col)
    return model

def split_fit_predict(model, df, train_low, train_upper, test_low, test_upper):
    X_train = df[train_low:train_upper]
    X_test = df[test_low:test_upper]
    model = model.fit(X_train)
    pred = model.predict(X_test.drop(columns='y'))
    rmse = sqrt(mean_squared_error(X_test['y'], pred['yhat']))
    print (f'RMSE {df.columns} = {rmse}')
    return pred


def csv_with_datetime(filepath, col_name): 
    '''reads in a csv and converts specified column to datetime format and lablel ds
    and sets index to datetime'''
    df = pd.read_csv(filepath)
    df['ds'] = pd.to_datetime(df[col_name])
    df.set_index(pd.to_datetime(df[col_name]), inplace=True)
    df.drop(col_name, axis=1, inplace=True)
    return df




def split_and_windowize(data, n_prev, n_test=365):
    n_predictions = len(data) - 2*n_prev
    
    n_train = n_predictions - n_test   
    
    x_train, y_train = windowize_data(data[:n_train], n_prev)
    x_test, y_test = windowize_data(data[n_train:], n_prev)
    
    return x_train, x_test, y_train, y_test

def windowize_data(data, n_prev):
    n_predictions = len(data) - n_prev
    y = data[n_prev:]
    indices = np.arange(n_prev) + np.arange(n_predictions)[:, None]
    x = data[indices, None]
    return x, y

def plot_trend_data(ax, name, series):
    ax.plot(series.index.date, series)
    ax.set_title("Trend For {}".format(name))
    
def make_design_matrix(arr):
    """Construct a design matrix from a numpy array, converting to a 2-d array
    and including an intercept term."""
    return sm.add_constant(arr.reshape(-1, 1), prepend=False)

def fit_linear_trend(series):
    """using OLS:"""
    """Fit a linear trend to a time series.  Return the fit trend as a numpy array."""
    X = make_design_matrix(np.arange(len(series)) + 1)
    linear_trend_ols = sm.OLS(series.values, X).fit()
    linear_trend = linear_trend_ols.predict(X)
    return linear_trend

def plot_linear_trend(ax, name, series):
    linear_trend = fit_linear_trend(series)
    plot_trend_data(ax, name, series)
    ax.plot(series.index.date, linear_trend)
    
def fit_moving_average_trend(series, window=6):
#    return pd.rolling_mean(series, window, center=True)
    return series.rolling(window, center=True).mean()

def plot_moving_average_trend(ax, name, series, window=6):
    moving_average_trend = fit_moving_average_trend(series, window)
    plot_trend_data(ax, name, series)
    ax.plot(series.index.date, moving_average_trend)
    
def create_monthly_dummies(series):
    month = series.index.month
    # Only take 11 of the 12 months to avoid strict colinearity.
    return pd.get_dummies(month).iloc[:, :11]

def fit_seasonal_trend(series):
    dummies = create_monthly_dummies(series)
    X = sm.add_constant(dummies.values, prepend=False)
    seasonal_model = sm.OLS(series.values, X).fit()
    return seasonal_model.predict(X)

def plot_seasonal_trend(ax, name, series):
    seasons_average_trend = fit_seasonal_trend(series)
    plot_trend_data(ax, name, series, )
    ax.plot(series.index.date, seasons_average_trend, '-')
    
#prepares axes for plotting of raw series & trend, seasonal, and residual components
def plot_shared_yscales(axs, x, ys, titles):
    ymiddles =  [ (y.max()+y.min())/2 for y in ys ]
    yrange = max( (y.max()-y.min())/2 for y in ys )
    for ax, y, title, ymiddle in zip(axs, ys, titles, ymiddles):
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_ylim((ymiddle-yrange, ymiddle+yrange))    

#plots the different components output by statsmodel seasonal decompose        
def plot_seasonal_decomposition(axs, series, sd):
    plot_shared_yscales(axs,
                        series.index,
                        [series, sd.trend, sd.seasonal, sd.resid],
                        ["Raw Series", 
                         "Trend Component $T_t$", 
                         "Seasonal Component $S_t$",
                         "Residual Component $R_t$"])
    
