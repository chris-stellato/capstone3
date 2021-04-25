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
    df.set_index(pd.to_datetime(df[col_name]), drop=False, inplace=True)
#     df.drop(col_name, axis=1, inplace=True)
    return df

def windowize_data(data, n_prev, y_var='y', predict_steps=365):
    n_predictions = len(data) - n_prev
    y = data[y_var].iloc[n_prev:].values
    y_indices = np.arange(predict_steps) + np.arange(len(y) - predict_steps+1)[:, None]
    y = y[y_indices]
    x_indices = np.arange(n_prev) + np.arange(n_predictions- predict_steps+1)[:, None]
    x = data.values[x_indices]
    return x, y

def year_eval_lstm(model, pred_year, master_df):
    df = master_df.copy()
    
    #making desired year last year in sample data
    samp_df = df[:pred_year]

    #trimming sample data to 6000 records total
    samp_df = samp_df.iloc[-6000:]

    # convert datetime column to continuous integer
    samp_df['ds'] = pd.to_datetime(df['ds']).sub(pd.Timestamp(df['ds'].iloc[0])).dt.days
    
    #don't pass in historical y to model
    samp_df.drop('hist_avg_y', axis=1, inplace=True)
    
    # scale entire dataframe except y column 
    scale_df = samp_df.copy()
    for column in scale_df.columns:
      if column != 'y':
        scaler = StandardScaler()
        # print (scale_df[column].values.shape)
        holder = scaler.fit_transform(scale_df[column].values.reshape(-1,1))
        scale_df[column] = holder.reshape(len(scale_df),)
        
    #test size depends if year is leap year or not
    test_size = 365
    if int(pred_year)%4 == 0:
        test_size = 366
        
    #these variables can't be changed without retraining models used in this project
    n_prev = 400 #model was trained on this input shape
    predict_steps = 30 #model was trained on this output shape
    
    #utilizes windowize_data function
    X, y = windowize_data(scale_df, n_prev, 'y', predict_steps)
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    #use model to make predictions and make 0 the lower limit of predictions
    y_pred = model.predict(X_test)
    y_pred[y_pred<0] = 0

    
    #grab predictions at 1-day, 14-days, and 30-days out for each day of the year
    day_1_pred = y_pred[:,:1]          
    day_14_pred = y_pred[:,13:14]
    day_30_pred = y_pred[:,-1:]
    
    #collect and print rmse for 1-day, 14-days, and 30-days prediction sets
    day_1_rmse  = sqrt(mean_squared_error(y_test[:,:1], day_1_pred))
    day_14_rmse = sqrt(mean_squared_error(y_test[:,13:14], day_14_pred))
    day_30_rmse = sqrt(mean_squared_error(y_test[:,-1:], day_30_pred))
    print(f'For prediction year {pred_year}:')
    print(f'1-Day RMSE = {day_1_rmse}')
    print(f'14-Day RMSE = {day_14_rmse}')
    print(f'30-Day RMSE = {day_30_rmse}')
    
    #collect historical average set, calculate rmse over actuals, and compare lift over historical average
    hist_avg_set = df[pred_year]['hist_avg_y']
    hist_avg_rmse = sqrt(mean_squared_error(y_test[:,:1], hist_avg_set))
    
    day_1_lift = 1-(day_1_rmse/hist_avg_rmse)
    day_14_lift = 1-(day_14_rmse/hist_avg_rmse)
    day_30_lift = 1-(day_30_rmse/hist_avg_rmse)
    
    #print lifts over historical average
    print(f'Historical average vs. actuals RMSE: {hist_avg_rmse}')
    print('\n')
    print('Model lift over historical average method')
    print(f'1-Day: {day_1_lift*100}%')
    print(f'14-Day: {day_14_lift*100}%')
    print(f'30-Day: {day_30_lift*100}%')
    
    return day_1_rmse, day_14_rmse, day_30_rmse, hist_avg_rmse, day_1_lift, day_14_lift, day_30_lift




def year_graph_lstm(model, model_name_string, pred_year, master_df):
    df = master_df.copy()
    
    #making desired year last year in sample data
    samp_df = df[:pred_year].copy()

    #trimming sample data to 6000 records total
    samp_df = samp_df.iloc[-6000:]

    # convert datetime column to continuous integer
    samp_df['ds'] = pd.to_datetime(df['ds']).sub(pd.Timestamp(df['ds'].iloc[0])).dt.days
    
    #don't pass in historical y to model
    samp_df.drop('hist_avg_y', axis=1, inplace=True)
    
    # scale entire dataframe except y column 
    scale_df = samp_df.copy()
    for column in scale_df.columns:
      if column != 'y':
        scaler = StandardScaler()
        # print (scale_df[column].values.shape)
        holder = scaler.fit_transform(scale_df[column].values.reshape(-1,1))
        scale_df[column] = holder.reshape(len(scale_df),)
        
    #test size depends if year is leap year or not
    test_size = 365
    if int(pred_year)%4 == 0:
        test_size = 366
        
    #these variables can't be changed without retraining models used in this project
    n_prev = 400 #model was trained on this input shape
    predict_steps = 30 #model was trained on this output shape
    
    #utilizes windowize_data function
    X, y = windowize_data(scale_df, n_prev, 'y', predict_steps)
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    #use model to make predictions and make 0 the lower limit of predictions
    y_pred = model.predict(X_test)
    y_pred[y_pred<0] = 0
#     print(y_pred)

    
    #grab predictions at 1-day, 14-days, and 30-days out for each day of the year
    day_1_pred = y_pred[:,:1]
    day_14_pred = y_pred[:,13:14]
    day_30_pred = y_pred[:,-1:]
#     print(df[pred_year].index[30:])
  
#     plot actual, hist_avg, and 1-day, 14-day, and 30-day predictions
    plt.figure(figsize=(30,10))
    plt.plot(df[pred_year].index, df[pred_year]['y'],label='actual', linewidth=5);
    plt.plot(df[pred_year].index, df[pred_year]['hist_avg_y'],label='historical average', linewidth=3, color='green', alpha=0.8, linestyle='dashed')
    plt.plot(df[pred_year].index[:-30], day_1_pred[30:], label='1-day forecast', linewidth=3, color='purple', alpha=0.3)
    plt.plot(df[pred_year].index[:-14], day_14_pred[14:], label='14-day forecast', linewidth=5, color='red')
    plt.plot(df[pred_year].index, day_30_pred, label='30-day forecast', linewidth=5, color='orange', alpha=0.4)
    #formatting
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Streamflow CFS', fontsize=30)
    plt.legend(prop={"size":30})
    plt.savefig(f'graphs/{model_name_string}_yearly_{pred_year}.jpg')
    plt.show()
    
    pass










def split_and_windowize(data, n_prev, n_test=365):
    n_predictions = len(data) - 2*n_prev
    
    n_train = n_predictions - n_test   
    
    x_train, y_train = windowize_data(data[:n_train], n_prev)
    x_test, y_test = windowize_data(data[n_train:], n_prev)
    
    return x_train, x_test, y_train, y_test

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
    
