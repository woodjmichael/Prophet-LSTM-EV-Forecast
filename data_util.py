import pandas as pd
import matplotlib.pyplot as plt
from pandas import concat
from sklearn.preprocessing import MinMaxScaler


# convert series to supervised learning
def series_to_supervised(df, out_col, n_in=1, n_out=1, lag=0, dropnan=True):
    # frame as supervised learning
    n_vars = 1 if type(df) is pd.Series else df.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t+l, t+l+1, ... t+l+n)
    df_out = df[out_col]
    n_vars = 1 if type(df_out) is pd.Series else df_out.shape[1]
    for i in range(lag, n_out + lag):
        cols.append(df_out.shift(-i))
        names += [('out%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# transform series into train and test sets for supervised learning
def prepare_data(data, data_opt, dropnan=True):
    # extract raw values
    reframed = series_to_supervised(df=data, out_col=data_opt['out_col'], n_in=data_opt['n_back'],
                                    n_out=data_opt['n_timesteps'], lag=data_opt['lag'], dropnan=dropnan)
    input_col = [col for col in reframed.columns if '-' in col]
    output_col = [col for col in reframed.columns if '+' in col]
    # output_col = list(set(reframed.columns) - set(input_col))

    # split into train and test sets
    values = reframed.values
    n_train_hours = int(data_opt['tr_per'] * reframed.shape[0])
    train = reframed.iloc[:n_train_hours, :]
    test = reframed.iloc[n_train_hours:, :]

    # time_tr_y = reframed.index[:n_train_hours]
    # time_tst_y = reframed.index[n_train_hours:]
    # split into input and outputs
    train_X, train_y = train.loc[:, input_col].values, train.loc[:, output_col].values
    test_X, test_y = test.loc[:, input_col].values, test.loc[:, output_col]

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    train_X = scaler_X.fit_transform(train_X)
    test_X = scaler_X.transform(test_X)

    train_y = scaler_y.fit_transform(train_y)

    # test_y = scaler_y.transform(test_y)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train.shape[0], data_opt['n_back'], data_opt['n_features']))
    test_X = test_X.reshape((test.shape[0], data_opt['n_back'], data_opt['n_features']))

    return train_X, train_y, test_X, test[output_col], scaler_X, scaler_y


# transform series into train and test sets for supervised learning
def prepare_data_train(data, data_opt):
    # extract raw values
    reframed = series_to_supervised(df=data, out_col=data_opt['out_col'], n_in=data_opt['n_back'],
                                    n_out=data_opt['n_timesteps'], lag=data_opt['lag'])
    input_col = [col for col in reframed.columns if '-' in col]
    output_col = [col for col in reframed.columns if '+' in col]

    df_X, df_y = reframed.loc[:, input_col].values, reframed.loc[:, output_col].values

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    df_X = scaler_X.fit_transform(df_X)

    df_y = scaler_y.fit_transform(df_y)

    df_X = df_X.reshape((reframed.shape[0], data_opt['n_back'], data_opt['n_features']))
    # test_X = test_X.reshape((test.shape[0], data_opt['n_back'], data_opt['n_features']))

    return df_X, df_y, scaler_X, scaler_y

# transform series into train and test sets for supervised learning
def prepare_data_test(data, scaler_X, data_opt, dropnan=False):
    # extract raw values
    reframed = series_to_supervised(df=data, out_col=data_opt['out_col'], n_in=data_opt['n_back'],
                                    n_out=data_opt['n_timesteps'], lag=data_opt['lag'], dropnan=dropnan)
    input_col = [col for col in reframed.columns if '-' in col]
    output_col = [col for col in reframed.columns if '+' in col]
    # output_col = list(set(reframed.columns) - set(input_col))
    reframed.dropna(subset=input_col)
    df_X, dft_y = reframed.loc[:, input_col].values, reframed.loc[:, output_col]

    df_X = scaler_X.transform(df_X)

    # test_y = scaler_y.transform(test_y)
    # reshape input to be 3D [samples, timesteps, features]
    df_X = df_X.reshape((reframed.shape[0], data_opt['n_back'], data_opt['n_features']))

    return df_X, reframed[output_col]

def plot_weekly(ds,
                title:str=None,
                ylabel:str=None,
                alpha:float=0.1,
                begin_on_monday:bool=True,
                n_days:int=7,
                colors:list=['indigo','gold','magenta']):
    """ Plot a series with weeks super-imposed on each other. Index should be complete (no gaps)
    for this to work right. Trims any remaining data after an integer number of weeks.

    Args:
        ds (pd.Series or list): pandas series(es) to plot
        title (str, optional): title
        ylabel (str, optional): what to call the y-axis        
        interval_min (int): timeseries data interval
        alpha (float, optional): transparency of plot lines, defaults to 0.1
        begin_on_monday (bool, optional): have the first day on the plot be monday, defaults to True
        n_days (int,optional): if only doing weekdays use 5, weekends use 2
        colors (list, optional): list of colors strings
    """
    if not isinstance(ds,(list,tuple)):
        ylabel = ds.name
        ds = [ds]
        
    interval_min = int(ds[0].index.to_series().diff().mean().seconds/60)
    dpd = int(24*60/interval_min) # data per day
    plt.figure(figsize=(10,5))
    t = [x/dpd for x in range(n_days*dpd)] # days    

    for ds2,color in zip(ds,colors):
        ds2 = ds2.copy(deep=True)
        dt_start = ds2.index.min()
        if dt_start != dt_start.floor('1d'):
            dt_start = dt_start.floor('1d') + pd.Timedelta(hours=24)
        if begin_on_monday and (dt_start.weekday() != 0):
            days = n_days - dt_start.weekday()
            dt_start += pd.Timedelta(hours=24*days)
        ds2 = ds2[dt_start:]
        n_weeks = len(ds2)//(n_days*dpd)
        ds2 = ds2.iloc[:int(n_weeks*n_days*dpd)]
        if len(ds)>1:
            plt.plot(t,ds2.values.reshape(n_weeks,n_days*dpd).T,color,alpha=alpha)
        else:
            plt.plot(t,ds2.values.reshape(n_weeks,n_days*dpd).T,alpha=alpha)
    plt.ylabel(ylabel)
    if begin_on_monday:
        plt.xlabel('Days from Monday 0:00')
    plt.title(title)
    if len(ds)>1:
        legend_items = []
        for s,color in zip(ds,colors):
            legend_items.append(mpatches.Patch(color=color, label=s.name))
        plt.legend(handles=legend_items)
    plt.show() 