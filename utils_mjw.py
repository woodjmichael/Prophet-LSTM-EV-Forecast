""" utils.py
Generally useufl utility functions
"""

import math
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plotly.subplots import make_subplots  

def order_of_magnitude(x:float)->int:
    """ Calculate order of magnitude

    Args:
        x (float): real number

    Returns:
        int: order of magnitude
    """
    
    return math.floor(math.log(x, 10))


def upsample_df(df,periods,freq,method='ffill'):
    df2 = pd.DataFrame([],index=pd.date_range(df.index[0],periods=periods,freq=freq))
    for col in df.columns:
        df2.loc[df.index,col] = df[col].values
    df2 = df2.fillna(method=method)
    return df2


def shift_and_wraparound(ds:pd.Series,i:int):
    """ Shift data and wraparound the end
    pos = shift left/up
    neg = shift right/down
    """
    return list(ds.values[i:]) + list(ds.values[:i])
    
    
def calc_monthly_peaks(ds:pd.Series,peak_begin:int,peak_end:int) -> pd.Series:
    """ Calculate max power (1 hr non-moving average) of the month

    Args:
        ds (pd.Series): timeseries to calculate on
        peak_begin (int): start of peak TOU period (inclusive)
        peak_end (int): end of peak TOU period (exclusive)

    Returns:
        pd.Series: _description_
    """
    ds = ds.resample('1h').mean()
    hours = list(range(peak_begin,peak_end))
    peak,ipeak = [],[]
    for year in ds.index.year.unique():
        ds_year = ds[ds.index.year==year]
        for month in ds_year.index.month.unique():
            ds_month = ds_year[ds_year.index.month==month]
            peak.append( ds_month[[True if h in hours else False for h in ds_month.index.hour]].max().round(1) )
            ipeak.append(ds_month[[True if h in hours else False for h in ds_month.index.hour]].idxmax())

    results = pd.DataFrame({f'{ds.name} kw':peak,
                            f'{ds.name} t':ipeak,})
    #results.index = list(range(1,13))
    return results


def plot_daily(ds:pd.Series,
               alpha:float=0.1,
               title=None):
    """ Plot a series with days super-imposed on each other. Index should be complete (no gaps)
    for this to work right. Trims any remaining data after an integer number of days.

    Args:
        ds (pd.Series): pandas series to plot
        interval_min (int): timeseries data interval
        alpha (float, optional): transparency of plot lines, defaults to 0.1
        begin_on_monday (bool, optional): have the first day on the plot be monday, defaults to True
    """
    interval_min = int(ds.index.to_series().diff().mean().seconds/60)
    dpd = int(24*60/interval_min) # data per day
    ds2 = ds.copy(deep=True)
    dt_start = ds.index.min()
    if dt_start != dt_start.floor('1d'):
        dt_start = dt_start.floor('1d') + pd.Timedelta(hours=24)
    ds2 = ds2[dt_start:]
    n_days = len(ds2)//(dpd)
    ds2 = ds2.iloc[:int(n_days*dpd)]
    
    t = [x*(24/dpd) for x in range(int(24*(dpd/24)))] # hours
    plt.plot(t, ds2.values.reshape(n_days,dpd).T,alpha=alpha)
    plt.ylabel(ds.name)
    plt.xlabel('Hours from 0:00')
    plt.title(title)
    plt.show()


def plot_weekly(ds,
                title:str=None,
                ylabel:str=None,
                alpha:float=0.1,
                begin_on_monday:bool=True,
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
        colors (list, optional): list of colors strings
    """
    if not isinstance(ds,(list,tuple)):
        ylabel = ds.name
        ds = [ds]
        
    interval_min = int(ds[0].index.to_series().diff().mean().seconds/60)
    dpd = int(24*60/interval_min) # data per day
    plt.figure(figsize=(10,5))
    t = [x/dpd for x in range(7*dpd)] # days    

    for ds2,color in zip(ds,colors):
        ds2 = ds2.copy(deep=True)
        dt_start = ds2.index.min()
        if dt_start != dt_start.floor('1d'):
            dt_start = dt_start.floor('1d') + pd.Timedelta(hours=24)
        if begin_on_monday and (dt_start.weekday() != 0):
            days = 7 - dt_start.weekday()
            dt_start += pd.Timedelta(hours=24*days)
        ds2 = ds2[dt_start:]
        n_weeks = len(ds2)//(7*dpd)
        ds2 = ds2.iloc[:int(n_weeks*7*dpd)]
        if len(ds)>1:
            plt.plot(t,ds2.values.reshape(n_weeks,7*dpd).T,color,alpha=alpha)
        else:
            plt.plot(t,ds2.values.reshape(n_weeks,7*dpd).T,alpha=alpha)
    plt.ylabel(ylabel)
    plt.xlabel('Days from Monday 0:00')
    plt.title(title)
    if len(ds)>1:
        legend_items = []
        for s,color in zip(ds,colors):
            legend_items.append(mpatches.Patch(color=color, label=s.name))
        plt.legend(handles=legend_items)
    plt.show()    


def plotly_stacked_old(_df:pd.DataFrame,
                   solar='solar',
                   solar_name='Solar',
                   load='load',
                   load_name='Load',
                   batt='batt',
                   discharge='discharge',
                   discharge_name='Battery Discharge',
                   charge='charge',
                   load_charge_name='Load + Charge',
                   utility='utility',
                   utility_name='Site Load',        
                   soc='soc',
                   soc_name='SOC (right axis)',
                   soe='soe',
                   soe_name='SOE (right axis)',
                   threshold=None,
                   threshold_h=None,
                   threshold_name='Threshold',
                   ylim=None,
                   size=None,
                   title=None,
                   fig=None,
                   units_power='kW',
                   units_energy='kWh',
                   round_digits=1):
    """ Make plotly graph with some data stacked in area-fill style
    """
    
    df = _df.copy(deep=True) # we'll be modifying this
    #export='export'
    loadPlusCharge = 'loadPlusCharge'

    if charge not in df.columns:
        df[charge] =    [max(0,-1*x) for x in df[batt]]
        df[discharge] =    [max(0,x) for x in df[batt]]    
    df[loadPlusCharge] = df[load]+df[charge]
    #df[export] = df[solar] - df[loadPlusCharge] #[-1*min(0,x) for x in df[utility]]
    df[utility] = [max(0,x) for x in df[utility]]
    df[solar] = df[solar]#df[load] - df[utility]
    
    if threshold is not None:
        df['Threshold'] = [threshold if x in threshold_h else pd.NA for x in df.index.hour]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            name=utility_name,
            x=df.index, y=df[utility].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0, color='darkseagreen'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=solar_name,
            x=df.index, y=df[solar].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0,color='gold'),
        ),
        secondary_y=False,
    )
    # fig.add_trace(
    #     go.Scatter(
    #         name='Export',
    #         x=df.index, y=df[export],
    #         mode='lines',
    #         stackgroup='one',
    #         line=dict(width=0,color='khaki'),
    #     ),
    #     secondary_y=False,
    # )
    fig.add_trace(
        go.Scatter(
            name=discharge_name,
            x=df.index, y=df[discharge].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0, color='dodgerblue'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=load_charge_name,
            x=df.index, y=df[loadPlusCharge].round(round_digits),
            mode='lines',
            #stackgroup='one',
            line=dict(width=1.5, dash='dash', color='dodgerblue'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=load_name,
            x=df.index, y=df[load].round(round_digits),
            mode='lines',
            #stackgroup='one',
            line=dict(width=1.5, color='indigo'),
        ),
        secondary_y=False,
    )
    if threshold is not None:
        fig.add_trace(
            go.Scatter(
                name=threshold_name,
                x=df.index, y=df['Threshold'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=1.5, color='mediumvioletred'),
            ),
            secondary_y=False,
        )
    if soc in df.columns:
        fig.add_trace(
            go.Scatter(
                name=soc_name,
                x=df.index, y=(df[soc]*100).round(round_digits),
                mode='lines',
                line=dict(width=1, dash='dot',color='lightcoral'),
            ),
            secondary_y=True,
        ) 
    elif soe in df.columns:
        fig.add_trace(
            go.Scatter(
                name=soe_name,
                x=df.index, y=df[soe].round(round_digits),
                mode='lines',
                line=dict(width=1, dash='dot',color='lightcoral'),
            ),
            secondary_y=True,
        )
           
    fig.update_traces(hovertemplate=None)#, xhoverformat='%{4}f')
    fig.update_layout(hovermode='x',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h'),
                      legend_traceorder='reversed',
                      title=title,)
    if size is not None:
        fig.update_layout(width=size[0],height=size[1])
    
    if soc in df.columns:
        fig.update_yaxes(title_text='%',range=(0, 100),secondary_y=True)
    elif soe in df.columns:
        fig.update_yaxes(title_text=units_energy,range=(0, df[soe].max()),secondary_y=True)

    else:
        fig.update_yaxes(title_text=units_power, secondary_y=False)

    if ylim is None:
        ymax = max(df[loadPlusCharge].max(),df[utility].max(),df[solar].max())
        fig.update_yaxes(range=(-.025*ymax, 1.1*ymax),secondary_y=False)
    else:
        fig.update_yaxes(range=(ylim[0], ylim[1]),secondary_y=False)
        
    fig.show()


def plotly_stacked(_df:pd.DataFrame,
                   solar='solar',
                   solar_name='Solar',
                   load='load',
                   load_name='Load',
                   batt='batt',
                   discharge='discharge',
                   discharge_name='Battery Discharge',
                   charge='charge',
                   load_charge_name='Load + Charge',
                   utility='utility',
                   utility_name='Site Load',        
                   soc='soc',
                   soc_name='SOC (right axis)',
                   soe='soe',
                   soe_name='SOE (right axis)',
                   threshold0=None,
                   threshold0_h=None,
                   threshold1=None,
                   threshold1_h=None,
                   threshold2=None,
                   threshold2_h=None,
                   ylim=None,
                   size=None,
                   title=None,
                   fig=None,
                   units_power='kW',
                   units_energy='kWh',
                   round_digits=1,
                   upsample_min=None):
    """ Make plotly graph with some data stacked in area-fill style
    """
    
    df = _df.copy(deep=True) # we'll be modifying this
    
    # upsample for more accurate viewing
    if upsample_min is not None:
        freq_min = int(df.index.to_series().diff().dropna().mean().seconds/60)
        new_length = len(df) * (freq_min / upsample_min)
        df = upsample_df(df,freq=f'{upsample_min}min',periods=new_length)
        
    # threshold vectors
    if threshold0 is not None:
        df['threshold0'] = [threshold0 if x in threshold0_h else pd.NA for x in df.index.hour]
    if threshold1 is not None:
        df['threshold1'] = [threshold1 if x in threshold1_h else pd.NA for x in df.index.hour]
    if threshold2 is not None:
        df['threshold2'] = [threshold2 if x in threshold2_h else pd.NA for x in df.index.hour]
    
    #export='export'
    loadPlusCharge = 'loadPlusCharge'

    if charge not in df.columns:
        df[charge] =    [max(0,-1*x) for x in df[batt]]
        df[discharge] =    [max(0,x) for x in df[batt]]    
    df[loadPlusCharge] = df[load]+df[charge]
    #df[export] = df[solar] - df[loadPlusCharge] #[-1*min(0,x) for x in df[utility]]
    df[utility] = [max(0,x) for x in df[utility]]
    df[solar] = df[solar]#df[load] - df[utility]
    
    # plot
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            name=utility_name,
            x=df.index, y=df[utility].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0, color='darkseagreen'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=solar_name,
            x=df.index, y=df[solar].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0,color='gold'),
        ),
        secondary_y=False,
    )
    # fig.add_trace(
    #     go.Scatter(
    #         name='Export',
    #         x=df.index, y=df[export],
    #         mode='lines',
    #         stackgroup='one',
    #         line=dict(width=0,color='khaki'),
    #     ),
    #     secondary_y=False,
    # )
    fig.add_trace(
        go.Scatter(
            name=discharge_name,
            x=df.index, y=df[discharge].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0, color='dodgerblue'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=load_charge_name,
            x=df.index, y=df[loadPlusCharge].round(round_digits),
            mode='lines',
            #stackgroup='one',
            line=dict(width=1.5, dash='dash', color='dodgerblue'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=load_name,
            x=df.index, y=df[load].round(round_digits),
            mode='lines',
            #stackgroup='one',
            line=dict(width=1.5, color='indigo'),
        ),
        secondary_y=False,
    )
    if threshold0 is not None:
        if threshold1 is None:
            name = 'Threshold'
        else:
            name = 'Threshold 0'
        fig.add_trace(
            go.Scatter(
                name=name,
                x=df.index, y=df['threshold0'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=1.5, color='palevioletred'),
            ),
            secondary_y=False,
        )
    if threshold1 is not None:
        fig.add_trace(
            go.Scatter(
                name='Threshold 1',
                x=df.index, y=df['threshold1'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=1.5, color='mediumvioletred'),
            ),
            secondary_y=False,
        )
    if threshold2 is not None:
        fig.add_trace(
            go.Scatter(
                name='Threshold 2',
                x=df.index, y=df['threshold2'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=1.5, color='crimson'),
            ),
            secondary_y=False,
        )        
    if soc in df.columns:
        fig.add_trace(
            go.Scatter(
                name=soc_name,
                x=df.index, y=(df[soc]*100).round(round_digits),
                mode='lines',
                line=dict(width=1, dash='dot',color='coral'),
            ),
            secondary_y=True,
        ) 
    elif soe in df.columns:
        fig.add_trace(
            go.Scatter(
                name=soe_name,
                x=df.index, y=df[soe].round(round_digits),
                mode='lines',
                line=dict(width=1, dash='dot',color='coral'),
            ),
            secondary_y=True,
        )
           
    fig.update_traces(hovertemplate=None)#, xhoverformat='%{4}f')
    fig.update_layout(hovermode='x',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h'),
                      legend_traceorder='reversed',
                      title=title,)
    if size is not None:
        fig.update_layout(width=size[0],height=size[1])
    
    if soc in df.columns:
        fig.update_yaxes(title_text='%',range=(0, 100),secondary_y=True)
    elif soe in df.columns:
        fig.update_yaxes(title_text=units_energy,range=(0, df[soe].max()),secondary_y=True)

    else:
        fig.update_yaxes(title_text=units_power, secondary_y=False)

    if ylim is None:
        ymax = max(df[loadPlusCharge].max(),df[utility].max(),df[solar].max())
        fig.update_yaxes(range=(-.025*ymax, 1.1*ymax),secondary_y=False)
    else:
        fig.update_yaxes(range=(ylim[0], ylim[1]),secondary_y=False)
        
    fig.show()