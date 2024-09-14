import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd

def plot_stock(df, stock, columns, signals=None, show='no', interval='1h'):
    """
    Plots the stock data using Plotly.

    Args:
        df (pandas.DataFrame): The DataFrame containing the stock data.
        stock (str): The name of the stock.
        show (str, optional): Determines which additional data to show on the plot. Defaults to 'no'.
        interval (str, optional): The time interval for the plot. Defaults to '1d'.

    Returns:
        None
    """
    # if 'MCAD' in columns:
        # pass

    df['Volume'] = df['Volume'] / 1000000
   

    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=stock))
    if(show == 'all'):
        for column in columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[column], name=column))
    if interval in ['1m', '1h']:
        fig.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[16, 9.5], pattern="hour")  # hide non-trading hours (16:00 to 09:30)
            ])

    if signals is not None:
        for index, row in signals.iterrows():
            if row['buy_signal']:
                fig.add_trace(go.Scatter(x=index, y=df['Close'][index],
                                        mode='markers',
                                        marker=dict(size=10, color='green'),
                                        name=signal['Signal']))
            if row['sell_signal']:
                fig.add_trace(go.Scatter(x=index, y=df['Close'][index],
                                        mode='markers',
                                        marker=dict(size=10, color='red'),
                                        name=signal['Signal']))
        
    return fig