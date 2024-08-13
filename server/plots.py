import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

def plot_stock(df, stock, columns, show='no', interval='1d'):
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

    df['Volume'] = df['Volume'] / 100000

    # Create a boolean mask for the dates you want to keep
    mask = ~(((df['Datetime'].dt.month == 12) & (df['Datetime'].dt.day.isin([24, 25]))) |
            ((df['Datetime'].dt.month == 2) & (df['Datetime'].dt.day == 19)))

    # Apply the mask to the DataFrame
    df = df[mask]

    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=df['Datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=stock))
    if(show == 'all'):
        for column in columns:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df[column], name=column))
    if (interval == '1h'):
        fig.update_xaxes(
                        rangeslider_visible=True,
                        rangebreaks=[
                # : Below values are bound (not single values), ie. hide x to y
                                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                                    dict(bounds=[16, 9.5], pattern="hour")]) # hide hours outside of 9.30am-4pm
                                
    
    return fig