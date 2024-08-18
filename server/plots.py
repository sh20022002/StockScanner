import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    if 'MCAD' in columns:
        pass

    df['Volume'] = df['Volume'] / 100000
    if interval == '1m':
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

def plot_stock_(df, stock, columns, show='no', interval='1d'):

    df['Volume'] = df['Volume'] / 100000

    # Create a boolean mask for the dates you want to keep
    mask = ~(((df['Datetime'].dt.month == 12) & (df['Datetime'].dt.day.isin([24, 25]))) |
            ((df['Datetime'].dt.month == 2) & (df['Datetime'].dt.day == 19)))

    # Apply the mask to the DataFrame
    df = df[mask]

    # Ensure 'Datetime' is set as the index
    df.set_index('Datetime', inplace=True)

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Stock Price', 'time'),
                        vertical_spacing=0.1)

    # Add stock price plot
    fig.add_trace(go.Candlestick(x=df['Datetime'],
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name=stock), row=1, col=1)

    # Add index plot (assuming 'Index' is the column name for the index values)
    # fig.add_trace(go.Scatter(x=df.index, y=df['Index'], name='Index Values'), row=2, col=1)

    if(show == 'all'):
        for column in columns:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df[column], name=column), row=1, col=1)

    if (interval == '1h'):
        fig.update_xaxes(
                        rangeslider_visible=True,
                        rangebreaks=[
                # : Below values are bound (not single values), ie. hide x to y
                                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                                    dict(bounds=[16, 9.5], pattern="hour")]) # hide hours outside of 9.30am-4pm

    # Update layout
    fig.update_layout(title=f'{stock} Stock Data',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      showlegend=False)

    return fig