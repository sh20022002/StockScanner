import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
import polars as pl

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
    # Ensure that 'Buy_Signal' and 'Sell_Signal' columns are in the signals DataFrame
        if 'Buy_Signal' in signals.columns and 'Sell_Signal' in signals.columns:
            # Extract indices where Buy_Signal is True
            buy_indices = signals[signals['Buy_Signal'] == True].index
            buy_prices = df.loc[buy_indices, 'Close']

            # Add green markers for buy signals
            fig.add_trace(go.Scatter(
                x=buy_indices,
                y=buy_prices,
                mode='markers',
                marker=dict(size=10, color='green', symbol='triangle-up'),
                name='Buy Signals'
            ))

            # Extract indices where Sell_Signal is True
            sell_indices = signals[signals['Sell_Signal'] == True].index
            sell_prices = df.loc[sell_indices, 'Close']

            # Add red markers for sell signals
            fig.add_trace(go.Scatter(
                x=sell_indices,
                y=sell_prices,
                mode='markers',
                marker=dict(size=10, color='red', symbol='triangle-down'),
                name='Sell Signals'
            ))
        else:
            print("Signals DataFrame must contain 'Buy_Signal' and 'Sell_Signal' columns.")

    # Update layout for better visualization
    fig.update_layout(
        title=f"{stock} Stock Price {interval} Interval",
        yaxis_title="Price",
        xaxis_title="Date",
        legend_title="Legend",
        xaxis_rangeslider_visible=False  # Hide range slider if not needed
    )

    return fig

def plot_signals(self, df: pl.DataFrame, signals_df: pl.DataFrame, symbol: str):
        """
        Plots the closing price along with buy and sell signals.

        Args:
            df (pl.DataFrame): The stock data.
            signals_df (pl.DataFrame): The buy and sell signals.
            symbol (str): Stock symbol.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(df['DateTime'], df['Close'], label='Close Price', alpha=0.5)

        # Buy signals
        buy_signals = signals_df.filter(pl.col('Buy_Signal') == True)
        plt.scatter(buy_signals['DateTime'].to_list(), buy_signals['Close'].to_list(), marker='^', color='green', label='Buy Signal', alpha=1)

        # Sell signals
        sell_signals = signals_df.filter(pl.col('Sell_Signal') == True)
        plt.scatter(sell_signals['DateTime'].to_list(), sell_signals['Close'].to_list(), marker='v', color='red', label='Sell Signal', alpha=1)

        plt.title(f'{symbol} Buy/Sell Signals')
        plt.xlabel('DateTime')
        plt.ylabel('Price')
        plt.legend()
        plt.show()