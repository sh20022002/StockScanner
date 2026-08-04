"""Plotly chart builders for the dashboard."""
import plotly.graph_objects as go
import polars as pl


# Price/volume columns are drawn by the candlestick itself — overlaying them
# again just clutters the chart.
_NOT_OVERLAYS = {'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'}


def plot_stock(df: pl.DataFrame, stock: str, columns=None, signals=None,
               show: str = 'no', interval: str = '1h'):
    """
    Build a candlestick chart with optional indicator overlays and signal markers.

    Args:
        df (pl.DataFrame): Stock data. Must carry a 'Datetime' COLUMN — this is a
            Polars frame, so there is no index to fall back on.
        stock (str): Symbol, used for the trace name and title.
        columns (list): Indicator columns to overlay when show == 'all'.
        signals (pl.DataFrame): Optional frame with Buy_Signal/Sell_Signal/Datetime.
        show (str): 'all' draws the overlays, anything else skips them.
        interval (str): Timeframe, used to hide non-trading hours on intraday charts.

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=stock,
    ))

    if show == 'all':
        # Was `x=df.index`, which raises AttributeError on a Polars frame and so
        # broke the chart for every caller that asked for overlays.
        for column in (columns or []):
            if column in _NOT_OVERLAYS or column not in df.columns:
                continue
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df[column], name=column))

    if interval in ['1m', '2m', '5m', '15m', '30m', '1h']:
        fig.update_xaxes(rangebreaks=[
            dict(bounds=["sat", "mon"]),            # hide weekends
            dict(bounds=[16, 9.5], pattern="hour"), # hide 16:00-09:30
        ])

    if signals is not None:
        if 'Buy_Signal' in signals.columns and 'Sell_Signal' in signals.columns:
            for flag, colour, symbol, label in (
                ('Buy_Signal',  'green', 'triangle-up',   'Buy Signals'),
                ('Sell_Signal', 'red',   'triangle-down', 'Sell Signals'),
            ):
                when = signals.filter(pl.col(flag))['Datetime']
                if len(when) == 0:
                    continue
                # Join back to prices so a marker cannot drift onto the wrong bar.
                at = df.filter(pl.col('Datetime').is_in(when)).select(['Datetime', 'Close'])
                fig.add_trace(go.Scatter(
                    x=at['Datetime'], y=at['Close'],
                    mode='markers',
                    marker=dict(size=10, color=colour, symbol=symbol),
                    name=label,
                ))
        else:
            print("Signals DataFrame must contain 'Buy_Signal' and 'Sell_Signal' columns.")

    fig.update_layout(
        title=f"{stock} Stock Price {interval} Interval",
        yaxis_title="Price",
        xaxis_title="Date",
        legend_title="Legend",
        xaxis_rangeslider_visible=False,
    )

    return fig
