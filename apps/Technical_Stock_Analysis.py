import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gradio as gr
import io
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import warnings
import timesfm
from prophet import Prophet

class StockDataFetcher:
    """Handles fetching and preprocessing stock data"""
    
    @staticmethod
    def fetch_stock_data(ticker, start_date, end_date):
        """Fetch and preprocess stock data"""
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Handle MultiIndex columns if present
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(level=1)
            
        # Standardize column names
        stock_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        
        return stock_data

# Function for TimesFM forecasting
def timesfm_forecast(ticker, start_date, end_date):
    try:
        # Fetch historical data using the StockDataFetcher class
        stock_data = StockDataFetcher.fetch_stock_data(ticker, start_date, end_date)

        # Reset index to have 'Date' as a column
        stock_data.reset_index(inplace=True)

        # Select relevant columns and rename them
        df = stock_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

        # Ensure the dates are in datetime format
        df['ds'] = pd.to_datetime(df['ds'])

        # Add a unique identifier for the time series
        df['unique_id'] = ticker

        # Initialize the TimesFM model
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="pytorch",
                per_core_batch_size=32,
                horizon_len=30,  # Predicting the next 30 days
                input_patch_len=32,
                output_patch_len=128,
                num_layers=50,
                model_dims=1280,
                use_positional_embedding=False,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

        # Forecast using the prepared DataFrame
        forecast_df = tfm.forecast_on_df(
            inputs=df,
            freq="D",  # Daily frequency
            value_name="y",
            num_jobs=-1,
        )

        # Ensure forecast_df has the correct columns
        forecast_df.rename(columns={"timesfm": "forecast"}, inplace=True)

        # Create an interactive plot with Plotly
        fig = go.Figure()

        # Add Actual Prices Line
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], 
                                mode="lines", name="Actual Prices", 
                                line=dict(color="#00FFFF", width=2)))  # Brighter cyan

        # Add Forecasted Prices Line
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["forecast"], 
                                mode="lines", name="Forecasted Prices", 
                                line=dict(color="#FF00FF", width=2, dash="dash")))  # Brighter magenta

        # Layout Customization
        fig.update_layout(
            title=f"{ticker} Stock Price Forecast (TimesFM)",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",  # Dark Theme
            hovermode="x unified",  # Show all values on hover
            legend=dict(bgcolor="rgba(0,0,0,0.8)", bordercolor="white", borderwidth=1),
            plot_bgcolor="#111111",  # Slightly lighter than black for contrast
            paper_bgcolor="#111111",
            font=dict(color="white", size=12),
            margin=dict(l=40, r=40, t=50, b=40),
        )

        # Add grid lines for better readability
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

        return fig  # Return the Plotly figure for Gradio
    
    except Exception as e:
        return f"Error: {str(e)}"

# Function for Prophet forecasting
def prophet_forecast(ticker, start_date, end_date):
    try:
        # Download stock market data using the StockDataFetcher class
        df = StockDataFetcher.fetch_stock_data(ticker, start_date, end_date)
        
        # Reset the index to get 'Date' back as a column
        df_plot = df.reset_index()
        
        # Prepare the data for Prophet
        df1 = df_plot[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Fit the model
        m = Prophet()
        m.fit(df1)
        
        # Create future dataframe and make predictions
        future = m.make_future_dataframe(periods=30, freq='D')
        forecast = m.predict(future)
        
        # Plotting stock closing prices with trend
        fig1 = go.Figure()
        
        # Add actual closing prices
        fig1.add_trace(go.Scatter(
            x=df1['ds'], 
            y=df1['y'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#36D7B7', width=2)
        ))
        
        # Add trend component
        fig1.add_trace(go.Scatter(
            x=forecast['ds'], 
            y=forecast['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        fig1.update_layout(
            title=f'{ticker} Price and Trend',
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white', size=12),
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(bgcolor="rgba(0,0,0,0.8)", bordercolor="white", borderwidth=1)
        )
        
        # Plotting forecast with confidence interval
        forecast_40 = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(40)
        fig2 = go.Figure()
        
        # Add forecast line
        fig2.add_trace(go.Scatter(
            x=forecast_40['ds'],
            y=forecast_40['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        # Add confidence interval
        fig2.add_trace(go.Scatter(
            x=forecast_40["ds"].tolist() + forecast_40["ds"].tolist()[::-1],
            y=forecast_40["yhat_upper"].tolist() + forecast_40["yhat_lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(78, 205, 196, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval"
        ))
        
        fig2.update_layout(
            title=f'{ticker} 30 Days Forecast (Prophet)',
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white', size=12),
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(bgcolor="rgba(0,0,0,0.8)", bordercolor="white", borderwidth=1)
        )
        
        # Create components figure
        components_fig = go.Figure()
        
        # Add components if they exist in the forecast
        if 'yearly' in forecast.columns:
            yearly_pattern = forecast.iloc[-365:] if len(forecast) > 365 else forecast
            components_fig.add_trace(go.Scatter(
                x=yearly_pattern['ds'], 
                y=yearly_pattern['yearly'],
                mode='lines',
                name='Yearly Pattern',
                line=dict(color='#4ECDC4', width=2)
            ))
        
        
        components_fig.update_layout(
            title=f'{ticker} Forecast Components',
            xaxis_title='Date',
            yaxis_title='Value',
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white', size=12),
            legend=dict(bgcolor="rgba(0,0,0,0.8)", bordercolor="white", borderwidth=1),
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        )
        
        # For backwards compatibility, still create the matplotlib figure
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='#111111')
            
            plt.rcParams.update({
                'text.color': 'white',
                'axes.labelcolor': 'white',
                'axes.edgecolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'grid.color': 'gray',
                'figure.facecolor': '#111111',
                'axes.facecolor': '#111111',
                'savefig.facecolor': '#111111',
            })
            
            m.plot_components(forecast, ax=ax)
            
            for ax in plt.gcf().get_axes():
                ax.set_facecolor('#111111')
                for spine in ax.spines.values():
                    spine.set_color('white')
                ax.tick_params(colors='white')
                ax.title.set_color('white')
                for line in ax.get_lines():
                    if line.get_color() == 'b':
                        line.set_color('#C678DD')
                    else:
                        line.set_color('#FF6B6B')
                
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor='#111111')
            buf.seek(0)
            plt.close(fig)
            
            img = Image.open(buf)
            
            return fig1, fig2, components_fig
        except Exception as e:
            print(f"Error with Matplotlib components: {e}")
            return fig1, fig2, components_fig
    
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}", None
    
# Functions for technical analysis
def calculate_sma(df, window):
    return df['Close'].rolling(window=window).mean()

def calculate_ema(df, window):
    return df['Close'].ewm(span=window, adjust=False).mean()

def calculate_macd(df):
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_rsi(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(df):
    middle_bb = df['Close'].rolling(window=20).mean()
    upper_bb = middle_bb + 2 * df['Close'].rolling(window=20).std()
    lower_bb = middle_bb - 2 * df['Close'].rolling(window=20).std()
    return middle_bb, upper_bb, lower_bb

def calculate_stochastic_oscillator(df):
    lowest_low = df['Low'].rolling(window=14).min()
    highest_high = df['High'].rolling(window=14).max()
    slowk = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    slowd = slowk.rolling(window=3).mean()
    return slowk, slowd

def calculate_cmf(df, window=20):
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    cmf = mfv.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    return cmf

def calculate_cci(df, window=20):
    """Calculate Commodity Channel Index (CCI)."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma = typical_price.rolling(window=window).mean()
    mean_deviation = (typical_price - sma).abs().rolling(window=window).mean()
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci

import numpy as np
import pandas as pd

def generate_trading_signals(df):
    # Calculate various indicators
    df['SMA_30'] = calculate_sma(df, 30)
    df['SMA_100'] = calculate_sma(df, 100)
    df['EMA_12'] = calculate_ema(df, 12)
    df['EMA_26'] = calculate_ema(df, 26)
    df['RSI'] = calculate_rsi(df)
    df['MiddleBB'], df['UpperBB'], df['LowerBB'] = calculate_bollinger_bands(df)
    df['SlowK'], df['SlowD'] = calculate_stochastic_oscillator(df)
    df['CMF'] = calculate_cmf(df)
    df['CCI'] = calculate_cci(df)

    # Ultra-strict SMA Signal - Require at least 3% difference
    df['SMA_Signal'] = np.where((df['SMA_30'] > df['SMA_100']) & 
                                 ((df['SMA_30'] - df['SMA_100']) / df['SMA_100'] > 0.03), 1, 0)
    
    macd, signal = calculate_macd(df)
    
    # Ultra-strict MACD Signal - Require a difference of at least 1.0
    df['MACD_Signal'] = np.select([
        (macd > signal) & (macd.shift(1) <= signal.shift(1)) & ((macd - signal) > 1.0),
        (macd < signal) & (macd.shift(1) >= signal.shift(1)) & ((signal - macd) > 1.0)
    ], [1, -1], default=0)
    
    # Ultra-strict RSI Signal - Extreme thresholds
    df['RSI_Signal'] = np.where(df['RSI'] < 15, 1, 0)
    df['RSI_Signal'] = np.where(df['RSI'] > 90, -1, df['RSI_Signal'])
    
    # Ultra-strict Bollinger Bands Signal - Require extreme deviations
    df['BB_Signal'] = np.where(df['Close'] < df['LowerBB'] * 0.97, 1, 0)
    df['BB_Signal'] = np.where(df['Close'] > df['UpperBB'] * 1.03, -1, df['BB_Signal'])
    
    # Ultra-strict Stochastic Signal - Extreme overbought/oversold conditions
    df['Stochastic_Signal'] = np.where((df['SlowK'] < 10) & (df['SlowD'] < 10), 1, 0)
    df['Stochastic_Signal'] = np.where((df['SlowK'] > 95) & (df['SlowD'] > 95), -1, df['Stochastic_Signal'])
    
    # Ultra-strict CMF Signal - Require stronger money flow confirmation
    df['CMF_Signal'] = np.where(df['CMF'] > 0.4, -1, np.where(df['CMF'] < -0.4, 1, 0))
    
    # Ultra-strict CCI Signal - Require extreme deviations
    df['CCI_Signal'] = np.where(df['CCI'] < -220, 1, 0)
    df['CCI_Signal'] = np.where(df['CCI'] > 220, -1, df['CCI_Signal'])
    
    # Combined signal for ultra-strict confirmations
    df['Combined_Signal'] = df[['MACD_Signal','RSI_Signal', 'BB_Signal', 
                                'Stochastic_Signal', 'CMF_Signal', 'CCI_Signal']].sum(axis=1)
    
    return df


def plot_combined_signals(df, ticker):
    # Create a figure
    fig = go.Figure()

    # Add closing price trace
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], 
        mode='lines', 
        name='Closing Price', 
        line=dict(color='#36D7B7', width=2)  # Brighter pink
    ))

    # Add buy signals
    buy_signals = df[df['Combined_Signal'] >= 3]
    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals['Close'], 
        mode='markers', 
        marker=dict(symbol='triangle-up', size=12, color='lightgreen'), 
        name='Buy Signal'
    ))

    # Add sell signals
    sell_signals = df[df['Combined_Signal'] <= -3]
    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals['Close'], 
        mode='markers', 
        marker=dict(symbol='triangle-down', size=12, color='red'), 
        name='Sell Signal'
    ))

    # Combined signal trace
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Combined_Signal'], 
        mode='lines', 
        name='Combined Signal', 
        line=dict(color='#36A2EB', width=1), # Blue
        yaxis='y2'
    ))

    # Update layout
    fig.update_layout(
        title=f'{ticker}: Stock Price and Combined Trading Signal (Last 120 Days)',
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title='Price',
            side='left',
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis2=dict(
            title='Combined Signal',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white', size=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig

def plot_individual_signals(df, ticker):
    # Create a figure
    fig = go.Figure()
    
    # Add closing price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], 
        mode='lines', 
        name='Closing Price', 
        line=dict(color='#D4B2FF', width=2)  # Brighter pink #D4B2FF
    ))

    # Define colors for different signals



    
    signal_colors = {
    'MACD_Signal': {'buy': 'purple', 'sell': 'lightpink'},  # Light purple / Pale butter
    'RSI_Signal': {'buy': 'purple', 'sell': 'lightpink'},  # Light purple / Pale butter
    'BB_Signal': {'buy': 'purple', 'sell': 'lightpink'},    # Purple / Chiffon yellow
    'Stochastic_Signal': {'buy': 'purple', 'sell': 'lightpink'},  # Purple / Corn silk
    'CMF_Signal': {'buy': 'purple', 'sell': 'lightpink'},  # Deep purple / Lemon chiffon
    'CCI_Signal': {'buy': 'purple', 'sell': 'lightpink'}   # Dark purple / Soft maize
}

    

    # Add buy/sell signals for each indicator
    signal_names = ['MACD_Signal', 'RSI_Signal', 'BB_Signal', 
                    'Stochastic_Signal', 'CMF_Signal', 
                    'CCI_Signal'] 
    
    for signal in signal_names:
        buy_signals = df[df[signal] == 1]
        sell_signals = df[df[signal] == -1]
        
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Close'], 
            mode='markers', 
            marker=dict(
                symbol='triangle-up',
                size=12,
                color=signal_colors[signal]['buy']
            ), 
            name=f'{signal} Buy Signal'
        ))

        fig.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals['Close'], 
            mode='markers', 
            marker=dict(
                symbol='triangle-down',
                size=12,
                color=signal_colors[signal]['sell']
            ), 
            name=f'{signal} Sell Signal'
        ))

    fig.update_layout(
        title=f'{ticker}: Individual Trading Signals',
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title='Price',
            side='left',
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        ),
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white', size=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig

def technical_analysis(ticker, start_date, end_date):
    try:
        # Download stock data using the StockDataFetcher class
        df = StockDataFetcher.fetch_stock_data(ticker, start_date, end_date)

        # Generate signals
        df = generate_trading_signals(df)
        
        # Last 120 days for plotting
        df_last_120 = df.tail(120)

        # Plot combined signals
        fig_signals = plot_combined_signals(df_last_120, ticker)

        # Plot individual signals
        fig_individual_signals = plot_individual_signals(df_last_120, ticker)

        return fig_signals, fig_individual_signals
    
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Advanced Stock Analysis & Forecasting App")
    gr.Markdown("Enter a stock ticker, start date, and end date to analyze and forecast stock prices.")
    
    with gr.Row():
        ticker_input = gr.Textbox(label="Enter Stock Ticker", value="NVDA")
        start_date_input = gr.Textbox(label="Enter Start Date (YYYY-MM-DD)", value="2022-01-01")
        end_date_input = gr.Textbox(label="Enter End Date (YYYY-MM-DD)", value="2026-01-01")
    
    # Create tabs for different analysis types
    with gr.Tabs() as tabs:

        with gr.TabItem("Technical Analysis"):
            analysis_button = gr.Button("Generate Technical Analysis")
        
            individual_signals = gr.Plot(label="Individual Trading Signals")
            combined_signals = gr.Plot(label="Combined Trading Signals")
            
            # Connect button to function
            analysis_button.click(
                technical_analysis,
                inputs=[ticker_input, start_date_input, end_date_input],
                outputs=[combined_signals, individual_signals]
            )

        with gr.TabItem("TimesFM Forecast"):
            timesfm_button = gr.Button("Generate TimesFM Forecast")
            timesfm_plot = gr.Plot(label="TimesFM Stock Price Forecast")
            
            # Connect button to function
            timesfm_button.click(
                timesfm_forecast,
                inputs=[ticker_input, start_date_input, end_date_input],
                outputs=timesfm_plot
            )
            
        with gr.TabItem("Prophet Forecast"):
            prophet_button = gr.Button("Generate Prophet Forecast")
            prophet_recent_plot = gr.Plot(label="Recent Stock Prices")
            prophet_forecast_plot = gr.Plot(label="Prophet 30-Day Forecast")
            prophet_components = gr.Plot(label="Forecast Components")  # Changed from gr.Image to gr.Plot
            
            # Connect button to function
            prophet_button.click(
                prophet_forecast,
                inputs=[ticker_input, start_date_input, end_date_input],
                outputs=[prophet_recent_plot, prophet_forecast_plot, prophet_components]
            )
            


# Launch the app
demo.launch()