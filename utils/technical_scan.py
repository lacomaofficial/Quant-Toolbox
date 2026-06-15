# technical_scan

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional



import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


 

def smooth_moving_average(series: pd.Series, window: int) -> pd.Series:
    if len(series) < window or window <= 0:
        return pd.Series(series.mean(), index=series.index)
    result = pd.Series(index=series.index, dtype=float)
    result.iloc[:window] = series.iloc[:window].mean()
    for i in range(window, len(series)):
        result.iloc[i] = (result.iloc[i-1] * (window - 1) + series.iloc[i]) / window
    return result.ffill().bfill().fillna(series.mean())

def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    if len(close) <= window:
        return pd.Series(50.0, index=close.index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = smooth_moving_average(gain, window)
    avg_loss = smooth_moving_average(loss, window)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=close.index).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(50.0)

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window=14, d_window=3):
    if len(close) < k_window:
        return pd.Series(50.0, index=close.index), pd.Series(50.0, index=close.index)
    lowest = low.rolling(k_window, min_periods=1).min()
    highest = high.rolling(k_window, min_periods=1).max()
    k_pct = ((close - lowest) / (highest - lowest + 1e-10)) * 100
    k_pct = k_pct.clip(0, 100)
    d_pct = k_pct.rolling(d_window, min_periods=1).mean()
    return k_pct.ffill().bfill().fillna(50.0), d_pct.ffill().bfill().fillna(50.0)

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, window=20):
    if len(close) < window:
        return pd.Series(0.0, index=close.index)
    typical_price = (high + low + close) / 3.0
    sma = typical_price.rolling(window, min_periods=1).mean()
    mean_deviation = (typical_price - sma).abs().rolling(window, min_periods=1).mean()
    cci = (typical_price - sma) / (0.015 * mean_deviation + 1e-10)
    return cci.ffill().bfill().fillna(0.0)

def calculate_sma_robust(series: pd.Series, window: int) -> pd.Series:
    if len(series) < window or window <= 0:
        return pd.Series(series.mean(), index=series.index)
    return series.rolling(window=window, min_periods=window).mean().ffill().bfill().fillna(series.mean())

def calculate_ema_robust(series: pd.Series, span: int) -> pd.Series:
    if len(series) < span or span <= 0:
        return pd.Series(series.mean(), index=series.index)
    return series.ewm(span=span, adjust=False, min_periods=span).mean().ffill().bfill().fillna(series.mean())

def calculate_bollinger_bands_robust(close: pd.Series, window=20, num_std=3.0):
    if len(close) < window:
        mid = pd.Series(close.mean(), index=close.index)
        return mid, mid, mid
    sma = calculate_sma_robust(close, window)
    std = close.rolling(window=window, min_periods=window).std().fillna(1e-10)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma.ffill().bfill(), upper.ffill().bfill(), lower.ffill().bfill()

def calculate_momentum(close: pd.Series) -> dict:
    """Calculate momentum for the past 7 calendar days"""
    if len(close) < 7:
        return {'weekly_return': 0}
    
    end_date = close.index[-1]
    start_date = end_date - timedelta(days=7)
    
    mask = close.index <= start_date
    if mask.any():
        idx_7_days_ago = close.index[mask][-1]
        price_7_days_ago = close.loc[idx_7_days_ago]
    else:
        price_7_days_ago = close.iloc[0]
    
    weekly_return = ((close.iloc[-1] - price_7_days_ago) / price_7_days_ago) * 100
    
    return {
        'weekly_return': round(weekly_return, 2)
    }





def get_todays_signals_with_momentum(df: pd.DataFrame) -> dict:
    """Get signals for today and check weekly momentum"""
    df = df.copy()
    close = df['Close']
    has_hl = all(col in df.columns for col in ['High', 'Low'])

    high = df['High'] if has_hl else close
    low = df['Low'] if has_hl else close

    # Calculate indicators
    rsi = calculate_rsi(close, window=14)
    stoch_k, stoch_d = calculate_stochastic(high, low, close, k_window=14, d_window=3)
    cci = calculate_cci(high, low, close, window=20)
    _, bb_upper, bb_lower = calculate_bollinger_bands_robust(close, window=20, num_std=3.0)

    # Get weekly momentum
    momentum = calculate_momentum(close)
    
    signals = {}
    
    # RSI Signal (20/80)
    if len(rsi) > 0:
        if rsi.iloc[-1] < 20:
            signals['RSI'] = 'BUY'
        elif rsi.iloc[-1] > 90:
            signals['RSI'] = 'SELL'
    
    # Bollinger Bands Signal (3.0σ)
    if len(close) > 0:
        if close.iloc[-1] <= bb_lower.iloc[-1]:
            signals['BB'] = 'BUY'
        elif close.iloc[-1] >= bb_upper.iloc[-1]:
            signals['BB'] = 'SELL'
    
    # Stochastic Signal (5/99)
    if len(stoch_k) > 0:
        if stoch_k.iloc[-1] < 5 and stoch_d.iloc[-1] < 5:
            signals['Stochastic'] = 'BUY'
        elif stoch_k.iloc[-1] > 99 and stoch_d.iloc[-1] > 99:
            signals['Stochastic'] = 'SELL'
    
    # CCI Signal (±200)
    if len(cci) > 0:
        if cci.iloc[-1] < -220:
            signals['CCI'] = 'BUY'
        elif cci.iloc[-1] > 220:
            signals['CCI'] = 'SELL'
    
    return {
        'signals': signals,
        'momentum': momentum,
        'current_price': close.iloc[-1]
    }

def multi_technical_analysis(tickers: list, verbose: bool = True, show_progress: bool = True) -> tuple:
    """
    Screen stocks for signals TODAY only using this week's momentum
    
    Parameters:
    - tickers: List of stock tickers
    - verbose: If True, print detailed output. If False, return only DataFrames.
    - show_progress: If True, show tqdm progress bar during scanning.
    
    Returns:
    - Tuple of (buys_df, sells_df)
    """
    buy_results = []
    sell_results = []
    
    # Counters for summary
    processed = 0
    skipped = 0
    errors = 0
    signals_found = 0
    
    if verbose:
        print("=" * 80)
        print("QUANTITATIVE SIGNAL SCREENING")
        print(f"SCAN DATE: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"UNIVERSE SIZE: {len(tickers)}")
        print("=" * 80)
        print("THRESHOLD CONFIGURATION:")
        print("  RSI: 20/90")
        print("  STOCHASTIC: 5/99 (K & D)")
        print("  CCI: +/-220")
        print("  BOLLINGER BANDS: 3.0 SIGMA")
        print("=" * 80)
        print("\nSCANNING...\n")
    
    # Setup progress bar
    if show_progress:
        pbar = tqdm(tickers, desc="Analyzing", unit="ticker", 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    else:
        pbar = tickers
    
    for ticker in pbar:
        # Update progress bar description with current ticker
        if show_progress:
            pbar.set_postfix_str(f"Current: {ticker}")
        
        try:
            stock_data = yf.download(ticker, period='2mo', progress=False)
            
            if len(stock_data) < 30:
                skipped += 1
                continue
            
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(level=1)
            
            stock_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            
            analysis = get_todays_signals_with_momentum(stock_data)
            signals = analysis['signals']
            momentum = analysis['momentum']
            current_price = analysis['current_price']
            
            processed += 1
            
            if signals:
                signals_found += 1
                buy_signals = sum(1 for s in signals.values() if s == 'BUY')
                sell_signals = sum(1 for s in signals.values() if s == 'SELL')
                
                result = {
                    'Ticker': ticker,
                    'Price': round(current_price, 2),
                    'Signal_Count': max(buy_signals, sell_signals),
                    '7D_Return_%': momentum['weekly_return'],
                    'Indicators': ', '.join([f"{k}({v})" for k, v in signals.items()])
                }
                
                if buy_signals > sell_signals:
                    buy_results.append(result)
                else:
                    sell_results.append(result)
            
        except Exception as e:
            errors += 1
            if show_progress:
                # Only write error to stderr or log, don't break the bar
                pass
    
    # Close progress bar
    if show_progress:
        pbar.close()
    
    buys_df = pd.DataFrame(buy_results) if buy_results else pd.DataFrame()
    sells_df = pd.DataFrame(sell_results) if sell_results else pd.DataFrame()
    
    if not buys_df.empty:
        buys_df = buys_df.sort_values('Signal_Count', ascending=False)
    if not sells_df.empty:
        sells_df = sells_df.sort_values('Signal_Count', ascending=False)
    
    if verbose:
        # Summary statistics
        print("\n" + "=" * 80)
        print("SCAN SUMMARY")
        print("=" * 80)
        print(f"  Total tickers in universe: {len(tickers)}")
        print(f"  Successfully processed: {processed}")
        print(f"  Skipped (<30 days data): {skipped}")
        print(f"  Download errors: {errors}")
        print(f"  Tickers with signals: {signals_found} ({signals_found/processed*100:.1f}% of processed)")
        print(f"    └─ BUY signals: {len(buy_results)}")
        print(f"    └─ SELL signals: {len(sell_results)}")
        
        # BUY SIGNALS TABLE
        if not buys_df.empty:
            print("\n" + "=" * 80)
            print("LONG SIGNALS (BUY)")
            print("=" * 80)
            print(buys_df.to_string(index=False))
        else:
            print("\n" + "=" * 80)
            print("LONG SIGNALS (BUY): NONE")
            print("=" * 80)
        
        # SELL SIGNALS TABLE
        if not sells_df.empty:
            print("\n" + "=" * 80)
            print("SHORT SIGNALS (SELL)")
            print("=" * 80)
            print(sells_df.to_string(index=False))
        else:
            print("\n" + "=" * 80)
            print("SHORT SIGNALS (SELL): NONE")
            print("=" * 80)
        
        print("\n" + "=" * 80)
    
    return buys_df, sells_df

'''
# Example usage:
from tickers_list import all_tickers 
buys_df, sells_df = multi_technical_analysis(all_tickers, verbose=True, show_progress=True)
    
'''