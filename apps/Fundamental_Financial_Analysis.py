import pandas as pd
import yfinance as yf
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from functools import lru_cache
import asyncio
import concurrent.futures
import time
from typing import Dict, List, Optional, Any, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_analyzer')

# Cache Yahoo Finance data to avoid rate limits
@lru_cache(maxsize=100)
def get_financial_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch financial data for a given stock ticker using Yahoo Finance.
    
    Args:
        ticker: Stock symbol to fetch data for
        
    Returns:
        Dictionary of financial metrics or None if fetch failed
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'Ticker': ticker,
            'PE_Ratio': info.get('forwardPE'),
            'Debt_to_Equity': info.get('debtToEquity'),
            'Revenue_Growth': info.get('revenueGrowth'),
            'ROE': info.get('returnOnEquity'),
            'ROA': info.get('returnOnAssets'),
            'Gross_Margin': info.get('grossMargins'),
            'EBITDA': info.get('ebitda'),
            'Market_Cap': info.get('marketCap'),
            'Dividend_Yield': info.get('dividendYield'),
            'Profit_Margin': info.get('profitMargins'),
            'EPS_Growth': info.get('earningsGrowth'),
            'Price_to_Book': info.get('priceToBook'),
            'Current_Price': info.get('currentPrice')
        }
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

# Fetch data concurrently for multiple tickers
async def fetch_data_concurrently(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch financial data for multiple tickers concurrently.
    
    Args:
        tickers: List of stock symbols
        
    Returns:
        List of financial data dictionaries for each ticker
    """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                get_financial_data,
                ticker
            )
            for ticker in tickers
        ]
        results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

# Robust normalization using winsorization (cap outliers at specified percentiles)
def normalize(series: pd.Series, reverse: bool = False, 
              lower_percentile: float = 0.10, upper_percentile: float = 0.90) -> pd.Series:
    """
    Normalize a series to a 0-10 scale using winsorization.
    
    Args:
        series: Data series to normalize
        reverse: If True, reverse the normalization (10 becomes low, 0 becomes high)
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping
        
    Returns:
        Normalized series
    """
    if series.isna().all() or len(series.unique()) <= 1:
        return pd.Series(5, index=series.index)  # Default to middle value if no variation
        
    q_low = series.quantile(lower_percentile)
    q_high = series.quantile(upper_percentile)
    
    # Avoid division by zero
    if q_high == q_low:
        return pd.Series(5, index=series.index)
        
    clipped_series = series.clip(q_low, q_high)
    
    if reverse:
        return 10 * (1 - (clipped_series - q_low) / (q_high - q_low))
    return 10 * ((clipped_series - q_low) / (q_high - q_low))

# Calculate scores with customizable weights
def calculate_scores(df: pd.DataFrame, growth_weight: float, 
                     value_weight: float, risk_weight: float) -> pd.DataFrame:
    """
    Calculate stock scores based on various financial metrics.
    
    Args:
        df: DataFrame containing financial metrics
        growth_weight: Weight for growth metrics
        value_weight: Weight for value metrics
        risk_weight: Weight for risk metrics
        
    Returns:
        DataFrame with added score columns
    """
    # Make a copy to avoid modifying the original
    scored_df = df.copy()
    
    # Growth Metrics (higher is better)
    scored_df['Revenue_Growth_Score'] = normalize(df['Revenue_Growth'])
    scored_df['EPS_Growth_Score'] = normalize(df['EPS_Growth'])
    scored_df['ROE_Score'] = normalize(df['ROE'])
    scored_df['ROA_Score'] = normalize(df['ROA'])
    
    # Calculate Growth Score with nan handling
    growth_cols = ['Revenue_Growth_Score', 'EPS_Growth_Score', 'ROE_Score', 'ROA_Score']
    scored_df['Growth_Score'] = scored_df[growth_cols].mean(axis=1)

    # Value Metrics (lower is better)
    scored_df['PE_Ratio_Score'] = normalize(df['PE_Ratio'], reverse=True)
    scored_df['Price_to_Book_Score'] = normalize(df['Price_to_Book'], reverse=True)
    scored_df['Dividend_Yield_Score'] = normalize(df['Dividend_Yield'])  # Higher yield is better
    
    # Calculate Value Score
    value_cols = ['PE_Ratio_Score', 'Price_to_Book_Score', 'Dividend_Yield_Score']
    scored_df['Value_Score'] = scored_df[value_cols].mean(axis=1)

    # Risk Metrics (higher values indicate lower risk)
    scored_df['Debt_to_Equity_No_Risk_Score'] = normalize(df['Debt_to_Equity'], reverse=True)  # Lower debt = lower risk
    scored_df['Profit_Margin_No_Risk_Score'] = normalize(df['Profit_Margin'])  # Higher margin = lower risk
    scored_df['Market_Cap_No_Risk_Score'] = normalize(df['Market_Cap'])  # Larger companies = lower risk
    
    # Calculate No_Risk_Score
    no_risk_cols = ['Debt_to_Equity_No_Risk_Score', 'Profit_Margin_No_Risk_Score', 'Market_Cap_No_Risk_Score']
    scored_df['No_Risk_Score'] = scored_df[no_risk_cols].mean(axis=1)

    # Normalize weights to ensure they sum to 1.0
    total = growth_weight + value_weight + risk_weight
    if total == 0:
        growth_weight = value_weight = risk_weight = 1/3
    else:
        growth_weight /= total
        value_weight /= total
        risk_weight /= total

    # Total Score (Weighted Average)
    scored_df['Total_Score'] = (
        growth_weight * scored_df['Growth_Score'] + 
        value_weight * scored_df['Value_Score'] + 
        risk_weight * scored_df['No_Risk_Score']
    )

    return scored_df

# Generate bar chart for scores with custom styling
def plot_bar_chart(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a bar chart showing Growth, Value, and No_Risk scores for each ticker.
    
    Args:
        df: DataFrame containing score data
        
    Returns:
        Matplotlib figure
    """
    # Set a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Custom colors
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    df.set_index('Ticker')[['Growth_Score', 'Value_Score', 'No_Risk_Score']].plot(
        kind='bar', 
        stacked=False,  # Changed to unstacked for better comparison
        color=colors,
        width=0.7,
        alpha=0.8,
        ax=ax
    )
    
    # Add total score as a line
    total_scores = df.set_index('Ticker')['Total_Score']
    ax2 = ax.twinx()
    ax2.plot(range(len(total_scores)), total_scores, 'ro-', linewidth=2.5, markersize=8, label='Total Score')
    ax2.set_ylim(0, 10.5)
    ax2.set_ylabel('Total Score', fontsize=12, color='r')
    
    # Enhance appearance
    ax.set_title("Stock Analysis Scores", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Component Scores (0-10)", fontsize=12)
    ax.set_xlabel("", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 10.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=4, frameon=True, fontsize=10)
    
    plt.tight_layout()
    return fig

# Generate radar plot for scores with improved styling
def plot_radar_chart(df: pd.DataFrame, tickers: List[str]) -> plt.Figure:
    """
    Generate a radar chart comparing selected tickers.
    
    Args:
        df: DataFrame containing score data
        tickers: List of tickers to include in the radar chart
        
    Returns:
        Matplotlib figure
    """
    # Filter for requested tickers only
    plot_df = df[df['Ticker'].isin(tickers)]
    
    if plot_df.empty:
        # Fallback if none of the requested tickers are found
        plot_df = df.head(min(3, len(df)))
        tickers = plot_df['Ticker'].tolist()
    
    # Categories and angles
    categories = ['Growth', 'Value', 'No_Risk', 'Total']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set up plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Custom colors for each ticker
    colors = plt.cm.viridis(np.linspace(0, 1, len(tickers)))
    
    # Plot each ticker
    for i, ticker in enumerate(tickers):
        ticker_data = plot_df[plot_df['Ticker'] == ticker]
        if ticker_data.empty:
            continue
            
        values = ticker_data[['Growth_Score', 'Value_Score', 'No_Risk_Score', 'Total_Score']].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=ticker)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Set chart properties
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_yticks(np.arange(2, 12, 2))
    ax.set_yticklabels(np.arange(2, 12, 2), size=10)
    ax.set_ylim(0, 10)
    
    # Add title and legend
    plt.title("Stock Comparison Radar Chart", size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True)
    
    return fig

# Generate a detailed metrics table
def create_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a detailed metrics table for display.
    
    Args:
        df: DataFrame with stock data
        
    Returns:
        DataFrame with formatted metrics
    """
    metrics_df = df[['Ticker', 'Current_Price', 'PE_Ratio', 'Price_to_Book', 
                     'Debt_to_Equity', 'ROE', 'ROA', 'Revenue_Growth', 
                     'EPS_Growth', 'Profit_Margin', 'Dividend_Yield']].copy()
    
    # Format percentages
    for col in ['ROE', 'ROA', 'Revenue_Growth', 'EPS_Growth', 'Profit_Margin', 'Dividend_Yield']:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
    
    # Format ratios
    for col in ['PE_Ratio', 'Price_to_Book', 'Debt_to_Equity']:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        
    # Format price
    metrics_df['Current_Price'] = metrics_df['Current_Price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
    
    return metrics_df

# Main analysis function for Gradio app
async def analyze_tickers(
    tickers: str, 
    growth_weight: float, 
    value_weight: float, 
    risk_weight: float,
    top_n: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure, plt.Figure]:
    """
    Analyze stock tickers and generate visualizations.
    
    Args:
        tickers: Comma-separated list of stock tickers
        growth_weight: Weight for growth metrics
        value_weight: Weight for value metrics
        risk_weight: Weight for risk metrics
        top_n: Number of top stocks to highlight
        
    Returns:
        Tuple containing scores DataFrame, metrics DataFrame, bar chart, and radar chart
    """
    start_time = time.time()
    
    # Parse and clean ticker list
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    
    if not ticker_list:
        return pd.DataFrame(), pd.DataFrame(), plt.figure(), plt.figure()
    
    # Fetch data asynchronously
    data = await fetch_data_concurrently(ticker_list)
    
    if not data:
        logger.warning("No valid data retrieved for any tickers")
        return pd.DataFrame(), pd.DataFrame(), plt.figure(), plt.figure()
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Handle missing values - replace with median for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Calculate scores
    df = calculate_scores(df, growth_weight, value_weight, risk_weight)
    
    # Sort by Total Score
    df = df.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)
    
    # Create metrics table
    metrics_table = create_metrics_table(df)
    
    # Generate bar chart for all tickers
    bar_chart = plot_bar_chart(df)
    
    # Generate radar chart for top N tickers
    top_tickers = df.head(min(top_n, len(df)))['Ticker'].tolist()
    radar_chart = plot_radar_chart(df, top_tickers)
    
    # Prepare scores table for display
    scores_table = df[['Ticker', 'Total_Score', 'Growth_Score', 'Value_Score', 'No_Risk_Score']].copy()
    scores_table = scores_table.round(2)
    
    logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
    
    return scores_table, metrics_table, bar_chart, radar_chart



# Custom CSS for better appearance
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.container {
    max-width: 1200px;
    margin: auto;
}

button#analyze-btn {
    background-color: #003366;  /* Dark blue color */
    color: white;
    border: none;
}


"""

# Gradio interface
def create_gradio_interface():
    """Create and configure the Gradio interface"""

    
    with gr.Blocks(theme=gr.themes.Monochrome(),css=custom_css) as iface:
        gr.Markdown("# Fundamental Financial Analysis")
        gr.Markdown("""
        Enter comma-separated stock tickers and adjust the weights to analyze stocks based on 
        growth potential, value metrics, and risk factors.
        """)
        
        with gr.Row():
            tickers_input = gr.Textbox(
                label="Stock Tickers (comma-separated)", 
                placeholder="AAPL, MSFT, GOOG, AMZN, TSLA",
                lines=1
            )
            analyze_btn = gr.Button("Analyze Stocks", variant="primary")
        
        with gr.Row():
            with gr.Column():
                growth_weight = gr.Slider(
                    minimum=0, maximum=1, step=0.05, 
                    label="Growth Weight", value=0.4
                )
            with gr.Column():
                value_weight = gr.Slider(
                    minimum=0, maximum=1, step=0.05, 
                    label="Value Weight", value=0.4
                )
            with gr.Column():
                risk_weight = gr.Slider(
                    minimum=0, maximum=1, step=0.05, 
                    label="Risk Weight", value=0.2
                )
        
        with gr.Tabs():
            with gr.TabItem("Scores & Charts"):
                with gr.Row():
                    with gr.Column():
                        scores_output = gr.Dataframe(label="Stock Scores")
                    with gr.Column():
                        metrics_output = gr.Dataframe(label="Financial Metrics")
                
                with gr.Row():
                    with gr.Column():
                        bar_chart_output = gr.Plot(label="Component Scores Chart")
                    with gr.Column():
                        radar_chart_output = gr.Plot(label="Top Stocks Comparison")
            
            with gr.TabItem("Help & Information"):
                gr.Markdown("""
                ## How to Use This Tool
                
                1. Enter stock tickers separated by commas (e.g., "AAPL, MSFT, GOOG")
                2. Adjust weights based on your investment strategy:
                   - **Growth Weight**: Emphasizes revenue growth, EPS growth, ROE, and ROA
                   - **Value Weight**: Focuses on PE ratio, price-to-book, and dividend yield
                   - **Risk Weight**: Considers debt-to-equity ratio, profit margins, and market cap
                3. Click "Analyze Stocks" to see results
                
                ## About the Scores
                
                All metrics are normalized on a scale of 0-10, with higher being better:
                - **Growth Score**: Higher values indicate stronger growth potential
                - **Value Score**: Higher values indicate the stock may be undervalued
                - **No_Risk_Score**: Higher values suggest lower relative risk
                - **Total Score**: Weighted average of the three component scores
                
                ## Data Source
                
                Financial data is provided by Yahoo Finance via the yfinance package.
                """)
        
        # Set up the async functionality with a wrapper function
        def analyze_wrapper(*args):
            return asyncio.run(analyze_tickers(*args))
        
        analyze_btn.click(
            analyze_wrapper,
            inputs=[tickers_input, growth_weight, value_weight, risk_weight],
            outputs=[scores_output, metrics_output, bar_chart_output, radar_chart_output]
        )
    
    return iface

# Entry point
if __name__ == "__main__":
    logger.info("Starting Stock Analyzer app")
    iface = create_gradio_interface()
    iface.launch() #share=False