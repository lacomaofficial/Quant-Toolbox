import os
import pandas as pd
import requests
import numpy as np
import gradio as gr
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Configuration
class Config:
    FINNHUB_API_KEY = "cuj17q1r01qm7p9n307gcuj17q1r01qm7p9n3080"
    DEFAULT_DAYS = 30  # Reduced from 365 to make it faster
    DATA_DIR = "data"
    
    @classmethod
    def initialize(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)

Config.initialize()

# Simple sentiment analyzer
class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text):
        if not isinstance(text, str) or not text.strip():
            return 0
        return self.analyzer.polarity_scores(text)['compound']

# News fetcher and sentiment analyzer
class StockNewsAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def get_file_path(self, file_type):
        return os.path.join(Config.DATA_DIR, f"{self.symbol}_{file_type}.csv")
    
    def get_news(self, days=Config.DEFAULT_DAYS, force_refresh=False):
        """Fetch news articles from Finnhub API"""
        file_path = self.get_file_path("news")
        
        # Return cached data if it exists and no refresh is forced
        if os.path.exists(file_path) and not force_refresh:
            try:
                return pd.read_csv(file_path, parse_dates=['datetime'])
            except Exception:
                # If the file is corrupted, fetch fresh data
                pass
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch from API
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": self.symbol,
            "from": start_date.strftime('%Y-%m-%d'),
            "to": end_date.strftime('%Y-%m-%d'),
            "token": Config.FINNHUB_API_KEY,
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if not data or not isinstance(data, list):
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(data)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                # Save to CSV
                df.to_csv(file_path, index=False)
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()
    
    def analyze_news_sentiment(self, days=Config.DEFAULT_DAYS, force_refresh=False):
        """Analyze sentiment from news articles"""
        news_df = self.get_news(days, force_refresh)
        
        if news_df.empty:
            return None, None, None
        
        # Add sentiment scores to headlines
        if 'headline' in news_df.columns:
            news_df['sentiment_score'] = news_df['headline'].apply(self.sentiment_analyzer.analyze)
            
            # Add date column for daily aggregation
            news_df['date'] = news_df['datetime'].dt.date
            news_df['date'] = pd.to_datetime(news_df['date'])
            
            # Get stock price for the same period
            try:
                start_date = news_df['date'].min() - timedelta(days=5)  # Get a few days before for context
                end_date = news_df['date'].max() + timedelta(days=1)
                stock_data = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
                if not stock_data.empty and 'Close' in stock_data.columns:
                    stock_data = stock_data[['Close']]
                    stock_data.columns = ['close']
                    stock_data = stock_data.reset_index()
                    stock_data.rename(columns={'Date': 'date'}, inplace=True)
                    stock_data['date'] = pd.to_datetime(stock_data['date'].dt.date)
                    stock_data.set_index('date', inplace=True)
                else:
                    stock_data = pd.DataFrame()
            except Exception:
                stock_data = pd.DataFrame()
            
            # Group by date for daily sentiment
            daily_sentiment = news_df.groupby('date').agg(
                avg_sentiment=('sentiment_score', 'mean'),
                article_count=('sentiment_score', 'count'),
                positive_count=('sentiment_score', lambda x: sum(x > 0.05)),
                negative_count=('sentiment_score', lambda x: sum(x < -0.05)),
                neutral_count=('sentiment_score', lambda x: sum((x >= -0.05) & (x <= 0.05)))
            ).reset_index()
            
            # Sort news articles by sentiment (most positive and most negative)
            news_df = news_df.sort_values('sentiment_score', ascending=False)
            
            # Get top 5 positive and negative headlines
            top_positive = news_df[news_df['sentiment_score'] > 0].head(5)
            top_negative = news_df[news_df['sentiment_score'] < 0].tail(5)
            
            # Return sentiment data and headlines
            return daily_sentiment, stock_data, pd.concat([top_positive, top_negative])
        
        return None, None, None

# Visualization Functions
def create_sentiment_overview(daily_sentiment, stock_data, top_headlines, symbol):
    """Create a sentiment overview visualization"""
    if daily_sentiment is None or daily_sentiment.empty:
        return None
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{}]], 
                         row_heights=[0.7, 0.3], vertical_spacing=0.1)
    
    # Add stock price if available
    if not stock_data.empty:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index, 
                y=stock_data['close'],
                name='Stock Price',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1, secondary_y=False
        )
    
    # Add daily sentiment score
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['date'], 
            y=daily_sentiment['avg_sentiment'],
            name='Sentiment Score',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Add article count as a bar
    fig.add_trace(
        go.Bar(
            x=daily_sentiment['date'], 
            y=daily_sentiment['article_count'],
            name='Article Count',
            marker_color='rgba(135, 206, 235, 0.5)',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add sentiment breakdown bars (positive, negative, neutral)
    fig.add_trace(
        go.Bar(
            x=daily_sentiment['date'], 
            y=daily_sentiment['positive_count'],
            name='Positive',
            marker_color='rgba(0, 128, 0, 0.7)'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=daily_sentiment['date'], 
            y=daily_sentiment['negative_count'],
            name='Negative',
            marker_color='rgba(255, 0, 0, 0.7)'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=daily_sentiment['date'], 
            y=daily_sentiment['neutral_count'],
            name='Neutral',
            marker_color='rgba(128, 128, 128, 0.7)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} News Sentiment Analysis",
        template='plotly_white',
        hovermode='x unified',
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=700,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Stock Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Article Count", row=2, col=1)
    
    return fig

def format_headlines(headlines_df):
    """Format headlines with sentiment scores for display"""
    if headlines_df is None or headlines_df.empty:
        return "No headlines available."
        
    # Sort by sentiment score (most positive first)
    headlines_df = headlines_df.sort_values('sentiment_score', ascending=False)
    
    result = "## Top Positive Headlines\n\n"
    for _, row in headlines_df[headlines_df['sentiment_score'] > 0].head(5).iterrows():
        date = row['datetime'].strftime('%Y-%m-%d')
        sentiment = row['sentiment_score']
        color = "green"
        result += f"- **{date}** | [{row['headline']}]({row['url']}) | <span style='color:{color};'>*{sentiment:.2f}*</span>\n\n"
    
    result += "## Top Negative Headlines\n\n"
    for _, row in headlines_df[headlines_df['sentiment_score'] < 0].sort_values('sentiment_score').head(5).iterrows():
        date = row['datetime'].strftime('%Y-%m-%d')
        sentiment = row['sentiment_score']
        color = "red"
        result += f"- **{date}** | [{row['headline']}]({row['url']}) | <span style='color:{color};'>*{sentiment:.2f}*</span>\n\n"
    
    return result

def create_summary(daily_sentiment, symbol):
    """Create a text summary of sentiment analysis"""
    if daily_sentiment is None or daily_sentiment.empty:
        return f"No sentiment data available for {symbol}."
    
    # Calculate overall sentiment statistics
    avg_sentiment = daily_sentiment['avg_sentiment'].mean()
    total_articles = daily_sentiment['article_count'].sum()
    total_positive = daily_sentiment['positive_count'].sum()
    total_negative = daily_sentiment['negative_count'].sum()
    total_neutral = daily_sentiment['neutral_count'].sum()
    
    # Determine sentiment trend
    sentiment_trend = "neutral"
    if avg_sentiment > 0.05:
        sentiment_trend = "positive"
    elif avg_sentiment < -0.05:
        sentiment_trend = "negative"
    
    # Create summary
    summary = f"""
## {symbol} Sentiment Summary

### Overview
- **Overall Sentiment**: {sentiment_trend.title()} (Score: {avg_sentiment:.2f})
- **Total Articles**: {total_articles}
- **Date Range**: {daily_sentiment['date'].min().strftime('%Y-%m-%d')} to {daily_sentiment['date'].max().strftime('%Y-%m-%d')}

### Sentiment Breakdown
- **Positive Articles**: {total_positive} ({total_positive/total_articles*100:.1f}%)
- **Negative Articles**: {total_negative} ({total_negative/total_articles*100:.1f}%)
- **Neutral Articles**: {total_neutral} ({total_neutral/total_articles*100:.1f}%)
    """
    
    return summary

# Gradio Interface
def analyze_stock_sentiment(symbol, days, refresh_data):
    """Main function for Gradio interface"""
    if not symbol:
        return "Please enter a valid stock symbol.", None, "No headlines available."
    
    # Make sure symbol is uppercase
    symbol = symbol.upper().strip()
    
    # Create analyzer
    analyzer = StockNewsAnalyzer(symbol)
    
    # Get sentiment data
    daily_sentiment, stock_data, top_headlines = analyzer.analyze_news_sentiment(days, refresh_data)
    
    if daily_sentiment is None or daily_sentiment.empty:
        return f"No news data available for {symbol}. Try another symbol or increase the time range.", None, "No headlines available."
    
    # Create visualization
    sentiment_plot = create_sentiment_overview(daily_sentiment, stock_data, top_headlines, symbol)
    
    # Generate summary
    summary = create_summary(daily_sentiment, symbol)
    
    # Format headlines
    headlines = format_headlines(top_headlines)
    
    return summary, sentiment_plot, headlines

# Build Gradio interface
def build_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="Stock Sentiment Analysis", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Stock News Sentiment Analysis")
        gr.Markdown("Analyze the sentiment of news articles for any stock symbol")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Inputs
                symbol_input = gr.Textbox(label="Stock Symbol", value="BABA", placeholder="e.g., AAPL, MSFT, GOOGL")
                days_input = gr.Slider(label="Days of History", minimum=7, maximum=90, value=90, step=1)
                refresh_data = gr.Checkbox(label="Refresh Data", value=False)
                analyze_button = gr.Button("Analyze Sentiment", variant="primary")
        
        # Outputs
        summary_text = gr.Markdown()
        sentiment_plot = gr.Plot()
        headlines_text = gr.Markdown()
        
        # Set up event handlers
        analyze_button.click(
            fn=analyze_stock_sentiment,
            inputs=[symbol_input, days_input, refresh_data],
            outputs=[summary_text, sentiment_plot, headlines_text]
        )
        

    return app

# Main function
def main():
    app = build_interface()
    app.launch()

if __name__ == "__main__":
    main()