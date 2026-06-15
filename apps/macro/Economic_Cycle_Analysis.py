import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import gradio as gr

class SimplifiedDalioAnalysis:
    """
    A simplified implementation of Ray Dalio's approach to macro investing.
    Uses a more straightforward calculation of the cycle score.
    """
    
    def __init__(self, start_date="2000-01-01"):
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Key instruments for analysis
        self.tickers = [
            '^TNX', '^IRX', '^VIX', 'TIP', 'IEF', 'XLY', 'XLP', 'LQD', 'HYG',
            'SPY', 'VGK', 'FXI', 'XLF', 'XLK', 'XLV', 'XLE', 'GLD', 'DBC', 'TLT',
            'XLI', 'XLB', 'XLU', 'XLRE'
        ]
        
        # Recession periods (NBER)
        self.recessions = [
            (datetime(2007,12,1), datetime(2009,6,1)),
            (datetime(2020,2,1), datetime(2020,4,1))
        ]
        
        # Colors for different cycle phases
        self.cycle_colors = {
            'Crisis': '#d62728',  # Red
            'Late Cycle': '#ff7f0e',  # Orange
            'Mid Cycle': '#2ca02c',  # Green
            'Early Recovery': '#1f77b4'  # Blue
        }
        
        # Portfolio allocations based on Dalio's principles
        self.allocations = {
            'Early Recovery': {
                'Equities': 40,
                'Long-Term Bonds': 15, 
                'Intermediate Bonds': 15,
                'Gold': 15,
                'Commodities': 15
            },
            'Mid Cycle': {
                'Equities': 50,
                'Long-Term Bonds': 20,
                'Intermediate Bonds': 15,
                'Gold': 7.5,
                'Commodities': 7.5
            },
            'Late Cycle': {
                'Equities': 30,
                'Long-Term Bonds': 40,
                'Intermediate Bonds': 15,
                'Gold': 7.5,
                'Commodities': 7.5
            },
            'Crisis': {
                'Equities': 15,
                'Long-Term Bonds': 40,
                'Intermediate Bonds': 15,
                'Gold': 20,
                'Commodities': 10
            }
        }
        
        # Tickers for each asset class
        self.asset_tickers = {
            'Equities': 'SPY',
            'Long-Term Bonds': 'TLT',
            'Intermediate Bonds': 'IEF',
            'Gold': 'GLD',
            'Commodities': 'DBC'
        }
        
        # All weather portfolio components
        self.all_weather = ['SPY', 'TLT', 'IEF', 'GLD', 'DBC']
    
    def download_data(self):
        """Download historical data for all tickers"""
        print(f"Downloading data for {len(self.tickers)} instruments...")
        try:
            self.data = yf.download(self.tickers, start=self.start_date)['Close']
            
            # Handle missing data
            self.data = self.data.ffill().bfill()
            
            # Calculate returns for performance analysis
            self.returns = self.data.pct_change().fillna(0)
            
            print(f"Data downloaded successfully from {self.data.index.min().date()} to {self.data.index.max().date()}")
            return self.data
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise
    
    def calculate_macro_indicators(self):
        """Calculate key macro indicators with simplified approach"""
        print("Calculating macro indicators...")
        d = self.data.copy()
        
        # Ensure key tickers exist
        required_cols = ['^TNX', '^IRX', '^VIX', 'TIP', 'IEF', 'XLY', 'XLP', 'LQD', 'HYG']
        for col in required_cols:
            if col not in d.columns:
                print(f"Warning: {col} not found in data. Using proxy data.")
                d[col] = 0
        
        # Calculate basic indicators
        d['YieldCurve'] = d['^TNX'] - d['^IRX']
        d['TIP_Spread'] = d['TIP'] - d['IEF']
        d['XLY_XLP_Ratio'] = d['XLY'] / d['XLP']
        d['CreditSpread'] = d['HYG'] - d['LQD']
        
        # Normalize indicators (z-score)
        d['Z_Yield'] = self.norm(d['YieldCurve'])
        d['Z_VIX'] = self.norm(d['^VIX'])
        d['Z_TIP'] = self.norm(d['TIP_Spread'])
        d['Z_XLYXLP'] = self.norm(d['XLY_XLP_Ratio'])
        d['Z_Credit'] = self.norm(d['CreditSpread'])
        
        # Compute cycle score with simple weighting
        d['Cycle_Score'] = (
            -1.0 * d['Z_Yield'] +     # Inverted: flatter curve = later cycle
            1.0 * d['Z_VIX'] +        # Higher volatility = stress
            1.0 * d['Z_TIP'] +        # Higher inflation expectations = later cycle
            -1.0 * d['Z_XLYXLP'] +    # Lower consumer discretionary vs staples = recession
            1.0 * d['Z_Credit']       # Credit spread (positive = stress)
        )
        
        # Simple smoothing for noise reduction
        d['Cycle_Score_Smooth'] = d['Cycle_Score'].rolling(window=5).mean().bfill()
        
        # Forecast: simple rolling average
        d['Forecast_Score'] = d['Cycle_Score_Smooth'].rolling(window=10).mean().shift(1)
        
        # Calculate momentum (rate of change)
        d['Cycle_Momentum'] = d['Cycle_Score_Smooth'].diff(10)
        
        # Label cycle phases
        d['Macro_Phase'] = d['Cycle_Score_Smooth'].apply(self.label_phase)
        
        self.macro_data = d
        print("Macro indicators calculated successfully")
        return self.macro_data
    
    def norm(self, series):
        """Normalize a series to z-scores"""
        return (series - series.mean()) / series.std()
    
    def label_phase(self, score):
        """Label the cycle phase based on score"""
        if score > 5:
            return 'Crisis'
        elif score > 3:
            return 'Late Cycle'
        elif score < -2:
            return 'Early Recovery'
        else:
            return 'Mid Cycle'
    
    def generate_investment_recommendations(self):
        """Generate investment recommendations based on the current cycle phase"""
        # Get most recent cycle phase
        current_phase = self.macro_data['Macro_Phase'].iloc[-1]
        current_score = self.macro_data['Cycle_Score_Smooth'].iloc[-1]
        momentum = self.macro_data['Cycle_Momentum'].iloc[-1]
        
        # Create recommendation
        recommendation = {
            'current_phase': current_phase,
            'cycle_score': round(float(current_score), 2),
            'momentum': round(float(momentum), 2),
            'risk_level': 'High' if current_phase in ['Crisis'] else 
                          'Medium' if current_phase in ['Late Cycle'] else 'Low',
            'allocation': self.allocations[current_phase],
            'tickers': self.asset_tickers
        }
        
        self.recommendation = recommendation
        return recommendation
    
    def plot_cycle_dashboard(self):
        """Create the main cycle score chart with market overlay"""
        d = self.macro_data
        
        # Create figure
        fig = go.Figure()
        
        # Add cycle score line
        fig.add_trace(
            go.Scatter(
                x=d.index, 
                y=d['Cycle_Score_Smooth'],
                name='Cycle Heat Score', 
                line=dict(color='#FF007F', width=1.5), 
                hovertemplate='%{x}<br>Score: %{y:.2f}<extra></extra>'
            )
        )
        
        # Add forecast line
        fig.add_trace(
            go.Scatter(
                x=d.index, 
                y=d['Forecast_Score'],
                name='Forecast (10-day avg)', 
                line=dict(color='#00FFEC', width=0.5, dash='dot'),
                hovertemplate='%{x}<br>Forecast: %{y:.2f}<extra></extra>'
            )
        )
        
        # Add recession shading
        for start, end in self.recessions:
            fig.add_vrect(
                x0=start, x1=end, 
                fillcolor='rgba(211, 211, 211, 0.3)', 
                layer='below', line_width=0.3,
                annotation_text="Recession",
                annotation_position="top left"
            )
            
        # Add threshold lines with pastel versions of colors
        fig.add_hline(y=5, line_dash='dash', line_color='#F27C7C',  # Soft pastel red (Crisis)
                     annotation_text='Crisis Threshold', annotation_position="top right")
        fig.add_hline(y=3, line_dash='dot', line_color='#D1A0D6',  # Soft pastel purple (Late Cycle)
                     annotation_text='Late Cycle', annotation_position="top right")
        fig.add_hline(y=-2, line_dash='dot', line_color='#A3D5A3',  # Pastel green (Early Recovery)
                     annotation_text='Early Recovery', annotation_position="bottom right")
            
        # Add market overlays
        market_tickers = ['SPY', 'VGK', 'FXI']
        market_labels = ['S&P 500', 'Europe', 'China']
    
        market_colors = ['#1F51FF',    # Neon blue (for SPY/S&P 500)
                         '#FF4500',    # Neon orange (for VGK/Europe)
                         '#FFFF00']    # Neon yellow (for FXI/China)
                
        for ticker, label, color in zip(market_tickers, market_labels, market_colors):
            if ticker in d.columns:
                fig.add_trace(
                    go.Scatter(
                        x=d.index, 
                        y=self.norm(d[ticker]), 
                        name=label, 
                        line=dict(width=1, color=color),
                        hovertemplate='%{x}<br>' + label + ': %{y:.2f}<extra></extra>'
                    )
                )
        
        # Layout improvements
        fig.update_layout(
            title={
                'text': "Macro Cycle Score (Ray Dalio Style) + Market Overlay",
                'font': {'size': 24, 'family': 'Arial, sans-serif'},
                'y': 0.95
            },
            height=600,
            template="plotly_dark",
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="center", 
                x=0.5
            ),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=80, b=40),
            xaxis_title="",
            yaxis_title="Z-Score / Composite Risk Index",
            yaxis=dict(
                tickfont=dict(size=12),
                title_font=dict(size=14)
            )
        )
        
        return fig
    
    def plot_indicators_chart(self):
        """Plot the key indicators in a separate chart"""
        d = self.macro_data
        
        # Create figure
        fig = go.Figure()
        


        # Main indicators with neon color palette
        indicators = {
            'Z_Yield':   ('Yield Curve', '#00FFFF'),  # Neon cyan blue: stability, finance
            'Z_VIX':     ('Volatility',  '#9400D3'),  # Neon purple: uncertainty, alert
            'Z_TIP':     ('Inflation',   '#00FF00'),  # Neon lime green: money, growth
            'Z_Credit':  ('Credit',      '#FF0000'),  # Neon red: risk, debt
            'Z_XLYXLP':  ('Consumer',    '#8A2BE2')   # Neon violet: lifestyle, mixed sectors
        }


        for indicator, (name, color) in indicators.items():
            fig.add_trace(
                go.Scatter(
                    x=d.index,
                    y=d[indicator],
                    name=name,
                    line=dict(color=color, width=1),
                    hovertemplate='%{x}<br>' + name + ': %{y:.2f}<extra></extra>'
                )
            )
        
        # Layout improvements
        fig.update_layout(
            title={
                'text': "Key Economic Indicators (Z-Scores)",
                'font': {'size': 20, 'family': 'Arial, sans-serif'},
                'y': 0.95
            },
            height=500,
            template="plotly_dark",
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="center", 
                x=0.5
            ),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=80, b=40),
            xaxis_title="",
            yaxis_title="Z-Score",
            yaxis=dict(
                tickfont=dict(size=12),
                title_font=dict(size=14),
                range=[-3, 3]  # Fixed range for better visualization of z-scores
            )
        )
        
        return fig
    
    def plot_allocation_chart(self):
        """Create a pie chart showing recommended allocation"""
        if not hasattr(self, 'recommendation'):
            return go.Figure()
        
        # Create allocation pie chart
        labels = list(self.recommendation['allocation'].keys())
        values = list(self.recommendation['allocation'].values())
        tickers = [self.recommendation['tickers'][asset] for asset in labels]
        
        # Create custom hover text with tickers
        hover_text = [f"{asset} ({ticker}): {value}%" 
                     for asset, ticker, value in zip(labels, tickers, values)]
        
        # Create figure with a better color scheme
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo='percent',
                hoverinfo='text',
                text=hover_text,
                marker=dict(
                    colors=px.colors.qualitative.Bold,
                    line=dict(color='white', width=2)
                ),
                hole=0.4
            )
        ])
        
        # Add phase indicator in the center
        current_phase = self.recommendation['current_phase']
        phase_color = self.cycle_colors[current_phase]
        
        fig.update_layout(
            title={
                'text': f"Current Recommended Allocation",
                'font': {'size': 20, 'family': 'Arial, sans-serif'},
                'y': 0.95
            },
            annotations=[
                dict(
                    text=f"{current_phase}<br>Phase",
                    x=0.5, y=0.5,
                    font=dict(size=16, color=phase_color),
                    showarrow=False
                )
            ],
            height=500,
            template="plotly_white",
            margin=dict(l=40, r=40, t=80, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
    
    def plot_all_weather_performance(self):
        """Plot the performance of All Weather Portfolio components"""
        # Get All Weather tickers that exist in our data
        all_weather = [ticker for ticker in self.all_weather if ticker in self.data.columns]
        
        if not all_weather:
            return go.Figure()
        
        # Normalize price data to starting point
        norm_data = self.data[all_weather].copy()
        for col in norm_data.columns:
            norm_data[col] = norm_data[col] / norm_data[col].iloc[0] * 100
        
        # Create equal-weighted portfolio
        norm_data['All_Weather'] = norm_data.mean(axis=1)
        
        # Plot with better colors
        fig = go.Figure()
        
        # Define nicer colors with better contrast
        colors = px.colors.qualitative.Bold
        
        # Plot individual components with thinner lines
        for i, ticker in enumerate(all_weather):
            fig.add_trace(
                go.Scatter(
                    x=norm_data.index,
                    y=norm_data[ticker],
                    name=ticker,
                    line=dict(width=1.5, color=colors[i % len(colors)]),
                    hovertemplate='%{x}<br>' + ticker + ': %{y:.2f}<extra></extra>'
                )
            )
        
        # Plot All Weather portfolio with thicker line for emphasis
        fig.add_trace(
            go.Scatter(
                x=norm_data.index,
                y=norm_data['All_Weather'],
                name='All Weather Portfolio',
                line=dict(width=3, color='black'),
                hovertemplate='%{x}<br>All Weather: %{y:.2f}<extra></extra>'
            )
        )
        
        # Add recession shading
        for start, end in self.recessions:
            fig.add_vrect(
                x0=start, x1=end, 
                fillcolor='rgba(200,200,200,0.3)', 
                layer='below', line_width=0
            )
        
        # Layout improvements
        fig.update_layout(
            title={
                'text': "All Weather Portfolio Performance",
                'font': {'size': 20, 'family': 'Arial, sans-serif'},
                'y': 0.95
            },
            height=500,
            template="plotly_white",
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="center", 
                x=0.5
            ),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=80, b=40),
            xaxis_title="",
            yaxis_title="Value (Start=100)",
            yaxis=dict(
                tickfont=dict(size=12),
                title_font=dict(size=14)
            )
        )
        
        return fig
    
    def create_dashboard_summary(self):
        """Create a clean summary of the current market phase and recommendations"""
        if not hasattr(self, 'recommendation'):
            return "Analysis not yet complete."
        
        rec = self.recommendation
        
        phase = rec['current_phase']
        score = rec['cycle_score']
        momentum = rec['momentum']
        risk = rec['risk_level']
        
        momentum_direction = "↑" if momentum > 0 else "↓"
        momentum_abs = abs(momentum)
        
        # Create colored text for risk levels
        risk_colors = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
        colored_risk = risk_colors.get(risk, risk)
        
        summary = f"""
        Current Market Analysis
        - Phase: {phase}  
        - Cycle Score: {score}  
        - Momentum: {momentum_direction} {momentum_abs:.2f}  
        - Risk Level: {colored_risk}
        
        Recommended Allocation:
        """
        
        for asset, allocation in rec['allocation'].items():
            ticker = rec['tickers'][asset]
            summary += f"- {asset}: {allocation}% ({ticker})\n"
        f"""

        """
        
        return summary
    
    def run_analysis(self):
        """Run the full analysis pipeline"""
        try:
            self.download_data()
            self.calculate_macro_indicators()
            
            try:
                self.generate_investment_recommendations()
            except Exception as e:
                print(f"Warning: Could not generate recommendations: {e}")
                self.recommendation = {
                    'current_phase': 'Unknown',
                    'cycle_score': 0,
                    'momentum': 0,
                    'risk_level': 'Medium',
                    'allocation': {'Equities': 30, 'Long-Term Bonds': 30, 'Intermediate Bonds': 15, 'Gold': 15, 'Commodities': 10},
                    'tickers': self.asset_tickers
                }
            
            # Create charts
            cycle_chart = self.plot_cycle_dashboard()
            indicators_chart = self.plot_indicators_chart()
            allocation_chart = self.plot_allocation_chart()
            all_weather_chart = self.plot_all_weather_performance()
            summary = self.create_dashboard_summary()
            
            # Export data for future use
            try:
                self.macro_data[['Cycle_Score_Smooth', 'Forecast_Score', 'Macro_Phase', 'Cycle_Momentum']].to_csv("dalio_macro_analysis.csv")
            except Exception as e:
                print(f"Warning: Could not export data: {e}")
            
            # Return analysis results
            return {
                'cycle_chart': cycle_chart,
                'indicators_chart': indicators_chart,
                'allocation_chart': allocation_chart,
                'all_weather_chart': all_weather_chart,
                'summary': summary,
                'macro_data': self.macro_data
            }
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal results to avoid crashing
            return {
                'cycle_chart': go.Figure(),
                'indicators_chart': go.Figure(),
                'allocation_chart': go.Figure(),
                'all_weather_chart': go.Figure(),
                'summary': f"Error in analysis: {e}",
                'macro_data': pd.DataFrame()
            }

def run_dalio_analysis(start_date):
    """Function to run for Gradio interface"""
    try:
        start_date = start_date or "2000-01-01"
        analyzer = SimplifiedDalioAnalysis(start_date=start_date)
        results = analyzer.run_analysis()
        
        # Return all plots and summary
        return (
            results['summary'],
            results['cycle_chart'],
            results['indicators_chart'], 
            results['allocation_chart'],
            results['all_weather_chart']
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {e}")
        print(error_details)
        return (
            f"Error in analysis: {e}",
            go.Figure(),
            go.Figure(),
            go.Figure(),
            go.Figure()
        )

# Create Gradio app with improved styling
def create_gradio_app():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🔄 Ray Dalio's Economic Cycle Analysis
        
        Analyze current market conditions using Ray Dalio's economic cycle framework. This tool helps identify:
        - Which phase of the economic cycle we're in
        - Current market risks and opportunities
        - Optimal asset allocation based on Dalio's principles
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                start_date_input = gr.Textbox(
                    label="Start Date (YYYY-MM-DD)", 
                    value="2000-01-01",
                    info="Data will be analyzed from this date to present"
                )
                analyze_btn = gr.Button("Run Analysis", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                summary_md = gr.Markdown(elem_id="summary_container")
        
        with gr.Tab("📊 Economic Cycle"):
            cycle_plot = gr.Plot(elem_id="cycle_plot")
            gr.Markdown("""
            **About this chart:** This visualizes the composite economic cycle score with Dalio's methodology.
            Higher scores indicate increased market stress. The horizontal lines show thresholds for different economic phases.
            """)
            
        with gr.Tab("📈 Key Indicators"):
            indicators_plot = gr.Plot(elem_id="indicators_plot")
            gr.Markdown("""
            **Key indicators explained:**
            - **Yield Curve**: The difference between long and short-term rates (flattening indicates late cycle)
            - **Volatility**: Market volatility (VIX) - higher values mean more stress
            - **Inflation**: TIPS vs Treasury spread - indicates inflation expectations
            - **Credit**: Credit spread between high yield and investment grade bonds
            - **Consumer**: Consumer discretionary vs staples - indicates economic confidence
            """)
            
        with gr.Tab("🥧 Recommended Allocation"):
            allocation_plot = gr.Plot(elem_id="allocation_plot")
            gr.Markdown("""
            **About the allocation:** This shows Dalio's recommended asset allocation based on the current phase.
            The center indicates which phase we're currently in. Hover over each segment to see the recommended percentage
            and corresponding ETF.
            """)
            
        with gr.Tab("💰 All Weather Performance"):
            allweather_plot = gr.Plot(elem_id="allweather_plot")
            gr.Markdown("""
            **About the All Weather Portfolio:** This shows how Dalio's balanced portfolio components perform over time.
            The portfolio includes stocks (SPY), long-term bonds (TLT), intermediate bonds (IEF), gold (GLD), and commodities (DBC).
            This allocation aims to perform well across all economic environments.
            """)
            
        analyze_btn.click(
            fn=run_dalio_analysis,
            inputs=[start_date_input],
            outputs=[summary_md, cycle_plot, indicators_plot, allocation_plot, allweather_plot]
        )
        
        gr.Markdown("""
        ### Understanding Dalio's Framework
        
        Ray Dalio, founder of Bridgewater Associates (the world's largest hedge fund), developed a comprehensive approach to understanding economic cycles. His framework divides the economy into four distinct phases:
        
        - **Early Recovery**: Economy is improving from a downturn, policy is accommodative
        - **Mid Cycle**: Steady growth with moderate inflation and tightening policy
        - **Late Cycle**: Growth slowing, inflation rising, policy becoming restrictive
        - **Crisis**: Economic contraction, high volatility, policy rapidly loosening
        
        The tool analyzes multiple economic indicators to determine the current phase and offers portfolio recommendations based on historical performance during similar periods.
        """)
    
    return app

# CSS for better styling
custom_css = """
<style>
#summary_container {
    background-color: #f5f5f5;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
"""

# Main execution
if __name__ == "__main__":
    app = create_gradio_app()
    
    app.launch()
