import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests
from typing import Dict, Optional

class MacroAnalyzer:
    def __init__(self):
        # Define sector investment rules
        self.sector_rules = {
            "Technology": {
                "Interest Rates": "Low (✅)",
                "Growth": "Positive (✅)",
                "Inflation": "Neutral (🟡)",
                "Best Conditions": "Low rates, strong GDP growth"
            },
            "Industrials": {
                "Interest Rates": "Neutral (🟡)",
                "Growth": "Positive (✅)",
                "Inflation": "Positive (✅)",
                "Best Conditions": "Strong GDP, moderate inflation"
            },
            "Energy": {
                "Interest Rates": "Neutral (🟡)",
                "Growth": "Neutral (🟡)",
                "Inflation": "Very High (🚀)",
                "Best Conditions": "High inflation, geopolitical risk"
            },
            "Consumer Discretionary": {
                "Interest Rates": "Low (✅)",
                "Growth": "Positive (✅)",
                "Inflation": "Neutral (🟡)",
                "Best Conditions": "Low rates, strong GDP"
            },
            "Consumer Staples": {
                "Interest Rates": "Positive (✅)",
                "Growth": "Negative (🚨)",
                "Inflation": "Positive (✅)",
                "Best Conditions": "High inflation, slow growth"
            },
            "Financials": {
                "Interest Rates": "Positive (✅)",
                "Growth": "Neutral (🟡)",
                "Inflation": "Neutral (🟡)",
                "Best Conditions": "Rising rates, stable GDP"
            },
            "Healthcare": {
                "Interest Rates": "Positive (✅)",
                "Growth": "Negative (🚨)",
                "Inflation": "Positive (✅)",
                "Best Conditions": "Recession, aging population"
            },
            "Utilities": {
                "Interest Rates": "Positive (✅)",
                "Growth": "Negative (🚨)",
                "Inflation": "Positive (✅)",
                "Best Conditions": "High inflation, low growth"
            },
            "Real Estate": {
                "Interest Rates": "Low (✅)",
                "Growth": "Neutral (🟡)",
                "Inflation": "Positive (✅)",
                "Best Conditions": "Low rates, moderate inflation"
            },
            "Materials": {
                "Interest Rates": "Neutral (🟡)",
                "Growth": "Positive (✅)",
                "Inflation": "High (🚨)",
                "Best Conditions": "Economic expansion, rising prices"
            },
            "Communication Services": {
                "Interest Rates": "Low (✅)",
                "Growth": "Positive (✅)",
                "Inflation": "Neutral (🟡)",
                "Best Conditions": "Low rates, tech-driven growth"
            }
        }
        
        # Enhanced US-focused ticker definitions
        self.tickers = {
            "United States": {
                "Rates": {
                    "3M Yield": "^IRX",
                    "2Y Yield": "^FVX",
                    "10Y Yield": "^TNX",
                    "30Y Yield": "^TYX"
                },
                "Equities": {
                    "S&P 500": "^GSPC",
                    "Nasdaq": "^IXIC",
                    "Dow Jones": "^DJI",
                    "Russell 2000": "^RUT"
                },
                "Volatility": {
                    "VIX": "^VIX",
                    "VXN": "^VXN"  # Nasdaq volatility
                },
                "Commodities": {
                    "Gold": "GC=F",
                    "Oil": "CL=F",
                    "Copper": "HG=F"
                },
                "Credit": {
                    "High Yield Spread": "HYG",
                    "Investment Grade": "LQD"
                },
                "Sectors": {
                    "XLK (Tech)": "XLK",
                    "XLI (Industrials)": "XLI",
                    "XLE (Energy)": "XLE",
                    "XLY (Consumer Disc)": "XLY",
                    "XLP (Staples)": "XLP",
                    "XLF (Financials)": "XLF",
                    "XLV (Healthcare)": "XLV",
                    "XLU (Utilities)": "XLU",
                    "XLRE (Real Estate)": "XLRE",
                    "XLB (Materials)": "XLB",
                    "XLC (Comm Services)": "XLC"
                }
            },
            # International markets for comparison (using ETFs for better reliability)
            "Europe": {
                "Equities": {"Euro Stoxx 50": "FEZ"}
            },
            "China": {
                "Equities": {"China Large-Cap": "FXI"}
            },
            "Japan": {
                "Equities": {"Japan": "EWJ"}
            },
            "South Korea": {
                "Equities": {"South Korea": "EWY"}
            },
            "India": {
                "Equities": {"India": "INDA"}
            }
        }
        
        # Alternative data sources for bonds
        self.bond_apis = {
            "US 10Y": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
            "Germany 10Y": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01DEM156N",
            "Japan 10Y": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01JPM156N"
        }
        
        self.data = {}
        self.regimes = {}
    
    def download_with_retry(self, ticker: str, asset_name: str) -> Optional[pd.Series]:
        """Enhanced download with multiple fallback options"""
        # First try Yahoo Finance
        for attempt in range(3):
            try:
                df = yf.download(ticker, period="1y", progress=False)
                if not df.empty:
                    return df['Close']
            except:
                time.sleep(1)
        
        # If Yahoo fails, try alternative sources for bonds
        if "Yield" in asset_name or "YR" in asset_name:
            return self.download_bond_data(asset_name)
        
        return None
    
    def download_bond_data(self, bond_name: str) -> Optional[pd.Series]:
        """Alternative method to get bond yields when Yahoo fails"""
        try:
            if bond_name in self.bond_apis:
                # Use FRED data for bonds
                df = pd.read_csv(self.bond_apis[bond_name], parse_dates=['DATE'])
                df.set_index('DATE', inplace=True)
                return df.iloc[:, 0]  # Return the first column
        except Exception as e:
            print(f"Failed to get bond data from alternative source: {e}")
            return None
    
    def download_all_data(self):
        """Download data for all regions"""
        for region in self.tickers.keys():
            self.download_region_data(region)
    
    def download_region_data(self, region: str) -> pd.DataFrame:
        """Download all data for a region with robust error handling"""
        if region not in self.tickers:
            return pd.DataFrame()
            
        region_data = pd.DataFrame()
        
        for category, assets in self.tickers[region].items():
            for name, ticker in assets.items():
                series = self.download_with_retry(ticker, name)
                if series is not None:
                    region_data[f"{category}_{name}"] = series
        
        if not region_data.empty:
            # Clean and process data
            region_data = region_data.ffill().bfill()
            
            # Calculate important derivatives (US only)
            if region == "United States":
                if "Rates_10Y Yield" in region_data and "Rates_2Y Yield" in region_data:
                    region_data["Yield_Spread"] = region_data["Rates_10Y Yield"] - region_data["Rates_2Y Yield"]
                    region_data["Yield_Inverted"] = region_data["Yield_Spread"] < 0
                
                if "Equities_S&P 500" in region_data:
                    region_data["Equity_3M_Return"] = region_data["Equities_S&P 500"].pct_change(63)
                    
                # Calculate VIX relative to its historical range
                if "Volatility_VIX" in region_data:
                    vix = region_data["Volatility_VIX"]
                    region_data["VIX_Percentile"] = vix.rank(pct=True) * 100
                    
                # Calculate sector relative performance
                for col in region_data.columns:
                    if "Sectors_" in col:
                        region_data[f"{col}_RelPerf"] = region_data[col].pct_change(63) - region_data["Equities_S&P 500"].pct_change(63)
            
            self.data[region] = region_data
        
        return region_data
    
    def classify_regime(self, region: str) -> pd.DataFrame:
        """Classify current economic regime with fallback logic"""
        if region not in self.data:
            return pd.DataFrame()
            
        df = self.data[region]
        conditions = {}
        
        # Interest Rate Classification with fallbacks (US only)
        if region == "United States":
            if "Rates_10Y Yield" in df:
                rate = df["Rates_10Y Yield"].iloc[-1]
                conditions["Interest Rates"] = np.select(
                    [rate < 1.5, (1.5 <= rate) & (rate <= 3.5), rate > 3.5],
                    ["Low (✅)", "Neutral (🟡)", "High (🚨)"],
                    "Unknown"
                )
            elif "Credit_High Yield Spread" in df:
                # Use credit spread as proxy if yields unavailable
                spread = df["Credit_High Yield Spread"].iloc[-1]
                conditions["Interest Rates"] = np.select(
                    [spread < 3, (3 <= spread) & (spread <= 5), spread > 5],
                    ["Low (✅)", "Neutral (🟡)", "High (🚨)"],
                    "Unknown"
                )
        
        # Growth Classification
        equity_col = next((col for col in df.columns if "Equities_" in col), None)
        if equity_col:
            growth = df[equity_col].pct_change(63).iloc[-1]
            conditions["Growth"] = np.select(
                [growth < -0.05, (-0.05 <= growth) & (growth <= 0.05), growth > 0.05],
                ["Negative (🚨)", "Neutral (🟡)", "Positive (✅)"],
                "Unknown"
            )
        
        # Inflation Classification (US only)
        if region == "United States":
            if "Commodities_Gold" in df and "Commodities_Oil" in df:
                gold_return = df["Commodities_Gold"].pct_change(63).iloc[-1]
                oil_return = df["Commodities_Oil"].pct_change(63).iloc[-1]
                commodity_return = (gold_return + oil_return) / 2
                conditions["Inflation"] = np.select(
                    [commodity_return < -0.05, (-0.05 <= commodity_return) & (commodity_return <= 0.05), commodity_return > 0.05],
                    ["Low (✅)", "Neutral (🟡)", "High (🚨)"],
                    "Unknown"
                )
        
        # Risk Sentiment Classification (US only)
        if region == "United States":
            if "Volatility_VIX" in df:
                vix = df["Volatility_VIX"].iloc[-1]
                conditions["Market Sentiment"] = np.select(
                    [vix < 15, (15 <= vix) & (vix <= 25), vix > 25],
                    ["Complacent (🟢)", "Neutral (🟡)", "Fearful (🔴)"],
                    "Unknown"
                )
        
        # Recession Risk Assessment (US only)
        if region == "United States":
            recession_flags = []
            if "Yield_Inverted" in df and df["Yield_Inverted"].iloc[-1]:
                recession_flags.append("Yield Curve Inverted")
            
            conditions["Recession Risk"] = "High (🚨)" if recession_flags else "Low (✅)"
            conditions["Recession Flags"] = ", ".join(recession_flags) if recession_flags else "None"
        
        regime_df = pd.DataFrame([conditions], index=[df.index[-1]])
        self.regimes[region] = regime_df
        return regime_df
    
    def evaluate_sectors(self, region: str) -> pd.DataFrame:
        """Generate sector recommendations with error handling"""
        if region not in self.regimes:
            return pd.DataFrame()
            
        current_regime = self.regimes[region].iloc[0]
        recommendations = []
        
        for sector, rules in self.sector_rules.items():
            match_score = 0
            total_factors = 0
            conditions_met = []
            
            for factor, expected in rules.items():
                if factor in current_regime:
                    total_factors += 1
                    if current_regime[factor] == expected:
                        match_score += 1
                        conditions_met.append(f"✓ {factor}: {current_regime[factor]}")
                    else:
                        conditions_met.append(f"✗ {factor}: {current_regime[factor]} (expected {expected})")
            
            confidence = match_score / total_factors if total_factors > 0 else 0
            
            recommendations.append({
                "Sector": sector,
                "Match": f"{match_score}/{total_factors}",
                "Confidence": f"{confidence:.0%}",
                "Recommendation": "Strong Buy" if confidence >= 0.8 else
                                "Buy" if confidence >= 0.6 else
                                "Neutral" if confidence >= 0.4 else
                                "Caution",
                "Conditions": "\n".join(conditions_met),
                "Best Environment": rules.get("Best Conditions", "")
            })
        
        return pd.DataFrame(recommendations).sort_values("Confidence", ascending=False)
    
    def create_yield_curve_plot(self, region: str) -> go.Figure:
        """Create yield curve plot with available data"""
        if region not in self.data:
            return go.Figure()
            
        df = self.data[region]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        has_yield_data = False
        
        # Add available yield curves (US only)
        if region == "United States":
            yield_tenors = []
            for col in df.columns:
                if "Rates_" in col and "Yield" in col:
                    tenor = col.replace("Rates_", "")
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df[col], name=tenor),
                        secondary_y=False
                    )
                    yield_tenors.append(tenor)
                    has_yield_data = True
            
            # Add spread if both yields available
            if "Yield_Spread" in df:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["Yield_Spread"], name="10Y-2Y Spread", line=dict(dash="dot")),
                    secondary_y=True
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", secondary_y=True)
        
        if not has_yield_data:
            # If no yield data, show message
            fig.add_annotation(
                text="Yield data not available" if region != "United States" else "US Yield data loading...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig.update_layout(
            title=f"{region} Yield Curve Dynamics" if has_yield_data else f"{region} Yield Data",
            yaxis_title="Yield (%)",
            hovermode="x unified",
            template="plotly_white"
        )
        
        if "Yield_Spread" in df:
            fig.update_yaxes(title_text="Spread (bps)", secondary_y=True)
        
        return fig
    
    def create_market_plot(self, region: str) -> go.Figure:
        """Enhanced market overview plot with multiple indicators"""
        if region not in self.data:
            return go.Figure()
            
        df = self.data[region]
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.05,
                          row_heights=[0.7, 0.3],
                          specs=[[{"secondary_y": True}], [{}]])
        
        has_data = False
        
        # Add equity index (primary)
        equity_col = next((col for col in df.columns if "Equities_" in col), None)
        if equity_col:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[equity_col], 
                          name=equity_col.replace("Equities_", ""),
                          line=dict(color='#636EFA', width=2)),
                row=1, col=1, secondary_y=False
            )
            
            # Add 50-day moving average
            ma50 = df[equity_col].rolling(50).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=ma50,
                          name="50-Day MA",
                          line=dict(color='#FFA15A', width=1, dash='dot')),
                row=1, col=1, secondary_y=False
            )
            has_data = True
        
        # Add volatility index (secondary, US only)
        if region == "United States":
            vix_col = next((col for col in df.columns if "Volatility_" in col), None)
            if vix_col:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[vix_col],
                              name="Volatility Index",
                              line=dict(color='#EF553B', width=1),
                              fill='tozeroy',
                              fillcolor='rgba(239, 85, 59, 0.1)'),
                    row=2, col=1
                )
                
                # Add volatility thresholds
                fig.add_hline(y=20, line_dash="dash", line_color="orange", 
                             row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="red",
                             row=2, col=1)
                has_data = True
        
        if not has_data:
            fig.add_annotation(
                text="Market data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig.update_layout(
            title=f"{region} Market Overview" if has_data else f"{region} Market Data",
            template="plotly_dark",
            hovermode="x unified",
            height=600,
            margin=dict(t=80, b=60),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Index Value", row=1, col=1)
        fig.update_yaxes(title_text="Volatility" if region == "United States" else " ", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
    
    def create_international_comparison(self) -> go.Figure:
        """Create comparison plot of international equity markets"""
        comparison_data = []
        
        for region in self.tickers.keys():
            if region in self.data:
                df = self.data[region]
                equity_col = next((col for col in df.columns if "Equities_" in col), None)
                if equity_col:
                    # Normalize to 100 for comparison
                    normalized = (df[equity_col] / df[equity_col].iloc[0] * 100).rename(region)
                    comparison_data.append(normalized)
        
        if not comparison_data:
            return go.Figure()
            
        comparison_df = pd.concat(comparison_data, axis=1).dropna()
        
        fig = go.Figure()
        
        # Color palette for regions
        colors = px.colors.qualitative.Plotly
        
        for i, region in enumerate(comparison_df.columns):
            fig.add_trace(
                go.Scatter(x=comparison_df.index, y=comparison_df[region],
                          name=region,
                          line=dict(color=colors[i], width=2),
                          hovertemplate=f"{region}: %{{y:.1f}}")
            )
        
        fig.update_layout(
            title="<b>International Equity Markets Comparison</b> (Normalized to 100 at start)",
            yaxis_title="Performance (100 = Starting Value)",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            margin=dict(t=80, b=60),
            annotations=[
                dict(
                    x=0.5, y=1.1,
                    xref="paper", yref="paper",
                    text="<b>Interpretation:</b> Shows relative performance of major markets. All indices normalized to 100 at start of period.",
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )
        
        return fig

# Initialize analyzer
analyzer = MacroAnalyzer()

def analyze_all():
    """Run analysis for US and international markets"""
    try:
        # Download all data
        analyzer.download_all_data()
        
        # Classify US regime
        analyzer.classify_regime("United States")
        
        # Get US results
        regime = analyzer.regimes.get("United States", pd.DataFrame()).T.reset_index()
        regime.columns = ['Factor', 'Status']
        
        sectors = analyzer.evaluate_sectors("United States")
        us_yield_plot = analyzer.create_yield_curve_plot("United States")
        us_market_plot = analyzer.create_market_plot("United States")
        
        # Create international comparison
        intl_comparison = analyzer.create_international_comparison()
        
        return [
            regime,
            sectors,
            us_yield_plot,
            us_market_plot,
            intl_comparison
        ]
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return [
            pd.DataFrame({"Factor": ["Error"], "Status": [str(e)[:100]]}),
            pd.DataFrame({"Sector": ["Error"], "Recommendation": [str(e)[:100]]}),
            go.Figure(),
            go.Figure(),
            go.Figure()
        ]

# Create Gradio interface
with gr.Blocks(title="Global Macro Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌍 Global Macro Dashboard
    ### US market analysis with international comparisons
    """)
    
    with gr.Row():
        update_btn = gr.Button("Update Data", variant="primary")
    
    with gr.Tabs():
        with gr.Tab("🇺🇸 US Market"):
            gr.Markdown("### Economic Conditions")
            regime_table = gr.DataFrame(label="Regime Classification")
    
            
            gr.Markdown("### Yield Curve Analysis")
            yield_plot = gr.Plot()
            
            gr.Markdown("### Market Overview")
            market_plot = gr.Plot()

            gr.Markdown("### Sector Recommendations")
            sector_table = gr.DataFrame(label="Top Opportunities")
        
        with gr.Tab("🌐 International Comparison"):
            gr.Markdown("### Equity Markets Performance (Normalized)")
            intl_comparison = gr.Plot()
            gr.Markdown("""
            **How to interpret this chart:**
            - All indices are normalized to 100 at the start of the period
            - The chart shows relative performance across different markets
            - Steeper lines indicate stronger performance
            - Using ETFs for international markets for better reliability
            """)

    
    update_btn.click(analyze_all, outputs=[regime_table, sector_table, yield_plot, market_plot, intl_comparison])
    demo.load(analyze_all, outputs=[regime_table, sector_table, yield_plot, market_plot, intl_comparison])

if __name__ == "__main__":
    demo.launch()