# stock_screen.py



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







class YahooFinanceAPI:
    """Fetch market data using Yahoo Finance's internal screener API."""

    API_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"

    SCREENERS = {
        "Most Active": {"primary": "MOST_ACTIVES", "fallback": "most_actives"},
        "Top Gainers": {"primary": "DAY_GAINERS", "fallback": "day_gainers"},
        "Top Losers": {"primary": "DAY_LOSERS", "fallback": "day_losers"},
        "52 Week Gainers": {
            "primary": "FIFTY_TWO_WK_GAINERS",
            "fallback": "fifty_two_wk_gainers",
        },
        "52 Week Losers": {
            "primary": "FIFTY_TWO_WK_LOSERS",
            "fallback": "fifty_two_wk_losers",
        },
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://finance.yahoo.com",
                "Referer": "https://finance.yahoo.com/",
            }
        )

    def fetch_screener(
        self, screener_config: dict, count: int = 100
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a specific screener category with fallback support."""
        for screener_id in [
            screener_config["primary"],
            screener_config.get("fallback"),
        ]:
            if not screener_id:
                continue
            try:
                params = {
                    "scrIds": screener_id,
                    "count": count,
                    "fmt": "json",
                    "lang": "en-US",
                    "region": "US",
                }
                response = self.session.get(self.API_URL, params=params, timeout=15)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                data = response.json()

                if (
                    "finance" not in data
                    or "result" not in data["finance"]
                    or not data["finance"]["result"]
                ):
                    continue

                quotes = data["finance"]["result"][0].get("quotes", [])
                if not quotes:
                    continue

                df = pd.DataFrame(quotes)
                return self._process_dataframe(df)
            except Exception:
                continue
        return None

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select, rename, and format columns for display."""
        column_map = {
            "symbol": "Symbol",
            "shortName": "Name",
            "regularMarketPrice": "Price",
            "regularMarketChange": "Change",
            "regularMarketChangePercent": "Change %",
            "regularMarketVolume": "Volume",
            "marketCap": "Market Cap",
            "fiftyTwoWeekHigh": "52W High",
            "fiftyTwoWeekLow": "52W Low",
        }
        available_cols = {k: v for k, v in column_map.items() if k in df.columns}
        if not available_cols:
            return pd.DataFrame()

        df = df.rename(columns=available_cols)[list(available_cols.values())]
        self._format_columns(df)
        return df

    @staticmethod
    def _format_columns(df: pd.DataFrame) -> None:
        """Format numeric columns for human-readable display (in-place)."""
        if "Change %" in df.columns:
            df["Change %"] = df["Change %"].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            )
        for col in ["Price", "52W High", "52W Low"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"${x:.2f}"
                    if pd.notna(x) and isinstance(x, (int, float))
                    else "N/A"
                )
        for col in ["Volume", "Market Cap"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"{int(x):,}"
                    if pd.notna(x) and isinstance(x, (int, float))
                    else "N/A"
                )

    def scrape_all(self, count: int = 100, quiet: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured screener categories."""
        results = {}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not quiet:
            print(f"\n{'='*70}")
            print(f"📊 Yahoo Finance Market Analysis - {timestamp}")
            print(f"{'='*70}\n")

        for category, config in self.SCREENERS.items():
            if not quiet:
                print(f"📈 Fetching {category}...")
            
            df = self.fetch_screener(config, count=count)

            if df is not None and not df.empty:
                results[category] = df
                if not quiet:
                    print(f"   ✓ Retrieved {len(df)} stocks")
            
            time.sleep(0.5)

        return results


def display_results(results: Dict[str, pd.DataFrame], max_rows: int = 5) -> None:
    """Display results in formatted console output."""
    if not results:
        print("⚠️  No data to display.")
        return

    print(f"\n{'='*70}")
    print("📊 MARKET SUMMARY")
    print(f"{'='*70}\n")

    key_cols = ["Symbol", "Name", "Price", "Change", "Change %", "Volume"]

    for category, df in results.items():
        print(f"📈 {category}")
        print("-" * 70)
        available_cols = [c for c in key_cols if c in df.columns]
        display_df = df[available_cols] if available_cols else df
        print(display_df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"\n   ... and {len(df) - max_rows} more rows")
        print()


def save_results(
    results: Dict[str, pd.DataFrame],
    output_dir: str,
    save_csv: bool,
    save_md: bool,
    timestamp: str,
    quiet: bool = False,
) -> List[str]:
    """Save results to specified formats and directory."""
    date_str = datetime.now().strftime("%Y%m%d")
    saved_files = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results:
        return saved_files

    if not quiet:
        print(f"\n{'='*70}")
        print("💾 Saving data...")
        print("-" * 70)

    md_content = f"# Yahoo Finance Market Analysis\n\n**Generated:** {timestamp}\n\n---\n\n"

    for category, df in results.items():
        safe_name = category.replace(" ", "_").lower()

        if save_csv:
            csv_path = os.path.join(output_dir, f"{safe_name}_{date_str}.csv")
            df_raw = df.copy()
            for col in df_raw.columns:
                if df_raw[col].dtype == "object":
                    df_raw[col] = df_raw[col].astype(str).str.replace(r"[$%,]", "", regex=True)
            df_raw.to_csv(csv_path, index=False)
            saved_files.append(csv_path)
            if not quiet:
                print(f"  ✅ Saved CSV: {csv_path}")

        if save_md:
            md_content += f"## {category}\n\n{df.to_markdown(index=False)}\n\n---\n\n"

    if save_md:
        md_path = os.path.join(output_dir, f"yahoo_markets_{date_str}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        saved_files.append(md_path)
        if not quiet:
            print(f"  ✅ Saved Markdown: {md_path}")

    return saved_files


def run_extraction(
    count: int = 100,
    output_dir: str = "./market_data",
    output_format: str = "both",
    quiet: bool = False
):
    """
    Core execution logic.
    
    Parameters:
    - count: Number of tickers per category.
    - output_dir: Folder to save files.
    - output_format: 'both', 'csv', or 'md'.
    - quiet: Suppress detailed logs.
    """
    if output_format not in ["both", "csv", "md"]:
        raise ValueError("output_format must be 'both', 'csv', or 'md'")

    save_csv = output_format in ["both", "csv"]
    save_md = output_format in ["both", "md"]

    api = YahooFinanceAPI()
    results = api.scrape_all(count=count, quiet=quiet)

    if not quiet:
        display_results(results, max_rows=5)

    if results:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        saved = save_results(
            results, output_dir=output_dir, save_csv=save_csv, 
            save_md=save_md, timestamp=timestamp, quiet=quiet
        )
        
        msg = f"✅ Complete. {len(saved)} files saved to '{output_dir}'"
        print(msg if quiet else f"\n{'='*70}\n{msg}\n{'='*70}")
    else:
        if not quiet:
            print("\n❌ No data retrieved.")
        sys.exit(1)


'''
# Usage example
from yft_max import run_extraction
run_extraction(
    count=100,              # Change this to fetch more/fewer stocks (e.g., 50, 200)
    output_dir="./docs", # Change this to save files elsewhere
    output_format="md",   # Options: "both", "csv", or "md"
    quiet=False             # Set to True to hide detailed logs
)
'''

