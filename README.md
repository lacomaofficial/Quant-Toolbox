# Quant-Lab

A collection of quantitative finance tools for market screening, technical analysis, and macro regime detection. Built with Gradio, Yahoo Finance, and Hugging Face models.

For related models and spaces, see [Hugging Face](https://huggingface.co/JayLacoma).

<br>

## Structure

```
Quant-Toolbox/
├── apps/
│   ├── fundamental.py         # Financial statement analysis
│   ├── sentiment.py           # News-based market sentiment
│   ├── technical.py           # Technical indicators and chart patterns
│   └── macro/
│       ├── economic_cycle.py  # Ray Dalio economic cycle framework
│       └── global_market.py   # US vs global market analysis
└── utils/
    ├── stock_screen.py        # Yahoo Finance API wrapper and data extraction
    └── technical_scan.py      # Indicator functions (RSI, Stochastic, CCI, Bollinger)
```

<br>

## Apps

| App | Description |
|-----|-------------|
| **Fundamental Analysis** | Compare companies by financial metrics and fundamentals |
| **Technical Analysis** | Trading signals via RSI, Stochastic, CCI, and Bollinger Bands |
| **News Sentiment** | Market sentiment from news articles and press releases |
| **Economic Cycle** | Regime detection using Ray Dalio's economic framework |
| **Global Market** | US equities in context of international market dynamics |
| **Stock Screener** | Yahoo Finance API scraper with multi-category extraction |
| **Signal Scanner** | Multi-ticker technical signal generation (BUY/SELL) |



<br>

## Related

- [`Time-Series-Models`](https://github.com/lacomaofficial/Transformer-Time-Series-Model) — Deep learning for time series forecasting
- [`YC Hedge Fund`](https://lacomaofficial.github.io) — Research group homepage

<br>

## License

MIT — use freely for research and commercial purposes.


