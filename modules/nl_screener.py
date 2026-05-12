"""
nl_screener.py — Natural Language Stock Screener for EquityLens.

Converts plain English queries into structured stock filters using
the Claude API, then applies those filters across a stock universe.

Example queries:
- "Show me profitable tech stocks with low debt and strong momentum"
- "Find undervalued small cap stocks with high revenue growth"
- "Give me defensive stocks with high dividends and low beta"
"""

import anthropic
import json
import yfinance as yf
import pandas as pd
from modules.screener import screen_ticker
from modules.fundamentals import analyze_ticker


# S&P 500 sample universe — expandable
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "BRK-B", "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD",
    "CVX", "MRK", "ABBV", "PEP", "KO", "AVGO", "COST", "MU",
    "AMD", "INTC", "CRM", "ADBE", "NFLX", "QCOM", "TXN",
    "WMT", "DIS", "BA", "GS", "MS", "C", "BAC", "WFC",
    "XOM", "CAT", "MMM", "GE", "F", "GM", "UBER", "LYFT"
]


def extract_filters(query: str) -> dict:
    """
    Use Claude API to extract structured filters from plain English.

    Sends the user's natural language query to Claude and asks it
    to return a JSON object with specific filter criteria.

    Args:
        query: Plain English screening query e.g.
               "profitable tech stocks with low debt"

    Returns:
        Dictionary of filter criteria with operator and value.
    """
    client = anthropic.Anthropic()

    system_prompt = """You are a financial data extraction assistant.
    
Your job is to convert plain English stock screening queries into 
structured JSON filters. You must respond with ONLY valid JSON — 
no explanation, no markdown, no backticks. Just the raw JSON object.

Available filter fields and their meanings:
- pe_ratio: Price to earnings ratio (lower = cheaper valuation)
- gross_margin: Gross profit margin as percentage (higher = better)
- revenue_cagr: 3-year revenue growth rate as percentage
- debt_equity: Debt to equity ratio (lower = less debt)
- current_ratio: Current assets / liabilities (higher = more liquid)
- beta: Market sensitivity (lower = more defensive)
- rsi: Relative strength index 0-100 (30-70 = healthy momentum)
- ma_signal: Moving average signal ("golden_cross" or "death_cross")
- market_cap: Company size in dollars

For each filter, return an object with:
- "operator": one of "gt" (greater than), "lt" (less than), "eq" (equals)
- "value": the threshold number or string

Example output for "profitable tech stocks with low debt":
{
  "gross_margin": {"operator": "gt", "value": 40},
  "debt_equity": {"operator": "lt", "value": 50},
  "revenue_cagr": {"operator": "gt", "value": 10}
}

Only include filters that are clearly implied by the query.
Do not add filters that weren't requested."""

    message = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 500,
        system     = system_prompt,
        messages   = [{"role": "user", "content": query}]
    )

    # Extract the text response
    response_text = message.content[0].text.strip()

    try:
        filters = json.loads(response_text)
        return filters
    except json.JSONDecodeError:
        print(f"[Error] Could not parse filters: {response_text}")
        return {}


def apply_filters(ticker: str, filters: dict) -> tuple:
    """
    Apply extracted filters to a single ticker.

    Fetches fundamental and technical data for the ticker and
    checks whether it passes all filter criteria.

    Args:
        ticker:  Stock symbol e.g. "AAPL"
        filters: Dictionary of filter criteria from extract_filters()

    Returns:
        Tuple of (passes: bool, data: dict)
    """
    try:
        # Fetch fundamental data
        fundamentals = analyze_ticker(ticker)
        screener     = screen_ticker(ticker)

        # Get market cap separately
        info       = yf.Ticker(ticker).info
        market_cap = info.get("marketCap", 0)

        # Combine all data into one flat dictionary for filtering
        data = {
            "ticker":       ticker,
            "pe_ratio":     fundamentals.get("pe_ratio"),
            "gross_margin": fundamentals.get("gross_margin"),
            "revenue_cagr": fundamentals.get("revenue_cagr"),
            "debt_equity":  fundamentals.get("debt_equity"),
            "current_ratio": fundamentals.get("current_ratio"),
            "beta":         fundamentals.get("beta"),
            "rsi":          screener.get("rsi"),
            "ma_signal":    screener.get("ma_signal"),
            "market_cap":   market_cap,
            "price":        fundamentals.get("price"),
            "overall_signal": screener.get("overall_signal")
        }

        # Check each filter
        for field, criteria in filters.items():
            value    = data.get(field)
            operator = criteria.get("operator")
            threshold = criteria.get("value")

            if value is None:
                return False, data

            if operator == "gt" and not (value > threshold):
                return False, data
            elif operator == "lt" and not (value < threshold):
                return False, data
            elif operator == "eq" and not (value == threshold):
                return False, data

        return True, data

    except Exception as e:
        print(f"[Error] Filter failed for {ticker}: {e}")
        return False, {}
def run_nl_screener(query: str, universe: list = None) -> pd.DataFrame:
    """
    Run the full natural language screening pipeline.

    1. Extract filters from plain English query using Claude API
    2. Apply filters across the stock universe
    3. Return ranked results as a DataFrame

    Args:
        query:    Plain English screening query
        universe: List of tickers to screen (default: STOCK_UNIVERSE)

    Returns:
        DataFrame of stocks passing all filters, ranked by revenue CAGR.
    """
    if universe is None:
        universe = STOCK_UNIVERSE

    print(f"Extracting filters from query: '{query}'")
    filters = extract_filters(query)

    if not filters:
        print("[Warning] No filters extracted. Check your query.")
        return pd.DataFrame()

    print(f"Extracted filters: {json.dumps(filters, indent=2)}")
    print(f"Screening {len(universe)} stocks...")

    results = []
    for ticker in universe:
        print(f"  Checking {ticker}...")
        passes, data = apply_filters(ticker, filters)
        if passes:
            results.append(data)

    if not results:
        print("No stocks passed all filters.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    if "revenue_cagr" in df.columns:
        df = df.sort_values("revenue_cagr", ascending=False)

    df.to_csv("outputs/nl_screener_results.csv", index=False)
    print(f"\nFound {len(df)} stocks matching your criteria.")

    return df