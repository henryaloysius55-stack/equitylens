"""
fetcher.py — Centralized data layer for EquityLens.

All modules should import from here instead of calling yfinance directly.
This ensures consistent data handling and a single point of maintenance.
"""

import yfinance as yf
import pandas as pd


def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical OHLCV price data for a given ticker.

    Args:
        ticker: Stock symbol as a string, e.g. "AAPL"
        period: How far back to fetch. Options: 1mo, 3mo, 6mo, 1y, 2y, 5y
    
    Returns:
        A pandas DataFrame with columns: Open, High, Low, Close, Volume
        Returns an empty DataFrame if the ticker is invalid or data unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            print(f"[Warning] No price data found for ticker: {ticker}")
            return pd.DataFrame()

        return df

    except Exception as e:
        print(f"[Error] Could not fetch price data for {ticker}: {e}")
        return pd.DataFrame()


def get_financials(ticker: str) -> dict:
    """
    Fetch fundamental financial statements for a given ticker.

    Args:
        ticker: Stock symbol as a string, e.g. "AAPL"

    Returns:
        A dictionary with keys: 'income_stmt', 'balance_sheet', 'cash_flow'
        Each value is a pandas DataFrame. Returns empty dict on failure.
    """
    try:
        stock = yf.Ticker(ticker)

        financials = {
            "income_stmt":   stock.income_stmt,
            "balance_sheet": stock.balance_sheet,
            "cash_flow":     stock.cash_flow,
            "info":          stock.info
        }

        return financials

    except Exception as e:
        print(f"[Error] Could not fetch financials for {ticker}: {e}")
        return {}
    
def get_current_price(ticker: str):
    """
    Fetch the current price for a given ticker.

    Args:
        ticker: Stock symbol as a string e.g. "AAPL"
    
    Returns:
        Current price as a float, or None if unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info["currentPrice"]
        return price
    except Exception as e:
        print(f"[Error] Could not fetch current price for {ticker}: {e}")
        return None