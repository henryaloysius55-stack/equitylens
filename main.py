"""
main.py — Entry point for EquityLens.

Run this file to generate a full analysis report for any stock ticker.
"""

from dashboard.report import render_report

if __name__ == "__main__":
    ticker = input("Enter a stock ticker to analyze: ").strip().upper()
    render_report(ticker)