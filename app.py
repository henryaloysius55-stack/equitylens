"""
app.py — Streamlit web dashboard for EquityLens.

A professional stock analysis web app combining technical screening,
fundamental analysis, and Monte Carlo DCF valuation.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from modules.screener import screen_ticker
from modules.fundamentals import analyze_ticker
from modules.dcf import run_dcf_analysis, run_monte_carlo

# ── Page Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="EquityLens",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0e1a;
        color: #ffffff;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1225 100%);
        border: 1px solid #2a3050;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }

    .metric-label {
        color: #8892b0;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
    }

    .metric-positive { color: #00d4aa; }
    .metric-negative { color: #ff4757; }
    .metric-neutral  { color: #ffd700; }

    /* Signal badge */
    .signal-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 50px;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .signal-strong-buy  { background: #00d4aa22; color: #00d4aa;
                          border: 2px solid #00d4aa; }
    .signal-buy         { background: #00ff8822; color: #00ff88;
                          border: 2px solid #00ff88; }
    .signal-hold        { background: #ffd70022; color: #ffd700;
                          border: 2px solid #ffd700; }
    .signal-sell        { background: #ff475722; color: #ff4757;
                          border: 2px solid #ff4757; }
    .signal-strong-sell { background: #ff000022; color: #ff0000;
                          border: 2px solid #ff0000; }

    /* Section headers */
    .section-header {
        color: #64ffda;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #2a3050;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def get_signal_class(signal: str) -> str:
    """Convert signal string to CSS class name."""
    return "signal-" + signal.lower().replace(" ", "-")


def render_header():
    """Render the EquityLens header and search bar."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 40px 0 20px 0;'>
            <h1 style='color: #64ffda; font-size: 48px; font-weight: 900;
                       letter-spacing: 4px; margin: 0;'>
                EQUITY<span style='color: #ffffff;'>LENS</span>
            </h1>
            <p style='color: #8892b0; font-size: 14px; letter-spacing: 2px;
                      margin-top: 8px;'>
                QUANTITATIVE STOCK ANALYSIS PLATFORM
            </p>
        </div>
        """, unsafe_allow_html=True)

        ticker = st.text_input(
            "",
            placeholder="Enter a stock ticker (e.g. AAPL, MSFT, NVDA)",
            key="ticker_input"
        ).upper().strip()

        analyze = st.button("ANALYZE", use_container_width=True, type="primary")

    return ticker, analyze


def render_signal_banner(signal: str, ticker: str, price: float):
    """Render the overall signal banner at the top."""
    signal_class = get_signal_class(signal)
    color = {
        "STRONG BUY":  "#00d4aa",
        "BUY":         "#00ff88",
        "HOLD":        "#ffd700",
        "SELL":        "#ff4757",
        "STRONG SELL": "#ff0000"
    }.get(signal, "#ffffff")

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a1f35, #0d1225);
                border: 1px solid {color}33;
                border-left: 4px solid {color};
                border-radius: 12px; padding: 24px 32px;
                display: flex; align-items: center;
                justify-content: space-between; margin: 20px 0;'>
        <div>
            <div style='color: #8892b0; font-size: 12px;
                        letter-spacing: 2px;'>ANALYZING</div>
            <div style='color: #ffffff; font-size: 32px;
                        font-weight: 900;'>{ticker}</div>
        </div>
        <div style='text-align: center;'>
            <div style='color: #8892b0; font-size: 12px;
                        letter-spacing: 2px;'>CURRENT PRICE</div>
            <div style='color: #ffd700; font-size: 32px;
                        font-weight: 700;'>${price}</div>
        </div>
        <div style='text-align: right;'>
            <div style='color: #8892b0; font-size: 12px;
                        letter-spacing: 2px; margin-bottom: 8px;'>
                OVERALL SIGNAL
            </div>
            <span class='signal-badge {signal_class}'>{signal}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_technical(screener: dict):
    """Render the technical signals section."""
    st.markdown("<div class='section-header'>Technical Signals</div>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    rsi = screener.get("rsi", "N/A")
    rsi_signal = screener.get("rsi_signal", "neutral")
    rsi_color = ("metric-negative" if rsi_signal == "overbought"
                 else "metric-positive" if rsi_signal == "oversold"
                 else "metric-neutral")

    ma_signal = screener.get("ma_signal", "neutral")
    ma_color = ("metric-positive" if ma_signal == "golden_cross"
                else "metric-negative" if ma_signal == "death_cross"
                else "metric-neutral")

    vol_spike = screener.get("volume_spike", False)
    vol_color = "metric-positive" if vol_spike else "metric-neutral"

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>RSI (14 Day)</div>
            <div class='metric-value {rsi_color}'>{rsi}</div>
            <div style='color: #8892b0; font-size: 12px;
                        margin-top: 4px;'>{rsi_signal.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        ma_label = ("GOLDEN CROSS 🟢" if ma_signal == "golden_cross"
                    else "DEATH CROSS 🔴" if ma_signal == "death_cross"
                    else "NEUTRAL")
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Moving Average</div>
            <div class='metric-value {ma_color}'
                 style='font-size: 16px;'>{ma_label}</div>
            <div style='color: #8892b0; font-size: 12px; margin-top: 4px;'>
                50d: {screener.get('ma_50')} /
                200d: {screener.get('ma_200')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Volume Spike</div>
            <div class='metric-value {vol_color}'>
                {'YES ⚡' if vol_spike else 'NO'}
            </div>
            <div style='color: #8892b0; font-size: 12px; margin-top: 4px;'>
                {int(screener.get('current_volume', 0)):,} shares
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_fundamentals(fundamentals: dict):
    """Render the fundamental analysis section."""
    st.markdown("<div class='section-header'>Fundamental Analysis</div>",
                unsafe_allow_html=True)

    metrics = [
        ("P/E Ratio",       fundamentals.get("pe_ratio"),      ""),
        ("EV/EBITDA",       fundamentals.get("ev_ebitda"),      ""),
        ("Debt / Equity",   fundamentals.get("debt_equity"),    ""),
        ("Current Ratio",   fundamentals.get("current_ratio"),  ""),
        ("Gross Margin",    fundamentals.get("gross_margin"),   "%"),
        ("Revenue CAGR",    fundamentals.get("revenue_cagr"),   "%"),
    ]

    cols = st.columns(6)
    for i, (label, value, suffix) in enumerate(metrics):
        with cols[i]:
            display = f"{value}{suffix}" if value is not None else "N/A"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'
                     style='font-size: 20px;'>{display}</div>
            </div>
            """, unsafe_allow_html=True)

    score = fundamentals.get("fundamental_score", "N/A")
    score_color = ("#00d4aa" if isinstance(score, int) and score >= 4
                   else "#ffd700" if isinstance(score, int) and score == 3
                   else "#ff4757")
    st.markdown(f"""
    <div style='text-align: center; margin-top: 16px;'>
        <span style='color: #8892b0; font-size: 12px;
                     letter-spacing: 2px;'>FUNDAMENTAL SCORE  </span>
        <span style='color: {score_color}; font-size: 24px;
                     font-weight: 700;'>{score} / 5</span>
    </div>
    """, unsafe_allow_html=True)


def render_dcf(dcf: dict):
    """Render the DCF valuation section."""
    st.markdown("<div class='section-header'>DCF Valuation</div>",
                unsafe_allow_html=True)

    current_price = dcf.get("current_price", 0)
    scenarios     = ["bear", "base", "bull"]
    colors        = ["#ff4757", "#ffd700", "#00d4aa"]
    labels        = ["BEAR", "BASE", "BULL"]

    cols = st.columns(3)
    for i, scenario in enumerate(scenarios):
        data = dcf.get(scenario, {})
        iv   = data.get("intrinsic_value", "N/A")
        mos  = data.get("margin_of_safety", "N/A")

        mos_color = ("#00d4aa" if isinstance(mos, float) and mos > 0
                     else "#ff4757")

        with cols[i]:
            st.markdown(f"""
            <div class='metric-card'
                 style='border-color: {colors[i]}44;
                        border-top: 3px solid {colors[i]};'>
                <div class='metric-label'>{labels[i]} CASE</div>
                <div class='metric-value'
                     style='color: {colors[i]};'>${iv}</div>
                <div style='color: {mos_color}; font-size: 14px;
                            margin-top: 8px; font-weight: 600;'>
                    {mos}% vs current
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_monte_carlo_chart(results: dict):
    """Render the Monte Carlo histogram using Plotly."""
    if "error" in results:
        st.warning(f"Monte Carlo: {results['error']}")
        return

    st.markdown(
        "<div class='section-header'>Monte Carlo Simulation "
        "(10,000 Scenarios)</div>",
        unsafe_allow_html=True
    )

    values        = results["intrinsic_values"]
    current_price = results["current_price"]
    p10           = results["p10"]
    p50           = results["p50"]
    p90           = results["p90"]
    prob          = results["prob_undervalued"]

    # Build histogram with red/green coloring
    fig = go.Figure()

    # Red bars — overvalued scenarios
    fig.add_trace(go.Histogram(
        x      = values[values < current_price],
        nbinsx = 80,
        marker_color = "#ff4757",
        opacity      = 0.8,
        name         = "Overvalued Scenarios"
    ))

    # Green bars — undervalued scenarios
    fig.add_trace(go.Histogram(
        x      = values[values >= current_price],
        nbinsx = 80,
        marker_color = "#00d4aa",
        opacity      = 0.8,
        name         = "Undervalued Scenarios"
    ))

    # Vertical lines
    for val, color, label in [
        (current_price, "#ffffff", f"Current Price ${current_price}"),
        (p50,           "#ffd700", f"Median ${p50}"),
        (p10,           "#ff8c00", f"P10 ${p10}"),
        (p90,           "#00bfff", f"P90 ${p90}"),
    ]:
        fig.add_vline(
            x           = val,
            line_color  = color,
            line_width  = 2,
            line_dash   = "dash",
            annotation_text      = label,
            annotation_position  = "top",
            annotation_font_color = color
        )

    fig.update_layout(
        barmode      = "overlay",
        paper_bgcolor = "#0a0e1a",
        plot_bgcolor  = "#0a0e1a",
        font_color    = "#ffffff",
        title = dict(
            text      = f"Probability Undervalued: {prob}%",
            font_size = 16,
            font_color = "#64ffda"
        ),
        xaxis = dict(
            title      = "Intrinsic Value Per Share ($)",
            gridcolor  = "#1a1f35",
            zerolinecolor = "#2a3050"
        ),
        yaxis = dict(
            title     = "Number of Scenarios",
            gridcolor = "#1a1f35"
        ),
        legend = dict(
            bgcolor     = "#1a1f35",
            bordercolor = "#2a3050"
        ),
        height = 400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3, col4, col5 = st.columns(5)
    stats = [
        ("P10 Deep Bear", f"${results['p10']}"),
        ("P25 Bear",      f"${results['p25']}"),
        ("P50 Median",    f"${results['p50']}"),
        ("P75 Bull",      f"${results['p90']}"),
        ("P90 Deep Bull", f"${results['p90']}"),
    ]

    for col, (label, value) in zip(
        [col1, col2, col3, col4, col5], stats
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'
                     style='font-size: 18px;'>{value}</div>
            </div>
            """, unsafe_allow_html=True)


def get_overall_signal(screener, fundamentals, dcf):
    """Calculate overall signal from all three modules."""
    score = 0

    tech_signal = screener.get("overall_signal", "neutral")
    if tech_signal == "strong_buy":
        score += 30
    elif tech_signal == "buy":
        score += 15
    elif tech_signal == "sell":
        score -= 15
    elif tech_signal == "strong_sell":
        score -= 30

    fund_score = fundamentals.get("fundamental_score", 3)
    score += (fund_score - 3) * 10

    base_mos = dcf.get("base", {}).get("margin_of_safety", 0)
    if base_mos is not None:
        if base_mos > 20:
            score += 40
        elif base_mos > 0:
            score += 20
        elif base_mos > -20:
            score -= 20
        else:
            score -= 40

    if score >= 30:
        return "STRONG BUY"
    elif score >= 10:
        return "BUY"
    elif score >= -10:
        return "HOLD"
    elif score >= -30:
        return "SELL"
    else:
        return "STRONG SELL"


# ── Main App ───────────────────────────────────────────────────────────
def main():
    ticker, analyze = render_header()

    if analyze and ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            screener     = screen_ticker(ticker)
            fundamentals = analyze_ticker(ticker)
            dcf          = run_dcf_analysis(ticker)
            mc_results   = run_monte_carlo(ticker)

        if "error" in screener:
            st.error(f"Could not find data for {ticker}. "
                     f"Please check the ticker symbol.")
            return

        signal = get_overall_signal(screener, fundamentals, dcf)
        price  = fundamentals.get("price", "N/A")

        render_signal_banner(signal, ticker, price)

        st.markdown("<br>", unsafe_allow_html=True)
        render_technical(screener)

        st.markdown("<br>", unsafe_allow_html=True)
        render_fundamentals(fundamentals)

        st.markdown("<br>", unsafe_allow_html=True)
        render_dcf(dcf)

        st.markdown("<br>", unsafe_allow_html=True)
        render_monte_carlo_chart(mc_results)

    elif analyze and not ticker:
        st.warning("Please enter a stock ticker.")


if __name__ == "__main__":
    main()