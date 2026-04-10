"""
app.py — Streamlit web dashboard for EquityLens.

A professional stock analysis web app combining technical screening,
fundamental analysis, DCF valuation, Monte Carlo simulation,
and FinBERT news sentiment analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from modules.screener import screen_ticker
from modules.fundamentals import analyze_ticker, calculate_analyst_rating
from modules.dcf import run_dcf_analysis, run_monte_carlo
from modules.sentiment import analyze_sentiment

# Page Configuration
st.set_page_config(
    page_title="EquityLens",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #ffffff; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1225 100%);
        border: 1px solid #2a3050; border-radius: 12px;
        padding: 20px; text-align: center; margin: 5px;
    }
    .metric-label {
        color: #8892b0; font-size: 12px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
    }
    .metric-value { color: #ffffff; font-size: 24px; font-weight: 700; }
    .metric-positive { color: #00d4aa; }
    .metric-negative { color: #ff4757; }
    .metric-neutral  { color: #ffd700; }
    .signal-badge {
        display: inline-block; padding: 8px 24px;
        border-radius: 50px; font-size: 18px; font-weight: 800;
        letter-spacing: 2px; text-transform: uppercase;
    }
    .signal-strong-buy  { background: #00d4aa22; color: #00d4aa; border: 2px solid #00d4aa; }
    .signal-buy         { background: #00ff8822; color: #00ff88; border: 2px solid #00ff88; }
    .signal-hold        { background: #ffd70022; color: #ffd700; border: 2px solid #ffd700; }
    .signal-sell        { background: #ff475722; color: #ff4757; border: 2px solid #ff4757; }
    .signal-strong-sell { background: #ff000022; color: #ff0000; border: 2px solid #ff0000; }
    .section-header {
        color: #64ffda; font-size: 14px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2px;
        border-bottom: 1px solid #2a3050; padding-bottom: 8px;
        margin-bottom: 16px;
    }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def get_signal_class(signal: str) -> str:
    return "signal-" + signal.lower().replace(" ", "-")


def get_overall_signal(screener, fundamentals, dcf):
    score = 0
    tech_signal = screener.get("overall_signal", "neutral")
    if tech_signal == "strong_buy":   score += 30
    elif tech_signal == "buy":        score += 15
    elif tech_signal == "sell":       score -= 15
    elif tech_signal == "strong_sell": score -= 30

    fund_score = fundamentals.get("fundamental_score", 3)
    score += (fund_score - 3) * 10

    base_mos = dcf.get("base", {}).get("margin_of_safety", 0)
    if base_mos is not None:
        if base_mos > 20:    score += 40
        elif base_mos > 0:   score += 20
        elif base_mos > -20: score -= 20
        else:                score -= 40

    if score >= 30:   return "STRONG BUY"
    elif score >= 10: return "BUY"
    elif score >= -10: return "HOLD"
    elif score >= -30: return "SELL"
    else:             return "STRONG SELL"


def render_header():
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


def render_signal_banner(signal, ticker, price):
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
                border: 1px solid {color}33; border-left: 4px solid {color};
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
                OVERALL SIGNAL</div>
            <span class='signal-badge {signal_class}'>{signal}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_price_chart(ticker: str):
    """Render candlestick chart with MA and RSI panels."""
    import yfinance as yf
    from plotly.subplots import make_subplots

    st.markdown(
        "<div class='section-header'>Price History and Technical Chart</div>",
        unsafe_allow_html=True
    )

    df = yf.Ticker(ticker).history(period="1y")
    if df.empty:
        st.warning("No price data available.")
        return

    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2]
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=ticker,
        increasing_line_color="#00d4aa",
        decreasing_line_color="#ff4757"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA50"], name="50-Day MA",
        line=dict(color="#ffd700", width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA200"], name="200-Day MA",
        line=dict(color="#00bfff", width=1.5)
    ), row=1, col=1)

    colors = ["#00d4aa" if c >= o else "#ff4757"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.7
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI (14)",
        line=dict(color="#a78bfa", width=1.5)
    ), row=3, col=1)

    fig.add_hline(y=70, line_dash="dash", line_color="#ff4757",
                  opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00d4aa",
                  opacity=0.5, row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font_color="#ffffff", height=700, showlegend=True,
        legend=dict(bgcolor="#1a1f35", bordercolor="#2a3050"),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    fig.update_xaxes(gridcolor="#1a1f35", zerolinecolor="#2a3050")
    fig.update_yaxes(gridcolor="#1a1f35", zerolinecolor="#2a3050")
    st.plotly_chart(fig, use_container_width=True)


def render_analyst_ratings(ticker: str):
    """Display analyst consensus and price targets."""
    import yfinance as yf

    st.markdown(
        "<div class='section-header'>Analyst Consensus</div>",
        unsafe_allow_html=True
    )

    try:
        info           = yf.Ticker(ticker).info
        current_price  = info.get("currentPrice", 0)
        target_mean    = info.get("targetMeanPrice")
        target_high    = info.get("targetHighPrice")
        target_low     = info.get("targetLowPrice")
        recommendation = info.get("recommendationKey", "N/A").upper()
        num_analysts   = info.get("numberOfAnalystOpinions", "N/A")

        upside = ((target_mean - current_price) / current_price * 100
                  if target_mean else None)

        rec_color = (
            "#00d4aa" if recommendation in ["STRONG_BUY", "BUY"]
            else "#ffd700" if recommendation == "HOLD"
            else "#ff4757"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Consensus</div>"
                f"<div class='metric-value' style='color: {rec_color}; "
                f"font-size: 18px;'>{recommendation}</div>"
                f"<div style='color: #8892b0; font-size: 12px; "
                f"margin-top: 4px;'>{num_analysts} analysts</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Price Target</div>"
                f"<div class='metric-value'>${target_mean:.2f}</div>"
                f"<div style='color: #8892b0; font-size: 12px; "
                f"margin-top: 4px;'>Consensus mean</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col3:
            upside_color = "#00d4aa" if upside and upside > 0 else "#ff4757"
            upside_text  = f"{upside:+.1f}%" if upside else "N/A"
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Upside to Target</div>"
                f"<div class='metric-value' style='color: {upside_color};'>"
                f"{upside_text}</div>"
                f"<div style='color: #8892b0; font-size: 12px; "
                f"margin-top: 4px;'>vs current price</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col4:
            low  = f"${target_low:.0f}"  if target_low  else "N/A"
            high = f"${target_high:.0f}" if target_high else "N/A"
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Target Range</div>"
                f"<div class='metric-value' style='font-size: 16px;'>"
                f"{low} - {high}</div>"
                f"<div style='color: #8892b0; font-size: 12px; "
                f"margin-top: 4px;'>Low / High</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.warning(f"Analyst data unavailable for {ticker}")


def render_technical(screener: dict):
    """Render the technical signals section."""
    st.markdown(
        "<div class='section-header'>Technical Signals</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    rsi        = screener.get("rsi", "N/A")
    rsi_signal = screener.get("rsi_signal", "neutral")
    rsi_color  = (
        "metric-negative" if rsi_signal == "overbought"
        else "metric-positive" if rsi_signal == "oversold"
        else "metric-neutral"
    )

    ma_signal = screener.get("ma_signal", "neutral")
    ma_color  = (
        "metric-positive" if ma_signal == "golden_cross"
        else "metric-negative" if ma_signal == "death_cross"
        else "metric-neutral"
    )

    vol_spike = screener.get("volume_spike", False)
    vol_color = "metric-positive" if vol_spike else "metric-neutral"

    with col1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>RSI (14 Day)</div>"
            f"<div class='metric-value {rsi_color}'>{rsi}</div>"
            f"<div style='color: #8892b0; font-size: 12px; "
            f"margin-top: 4px;'>{rsi_signal.upper()}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col2:
        ma_label = (
            "GOLDEN CROSS" if ma_signal == "golden_cross"
            else "DEATH CROSS" if ma_signal == "death_cross"
            else "NEUTRAL"
        )
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Moving Average</div>"
            f"<div class='metric-value {ma_color}' "
            f"style='font-size: 16px;'>{ma_label}</div>"
            f"<div style='color: #8892b0; font-size: 12px; "
            f"margin-top: 4px;'>50d: {screener.get('ma_50')} / "
            f"200d: {screener.get('ma_200')}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Volume Spike</div>"
            f"<div class='metric-value {vol_color}'>"
            f"{'YES' if vol_spike else 'NO'}</div>"
            f"<div style='color: #8892b0; font-size: 12px; "
            f"margin-top: 4px;'>"
            f"{int(screener.get('current_volume', 0)):,} shares</div>"
            f"</div>",
            unsafe_allow_html=True
        )


def render_fundamentals(fundamentals: dict):
    """Render the fundamental analysis section."""
    st.markdown(
        "<div class='section-header'>Fundamental Analysis</div>",
        unsafe_allow_html=True
    )

    metrics = [
        ("P/E Ratio",     fundamentals.get("pe_ratio"),      ""),
        ("EV/EBITDA",     fundamentals.get("ev_ebitda"),      ""),
        ("Debt/Equity",   fundamentals.get("debt_equity"),    ""),
        ("Current Ratio", fundamentals.get("current_ratio"),  ""),
        ("Gross Margin",  fundamentals.get("gross_margin"),   "%"),
        ("Revenue CAGR",  fundamentals.get("revenue_cagr"),   "%"),
        ("Beta",          fundamentals.get("beta"),           ""),
    ]

    cols = st.columns(7)
    for i, (label, value, suffix) in enumerate(metrics):
        with cols[i]:
            display = f"{value}{suffix}" if value is not None else "N/A"
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='font-size: 20px;'>"
                f"{display}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    score       = fundamentals.get("fundamental_score", "N/A")
    score_color = (
        "#00d4aa" if isinstance(score, int) and score >= 4
        else "#ffd700" if isinstance(score, int) and score == 3
        else "#ff4757"
    )
    st.markdown(
        f"<div style='text-align: center; margin-top: 16px;'>"
        f"<span style='color: #8892b0; font-size: 12px; "
        f"letter-spacing: 2px;'>FUNDAMENTAL SCORE  </span>"
        f"<span style='color: {score_color}; font-size: 24px; "
        f"font-weight: 700;'>{score} / 5</span>"
        f"</div>",
        unsafe_allow_html=True
    )


def render_analyst_rating(rating: dict):
    """Display the 0-10 analyst rating with category breakdown."""
    if not rating:
        return

    score     = rating.get("score", 0)
    label     = rating.get("label", "N/A")
    breakdown = rating.get("breakdown", {})

    color = (
        "#00d4aa" if label in ["STRONG BUY", "BUY"]
        else "#ffd700" if label == "HOLD"
        else "#ff4757"
    )

    st.markdown(
        "<div class='section-header'>EquityLens Analyst Rating</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f"<div class='metric-card' style='border-top: 3px solid {color}; "
            f"text-align: center; padding: 30px;'>"
            f"<div class='metric-label'>ANALYST RATING</div>"
            f"<div style='color: {color}; font-size: 64px; "
            f"font-weight: 900; line-height: 1;'>{score}</div>"
            f"<div style='color: #8892b0; font-size: 12px; "
            f"margin: 4px 0;'>OUT OF 10</div>"
            f"<div style='color: {color}; font-size: 18px; "
            f"font-weight: 700; margin-top: 8px;'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col2:
        categories = [
            ("Valuation",        breakdown.get("valuation", 0), 3),
            ("Business Quality", breakdown.get("quality",   0), 3),
            ("Momentum",         breakdown.get("momentum",  0), 2),
            ("Risk",             breakdown.get("risk",      0), 2),
        ]

        for cat_name, cat_score, cat_max in categories:
            fill      = (cat_score / cat_max) * 100
            bar_color = (
                "#00d4aa" if fill >= 66
                else "#ffd700" if fill >= 33
                else "#ff4757"
            )
            st.markdown(
                f"<div style='margin-bottom: 16px;'>"
                f"<div style='display: flex; justify-content: space-between; "
                f"margin-bottom: 4px;'>"
                f"<span style='color: #8892b0; font-size: 12px; "
                f"font-weight: 600; text-transform: uppercase; "
                f"letter-spacing: 1px;'>{cat_name}</span>"
                f"<span style='color: {bar_color}; font-size: 12px; "
                f"font-weight: 700;'>{cat_score}/{cat_max}</span>"
                f"</div>"
                f"<div style='background: #1a1f35; border-radius: 4px; "
                f"height: 8px; overflow: hidden;'>"
                f"<div style='background: {bar_color}; width: {fill}%; "
                f"height: 100%; border-radius: 4px;'></div>"
                f"</div></div>",
                unsafe_allow_html=True
            )


def render_dcf(dcf: dict):
    """Render the DCF valuation section."""
    st.markdown(
        "<div class='section-header'>DCF Valuation</div>",
        unsafe_allow_html=True
    )

    current_price = dcf.get("current_price", 0)
    scenarios     = ["bear", "base", "bull"]
    colors        = ["#ff4757", "#ffd700", "#00d4aa"]
    labels        = ["BEAR", "BASE", "BULL"]

    cols = st.columns(3)
    for i, scenario in enumerate(scenarios):
        data = dcf.get(scenario, {})
        iv   = data.get("intrinsic_value", "N/A")
        mos  = data.get("margin_of_safety", "N/A")

        mos_color = (
            "#00d4aa" if isinstance(mos, float) and mos > 0
            else "#ff4757"
        )

        with cols[i]:
            st.markdown(
                f"<div class='metric-card' "
                f"style='border-color: {colors[i]}44; "
                f"border-top: 3px solid {colors[i]};'>"
                f"<div class='metric-label'>{labels[i]} CASE</div>"
                f"<div class='metric-value' "
                f"style='color: {colors[i]};'>${iv}</div>"
                f"<div style='color: {mos_color}; font-size: 14px; "
                f"margin-top: 8px; font-weight: 600;'>"
                f"{mos}% vs current</div>"
                f"</div>",
                unsafe_allow_html=True
            )


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

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=values[values < current_price], nbinsx=80,
        marker_color="#ff4757", opacity=0.8,
        name="Overvalued Scenarios"
    ))
    fig.add_trace(go.Histogram(
        x=values[values >= current_price], nbinsx=80,
        marker_color="#00d4aa", opacity=0.8,
        name="Undervalued Scenarios"
    ))

    for val, color, label in [
        (current_price, "#ffffff", f"Current Price ${current_price}"),
        (p50,           "#ffd700", f"Median ${p50}"),
        (p10,           "#ff8c00", f"P10 ${p10}"),
        (p90,           "#00bfff", f"P90 ${p90}"),
    ]:
        fig.add_vline(
            x=val, line_color=color, line_width=2, line_dash="dash",
            annotation_text=label, annotation_position="top",
            annotation_font_color=color
        )

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font_color="#ffffff",
        title=dict(
            text=f"Probability Undervalued: {prob}%",
            font_size=16, font_color="#64ffda"
        ),
        xaxis=dict(
            title="Intrinsic Value Per Share ($)",
            gridcolor="#1a1f35", zerolinecolor="#2a3050"
        ),
        yaxis=dict(title="Number of Scenarios", gridcolor="#1a1f35"),
        legend=dict(bgcolor="#1a1f35", bordercolor="#2a3050"),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    stats = [
        ("P10 Deep Bear", f"${results['p10']}"),
        ("P25 Bear",      f"${results['p25']}"),
        ("P50 Median",    f"${results['p50']}"),
        ("P75 Bull",      f"${results['p75']}"),
        ("P90 Deep Bull", f"${results['p90']}"),
    ]
    for col, (label, value) in zip(
        [col1, col2, col3, col4, col5], stats
    ):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='font-size: 18px;'>"
                f"{value}</div>"
                f"</div>",
                unsafe_allow_html=True
            )


def render_sentiment(sentiment: dict):
    """Display FinBERT news sentiment analysis."""
    if not sentiment or "error" in sentiment:
        st.warning("Sentiment data unavailable.")
        return

    st.markdown(
        "<div class='section-header'>News Sentiment Analysis</div>",
        unsafe_allow_html=True
    )

    label          = sentiment.get("sentiment_label", "neutral")
    display_score  = sentiment.get("display_score", 50)
    headline_count = sentiment.get("headline_count", 0)
    headlines      = sentiment.get("headlines", [])

    color = (
        "#00d4aa" if label == "positive"
        else "#ff4757" if label == "negative"
        else "#ffd700"
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f"<div class='metric-card' style='text-align: center; "
            f"padding: 30px; border-top: 3px solid {color};'>"
            f"<div class='metric-label'>SENTIMENT SCORE</div>"
            f"<div style='color: {color}; font-size: 56px; "
            f"font-weight: 900; line-height: 1;'>{display_score:.0f}</div>"
            f"<div style='color: #8892b0; font-size: 12px; "
            f"margin: 4px 0;'>OUT OF 100</div>"
            f"<div style='color: {color}; font-size: 18px; "
            f"font-weight: 700; margin-top: 8px;'>{label.upper()}</div>"
            f"<div style='color: #8892b0; font-size: 11px; "
            f"margin-top: 8px;'>Based on {headline_count} "
            f"recent headlines</div></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            "<div style='color: #64ffda; font-size: 12px; "
            "font-weight: 700; text-transform: uppercase; "
            "letter-spacing: 1px; margin-bottom: 12px;'>"
            "Recent Headlines</div>",
            unsafe_allow_html=True
        )

        for item in headlines[:6]:
            h_label  = item.get("label", "neutral")
            h_score  = item.get("score", 0)
            headline = item.get("headline", "")

            h_color   = (
                "#00d4aa" if h_label == "positive"
                else "#ff4757" if h_label == "negative"
                else "#8892b0"
            )
            indicator = (
                "+" if h_label == "positive"
                else "-" if h_label == "negative"
                else "o"
            )

            st.markdown(
                f"<div style='display: flex; align-items: flex-start; "
                f"margin-bottom: 10px; padding: 8px 12px; "
                f"background: #1a1f35; border-radius: 8px; "
                f"border-left: 3px solid {h_color};'>"
                f"<span style='color: {h_color}; font-size: 14px; "
                f"margin-right: 8px; flex-shrink: 0;'>{indicator}</span>"
                f"<div><div style='color: #ffffff; font-size: 12px; "
                f"line-height: 1.4;'>{headline}</div>"
                f"<div style='color: {h_color}; font-size: 11px; "
                f"margin-top: 2px;'>{h_label.upper()} "
                f"confidence {h_score:.0%}</div></div></div>",
                unsafe_allow_html=True
            )


def main():
    ticker, analyze = render_header()

    if analyze and ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            screener     = screen_ticker(ticker)
            fundamentals = analyze_ticker(ticker)
            dcf          = run_dcf_analysis(ticker)
            mc_results   = run_monte_carlo(ticker)
            sentiment    = analyze_sentiment(ticker)

        if "error" in screener:
            st.error(f"Could not find data for {ticker}.")
            return

        import yfinance as yf
        rec = yf.Ticker(ticker).info.get(
            "recommendationKey", "N/A"
        ).upper()

        rating = calculate_analyst_rating(
            gross_margin      = fundamentals.get("gross_margin"),
            revenue_cagr      = fundamentals.get("revenue_cagr"),
            fundamental_score = fundamentals.get("fundamental_score"),
            rsi               = screener.get("rsi"),
            ma_signal         = screener.get("ma_signal"),
            base_mos          = dcf.get("base", {}).get("margin_of_safety"),
            prob_undervalued  = mc_results.get("prob_undervalued"),
            recommendation    = rec,
            beta              = fundamentals.get("beta"),
            current_ratio     = fundamentals.get("current_ratio")
        )

        signal = get_overall_signal(screener, fundamentals, dcf)
        price  = fundamentals.get("price", "N/A")

        render_signal_banner(signal, ticker, price)
        render_analyst_rating(rating)
        render_price_chart(ticker)
        render_analyst_ratings(ticker)

        st.markdown("<br>", unsafe_allow_html=True)
        render_technical(screener)

        st.markdown("<br>", unsafe_allow_html=True)
        render_fundamentals(fundamentals)

        st.markdown("<br>", unsafe_allow_html=True)
        render_dcf(dcf)

        st.markdown("<br>", unsafe_allow_html=True)
        render_monte_carlo_chart(mc_results)

        st.markdown("<br>", unsafe_allow_html=True)
        render_sentiment(sentiment)

    elif analyze and not ticker:
        st.warning("Please enter a stock ticker.")


if __name__ == "__main__":
    main()