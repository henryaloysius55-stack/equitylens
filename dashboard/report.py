"""
report.py — Unified dashboard for EquityLens.

Combines screener, fundamental, and DCF analysis into a single
professional terminal report using the rich library.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.columns import Columns
from rich.text import Text

from modules.screener import screen_ticker
from modules.fundamentals import analyze_ticker
from modules.dcf import run_dcf_analysis, run_monte_carlo

console = Console()


def get_overall_signal(screener: dict, fundamentals: dict, dcf: dict) -> tuple:
    """
    Combine signals from all three modules into one overall recommendation.

    Weighting:
    - Technical signal: 30%
    - Fundamental score: 30%
    - DCF margin of safety: 40%

    Args:
        screener:     Output from screen_ticker()
        fundamentals: Output from analyze_ticker()
        dcf:          Output from run_dcf_analysis()

    Returns:
        Tuple of (signal string, color string) e.g. ("BUY", "green")
    """
    score = 0

    # Technical signal — 30% weight
    tech_signal = screener.get("overall_signal", "neutral")
    if tech_signal == "strong_buy":
        score += 30
    elif tech_signal == "buy":
        score += 15
    elif tech_signal == "sell":
        score -= 15
    elif tech_signal == "strong_sell":
        score -= 30

    # Fundamental score — 30% weight
    fund_score = fundamentals.get("fundamental_score", 3)
    score += (fund_score - 3) * 10

    # DCF margin of safety — 40% weight using base case
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

    # Convert score to signal
    if score >= 30:
        return "STRONG BUY", "bold green"
    elif score >= 10:
        return "BUY", "green"
    elif score >= -10:
        return "HOLD", "yellow"
    elif score >= -30:
        return "SELL", "red"
    else:
        return "STRONG SELL", "bold red"


def render_report(ticker: str):
    """
    Run full EquityLens analysis and render a professional terminal report.

    Args:
        ticker: Stock symbol as a string e.g. "AAPL"
    """
    ticker = ticker.upper()
    console.print(f"\n[bold cyan]EquityLens — Analyzing {ticker}...[/bold cyan]\n")

    # Run all three modules
    screener     = screen_ticker(ticker)
    fundamentals = analyze_ticker(ticker)
    dcf          = run_dcf_analysis(ticker)
    # ─── HEADER ───────────────────────────────────────────────
    price = fundamentals.get("price", "N/A")
    signal, signal_color = get_overall_signal(screener, fundamentals, dcf)

    header = Text()
    header.append(f"{ticker}", style="bold white")
    header.append(f"  ${price}", style="bold yellow")
    header.append(f"  ●  Overall Signal: ", style="white")
    header.append(f"{signal}", style=signal_color)

    console.print(Panel(header, title="EquityLens Report", 
                        border_style="cyan", box=box.DOUBLE))

    # ─── TECHNICAL SIGNALS TABLE ──────────────────────────────
    tech_table = Table(title="Technical Signals", 
                       box=box.SIMPLE_HEAVY, 
                       border_style="cyan")
    tech_table.add_column("Indicator",  style="white")
    tech_table.add_column("Value",      style="yellow")
    tech_table.add_column("Signal",     style="green")

    rsi = screener.get("rsi", "N/A")
    rsi_signal = screener.get("rsi_signal", "N/A")
    rsi_color = "red" if rsi_signal == "overbought" else \
                "green" if rsi_signal == "oversold" else "yellow"

    ma_signal = screener.get("ma_signal", "N/A")
    ma_color = "green" if ma_signal == "golden_cross" else \
               "red" if ma_signal == "death_cross" else "yellow"

    vol_spike = screener.get("volume_spike", False)
    vol_color = "green" if vol_spike else "yellow"

    tech_table.add_row("RSI (14)", 
                       str(rsi), 
                       Text(rsi_signal, style=rsi_color))
    tech_table.add_row("MA Crossover", 
                       f"50d: {screener.get('ma_50')} / 200d: {screener.get('ma_200')}", 
                       Text(ma_signal, style=ma_color))
    tech_table.add_row("Volume Spike", 
                       f"{screener.get('current_volume'):,.0f}", 
                       Text("YES" if vol_spike else "NO", style=vol_color))

    # ─── FUNDAMENTALS TABLE ───────────────────────────────────
    fund_table = Table(title="Fundamental Analysis", 
                       box=box.SIMPLE_HEAVY, 
                       border_style="magenta")
    fund_table.add_column("Metric",   style="white")
    fund_table.add_column("Value",    style="yellow")

    fund_table.add_row("P/E Ratio",       str(fundamentals.get("pe_ratio", "N/A")))
    fund_table.add_row("EV/EBITDA",       str(fundamentals.get("ev_ebitda", "N/A")))
    fund_table.add_row("Debt / Equity",   str(fundamentals.get("debt_equity", "N/A")))
    fund_table.add_row("Current Ratio",   str(fundamentals.get("current_ratio", "N/A")))
    fund_table.add_row("Gross Margin",    f"{fundamentals.get('gross_margin', 'N/A')}%")
    fund_table.add_row("Revenue CAGR",    f"{fundamentals.get('revenue_cagr', 'N/A')}%")
    fund_table.add_row("Fundamental Score", 
                       f"{fundamentals.get('fundamental_score', 'N/A')} / 5")

    # ─── DCF TABLE ────────────────────────────────────────────
    dcf_table = Table(title="DCF Valuation", 
                      box=box.SIMPLE_HEAVY, 
                      border_style="green")
    dcf_table.add_column("Scenario", style="white")
    dcf_table.add_column("Intrinsic Value", style="yellow")
    dcf_table.add_column("Margin of Safety", style="white")

    for scenario in ["bear", "base", "bull"]:
        data = dcf.get(scenario, {})
        iv   = data.get("intrinsic_value", "N/A")
        mos  = data.get("margin_of_safety", "N/A")

        mos_color = "green" if isinstance(mos, float) and mos > 0 else "red"
        mos_text  = Text(f"{mos}%", style=mos_color)

        dcf_table.add_row(scenario.upper(), f"${iv}", mos_text)

    # ─── RENDER ALL TABLES ────────────────────────────────────
    console.print(Columns([tech_table, fund_table]))
    console.print(dcf_table)
    render_monte_carlo_summary(ticker)
    console.print()
def render_monte_carlo_summary(ticker: str):
    """
    Add Monte Carlo summary statistics to the terminal report.

    Args:
        ticker: Stock symbol as a string e.g. "AAPL"
    """
    results = run_monte_carlo(ticker)

    if "error" in results:
        console.print(f"[red]Monte Carlo failed: {results['error']}[/red]")
        return

    mc_table = Table(
        title="Monte Carlo DCF Simulation (10,000 Scenarios)",
        box=box.SIMPLE_HEAVY,
        border_style="yellow"
    )

    mc_table.add_column("Percentile",      style="white")
    mc_table.add_column("Intrinsic Value", style="yellow")
    mc_table.add_column("vs Current Price", style="white")

    current = results["current_price"]

    rows = [
        ("P10 — Deep Bear", results["p10"]),
        ("P25 — Bear",      results["p25"]),
        ("P50 — Median",    results["p50"]),
        ("P75 — Bull",      results["p75"]),
        ("P90 — Deep Bull", results["p90"]),
    ]

    for label, value in rows:
        diff       = ((value - current) / current) * 100
        diff_color = "green" if diff > 0 else "red"
        diff_text  = Text(f"{diff:+.1f}%", style=diff_color)
        mc_table.add_row(label, f"${value:.2f}", diff_text)

    prob = results["prob_undervalued"]
    prob_color = "green" if prob > 50 else "red"

    console.print(mc_table)
    console.print(
        Panel(
            f"[white]Probability Undervalued: "
            f"[{prob_color}]{prob}%[/{prob_color}]  |  "
            f"Current Price: [yellow]${current}[/yellow]  |  "
            f"Median Intrinsic Value: [yellow]${results['p50']}[/yellow]  |  "
            f"Uncertainty (±1σ): [yellow]${results['std']:.2f}[/yellow][/white]",
            border_style="yellow"
        )
    )