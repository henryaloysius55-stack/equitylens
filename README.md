# EquityLens 📈

A professional-grade stock analysis tool built in Python that combines 
technical screening, fundamental analysis, and DCF valuation into a 
single unified terminal dashboard.

Built by a 9th-grade student as a second major Python project, 
EquityLens demonstrates quantitative finance concepts used by 
professional analysts at hedge funds and investment banks.

---

## What It Does

For any stock ticker, EquityLens produces a one-page terminal report showing:

- **Technical Signal** — RSI, moving average crossover, and volume spike detection
- **Fundamental Score** — P/E, EV/EBITDA, debt/equity, current ratio, 
  gross margin, and 3-year revenue CAGR
- **DCF Valuation** — Bull, base, and bear intrinsic value estimates 
  with margin of safety
- **Overall Signal** — A weighted BUY / HOLD / SELL recommendation 
  combining all three analyses

---

## Architecture
```
equitylens/
│
├── data/
│   └── fetcher.py          # Centralized data layer (yfinance)
│
├── modules/
│   ├── screener.py         # Technical analysis engine
│   ├── fundamentals.py     # Fundamental ratio calculator
│   └── dcf.py              # Discounted cash flow model
│
├── dashboard/
│   └── report.py           # Unified terminal dashboard (rich)
│
├── outputs/                # Generated CSV watchlists
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

---

## Key Financial Concepts Implemented

**RSI (Relative Strength Index)** — Measures momentum by comparing 
average gains to average losses over 14 days. Above 70 signals 
overbought conditions, below 30 signals oversold.

**Golden Cross / Death Cross** — When the 50-day moving average crosses 
above or below the 200-day moving average, signaling medium-term momentum 
shifts that institutional investors watch closely.

**EV/EBITDA** — Enterprise value divided by earnings before interest, 
taxes, depreciation and amortization. Preferred over P/E by professional 
analysts because it is capital structure neutral.

**DCF Valuation** — Projects free cash flow over 10 years, discounts 
each year back to present value using WACC, and adds a terminal value 
representing all cash flows beyond the projection period.

**Margin of Safety** — The difference between intrinsic value and current 
price, expressed as a percentage. Coined by Benjamin Graham, Warren 
Buffett's mentor.

---

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/equitylens.git
cd equitylens

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run EquityLens
python main.py
```

---

## Example Output
```
╔══════════════════════ EquityLens Report ══════════════════════╗
║ AAPL  $248.80  ●  Overall Signal: STRONG SELL                 ║
╚═══════════════════════════════════════════════════════════════╝

Technical Signals
  RSI (14)       37.97    neutral
  MA Crossover   golden_cross
  Volume Spike   NO

Fundamental Analysis
  P/E Ratio          31.45
  EV/EBITDA          24.04
  Gross Margin       47.33%
  Fundamental Score  2 / 5

DCF Valuation
  BEAR   $87.59    -64.8%
  BASE   $149.00   -40.1%
  BULL   $281.99   +13.3%
```

---

## What I Learned

- **Software architecture** — Why separating concerns into modules 
  makes code maintainable and professional
- **Financial data pipelines** — How to fetch, clean, and transform 
  real market data using yfinance and pandas
- **Technical analysis** — How traders use RSI, moving averages, 
  and volume to identify momentum signals
- **Fundamental analysis** — How to read financial statements and 
  calculate the ratios professional analysts use daily
- **DCF valuation** — How discounted cash flow models work, why 
  terminal value dominates the output, and why single-point estimates 
  are misleading
- **Critical thinking about models** — Why Amazon scores 5/5 on 
  fundamentals but appears massively overvalued in DCF — and what 
  that tells us about the limits of quantitative models

---

## Limitations and Future Work

- DCF assumptions are simplified — a production model would use 
  CAPM to estimate WACC from live risk-free rates and beta
- yfinance is an unofficial API and can break if Yahoo changes 
  their data structure
- Future upgrade: Monte Carlo simulation to replace fixed DCF 
  scenarios with probability distributions across 10,000 simulated 
  outcomes — the approach used by quantitative analysts at hedge funds

---

## Libraries Used

- `yfinance` — Market data fetching
- `pandas` — Data manipulation and analysis
- `numpy` — Numerical calculations
- `ta` — Technical indicators (RSI)
- `rich` — Terminal formatting and display