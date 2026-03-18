"""
S&P 500 Sector ETF Analysis
============================
Analyzes 3 years of S&P 500 sector ETF data to identify defensive sectors
with superior risk-adjusted returns.

Outputs:
  - Correlation heatmap (output/correlation_heatmap.png)
  - Rolling return charts (output/rolling_returns.png)
  - Summary statistics (output/sector_summary.csv)
  - Risk-return scatter plot (output/risk_return_scatter.png)
  - Sector performance dashboard (output/sector_dashboard.png)

Author: Robert
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
import os
import sqlite3

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLF': 'Financials',
    'XLY': 'Cons. Discretionary',
    'XLP': 'Cons. Staples',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLU': 'Utilities',
    'XLC': 'Communication Svcs'
}

BENCHMARK = 'SPY'  # S&P 500 benchmark
RISK_FREE_RATE = 0.04  # ~4% annualized (approximate recent T-bill rate)
TRADING_DAYS = 252

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=3*365)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
SQL_DIR = os.path.join(BASE_DIR, 'sql')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SQL_DIR, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})


# ──────────────────────────────────────────────
# 1. DATA ACQUISITION
# ──────────────────────────────────────────────
def fetch_data():
    """Download sector ETF and benchmark price data from Yahoo Finance."""
    print("=" * 60)
    print("STEP 1: Fetching S&P 500 Sector ETF Data")
    print("=" * 60)

    tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Period:  {START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")

    raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True)
    prices = raw['Close'].dropna()

    # Rename columns to sector names for readability
    rename_map = {**SECTOR_ETFS, BENCHMARK: 'S&P 500'}
    prices.columns = [rename_map.get(c, c) for c in prices.columns]

    print(f"  ✓ Downloaded {len(prices)} trading days of data")
    print(f"  ✓ Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")

    # Save raw data
    prices.to_csv(os.path.join(DATA_DIR, 'sector_prices.csv'))
    print(f"  ✓ Saved to data/sector_prices.csv")

    return prices


# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────
def compute_returns(prices):
    """Calculate daily, cumulative, and rolling returns."""
    print("\n" + "=" * 60)
    print("STEP 2: Computing Returns & Risk Metrics")
    print("=" * 60)

    daily_returns = prices.pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod() - 1

    # Rolling 60-day (≈3 month) annualized return
    rolling_returns = daily_returns.rolling(window=60).mean() * TRADING_DAYS

    # Save returns
    daily_returns.to_csv(os.path.join(DATA_DIR, 'daily_returns.csv'))
    print(f"  ✓ Computed daily returns ({len(daily_returns)} observations)")
    print(f"  ✓ Computed cumulative returns")
    print(f"  ✓ Computed 60-day rolling annualized returns")

    return daily_returns, cumulative_returns, rolling_returns


def compute_risk_metrics(daily_returns):
    """Calculate key risk-adjusted performance metrics for each sector."""
    sectors = [c for c in daily_returns.columns if c != 'S&P 500']

    metrics = []
    for sector in sectors:
        r = daily_returns[sector]
        ann_return = r.mean() * TRADING_DAYS
        ann_vol = r.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol
        sortino_downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
        sortino = (ann_return - RISK_FREE_RATE) / sortino_downside if sortino_downside > 0 else np.nan
        max_dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan
        beta = r.cov(daily_returns['S&P 500']) / daily_returns['S&P 500'].var()
        alpha = ann_return - (RISK_FREE_RATE + beta * (daily_returns['S&P 500'].mean() * TRADING_DAYS - RISK_FREE_RATE))

        metrics.append({
            'Sector': sector,
            'Ann. Return (%)': round(ann_return * 100, 2),
            'Ann. Volatility (%)': round(ann_vol * 100, 2),
            'Sharpe Ratio': round(sharpe, 3),
            'Sortino Ratio': round(sortino, 3),
            'Max Drawdown (%)': round(max_dd * 100, 2),
            'Calmar Ratio': round(calmar, 3),
            'Beta': round(beta, 3),
            'Alpha (%)': round(alpha * 100, 2),
        })

    df_metrics = pd.DataFrame(metrics).set_index('Sector')
    df_metrics.to_csv(os.path.join(DATA_DIR, 'sector_summary.csv'))

    print(f"\n  Risk Metrics Summary:")
    print(df_metrics.to_string())

    return df_metrics


# ──────────────────────────────────────────────
# 3. IDENTIFY DEFENSIVE SECTORS
# ──────────────────────────────────────────────
def identify_defensive_sectors(df_metrics):
    """
    Identify defensive sectors using a composite score:
      - High Sharpe Ratio (risk-adjusted return)
      - Low Beta (less market sensitivity)
      - Shallow Max Drawdown (downside protection)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Identifying Defensive Sectors")
    print("=" * 60)

    df = df_metrics.copy()

    # Normalize each factor to [0, 1] (higher = more defensive)
    df['Sharpe_score'] = (df['Sharpe Ratio'] - df['Sharpe Ratio'].min()) / \
                         (df['Sharpe Ratio'].max() - df['Sharpe Ratio'].min())
    df['Beta_score'] = 1 - (df['Beta'] - df['Beta'].min()) / \
                       (df['Beta'].max() - df['Beta'].min())  # Lower beta = higher score
    df['DD_score'] = 1 - (df['Max Drawdown (%)'].abs() - df['Max Drawdown (%)'].abs().min()) / \
                     (df['Max Drawdown (%)'].abs().max() - df['Max Drawdown (%)'].abs().min())

    df['Defensive Score'] = (df['Sharpe_score'] * 0.4 +
                              df['Beta_score'] * 0.3 +
                              df['DD_score'] * 0.3)

    df_ranked = df.sort_values('Defensive Score', ascending=False)
    top_3 = df_ranked.head(3).index.tolist()

    print(f"\n  Defensive Ranking (composite score):")
    for i, (sector, row) in enumerate(df_ranked.iterrows(), 1):
        marker = " ★" if sector in top_3 else ""
        print(f"    {i:2d}. {sector:<22s}  Score: {row['Defensive Score']:.3f}"
              f"  (Sharpe: {row['Sharpe Ratio']:.3f}, Beta: {row['Beta']:.2f}, "
              f"MaxDD: {row['Max Drawdown (%)']:.1f}%){marker}")

    print(f"\n  → Top 3 Defensive Sectors: {', '.join(top_3)}")

    return top_3, df_ranked


# ──────────────────────────────────────────────
# 4. VISUALIZATIONS
# ──────────────────────────────────────────────
def plot_correlation_heatmap(daily_returns):
    """Generate sector correlation heatmap."""
    print("\n" + "=" * 60)
    print("STEP 4a: Generating Correlation Heatmap")
    print("=" * 60)

    sectors_only = daily_returns.drop(columns=['S&P 500'], errors='ignore')
    corr = sectors_only.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', square=True, linewidths=0.8,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                ax=ax)

    ax.set_title('S&P 500 Sector ETF Correlation Matrix\n(3-Year Daily Returns)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_rolling_returns(rolling_returns, top_3):
    """Generate rolling return comparison chart with defensive sectors highlighted."""
    print("\n" + "=" * 60)
    print("STEP 4b: Generating Rolling Return Charts")
    print("=" * 60)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: All sectors
    ax1 = axes[0]
    sectors = [c for c in rolling_returns.columns if c != 'S&P 500']
    for sector in sectors:
        is_top = sector in top_3
        ax1.plot(rolling_returns.index, rolling_returns[sector],
                 linewidth=2.2 if is_top else 0.8,
                 alpha=1.0 if is_top else 0.3,
                 label=sector if is_top else None,
                 zorder=10 if is_top else 1)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('60-Day Rolling Annualized Returns by Sector\n(Defensive Sectors Highlighted)',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Annualized Return')
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.legend(loc='upper left', framealpha=0.9)

    # Bottom panel: Defensive vs S&P 500
    ax2 = axes[1]
    defensive_avg = rolling_returns[top_3].mean(axis=1)
    ax2.plot(rolling_returns.index, defensive_avg,
             color='#2ecc71', linewidth=2, label='Defensive Avg (Top 3)')
    ax2.plot(rolling_returns.index, rolling_returns['S&P 500'],
             color='#3498db', linewidth=2, label='S&P 500', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(rolling_returns.index,
                     defensive_avg, rolling_returns['S&P 500'],
                     where=defensive_avg > rolling_returns['S&P 500'],
                     alpha=0.15, color='green', label='Defensive Outperformance')
    ax2.fill_between(rolling_returns.index,
                     defensive_avg, rolling_returns['S&P 500'],
                     where=defensive_avg <= rolling_returns['S&P 500'],
                     alpha=0.15, color='red')

    ax2.set_title('Defensive Sectors vs S&P 500 Benchmark', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Annualized Return')
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'rolling_returns.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_risk_return_scatter(df_metrics, top_3):
    """Generate risk-return scatter plot with Sharpe ratio reference lines."""
    print("\n" + "=" * 60)
    print("STEP 4c: Generating Risk-Return Scatter Plot")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(11, 8))

    for sector in df_metrics.index:
        x = df_metrics.loc[sector, 'Ann. Volatility (%)']
        y = df_metrics.loc[sector, 'Ann. Return (%)']
        is_top = sector in top_3
        color = '#2ecc71' if is_top else '#95a5a6'
        edge = '#27ae60' if is_top else '#7f8c8d'

        ax.scatter(x, y, s=180 if is_top else 100,
                   c=color, edgecolors=edge, linewidth=2, zorder=10 if is_top else 5)
        offset_y = 0.6
        ax.annotate(sector, (x, y), textcoords='offset points',
                    xytext=(8, offset_y), fontsize=9,
                    fontweight='bold' if is_top else 'normal',
                    color='#2c3e50' if is_top else '#7f8c8d')

    # Sharpe = 0.5 and 1.0 reference lines
    vol_range = np.linspace(0, df_metrics['Ann. Volatility (%)'].max() * 1.15, 100)
    for sharpe_val, ls in [(0.5, '--'), (1.0, ':')]:
        ret_line = RISK_FREE_RATE * 100 + sharpe_val * vol_range
        ax.plot(vol_range, ret_line, ls, color='gray', alpha=0.5, linewidth=1,
                label=f'Sharpe = {sharpe_val}')

    ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
    ax.set_ylabel('Annualized Return (%)', fontsize=12)
    ax.set_title('Risk-Return Profile: S&P 500 Sector ETFs\n(Green = Top 3 Defensive Sectors)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'risk_return_scatter.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_dashboard(prices, cumulative_returns, df_metrics, top_3):
    """Generate a 4-panel summary dashboard."""
    print("\n" + "=" * 60)
    print("STEP 4d: Generating Summary Dashboard")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Cumulative returns
    ax = axes[0, 0]
    for sector in cumulative_returns.columns:
        is_top = sector in top_3
        ax.plot(cumulative_returns.index, cumulative_returns[sector] * 100,
                linewidth=2 if is_top else 0.8,
                alpha=1.0 if is_top else 0.25,
                label=sector if is_top else None)
    ax.set_title('Cumulative Returns', fontweight='bold')
    ax.set_ylabel('Return (%)')
    ax.legend(fontsize=8)

    # Panel 2: Sharpe ratios bar chart
    ax = axes[0, 1]
    sectors_sorted = df_metrics.sort_values('Sharpe Ratio', ascending=True)
    colors = ['#2ecc71' if s in top_3 else '#bdc3c7' for s in sectors_sorted.index]
    ax.barh(sectors_sorted.index, sectors_sorted['Sharpe Ratio'], color=colors, edgecolor='white')
    ax.set_title('Sharpe Ratios by Sector', fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)

    # Panel 3: Beta comparison
    ax = axes[1, 0]
    sectors_sorted_beta = df_metrics.sort_values('Beta', ascending=True)
    colors = ['#2ecc71' if s in top_3 else '#bdc3c7' for s in sectors_sorted_beta.index]
    ax.barh(sectors_sorted_beta.index, sectors_sorted_beta['Beta'], color=colors, edgecolor='white')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, label='Market Beta = 1')
    ax.set_title('Beta (Market Sensitivity)', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 4: Max Drawdown
    ax = axes[1, 1]
    sectors_sorted_dd = df_metrics.sort_values('Max Drawdown (%)', ascending=False)
    colors = ['#2ecc71' if s in top_3 else '#bdc3c7' for s in sectors_sorted_dd.index]
    ax.barh(sectors_sorted_dd.index, sectors_sorted_dd['Max Drawdown (%)'], color=colors, edgecolor='white')
    ax.set_title('Maximum Drawdown (%)', fontweight='bold')

    fig.suptitle('S&P 500 Sector ETF Analysis Dashboard',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'sector_dashboard.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


# ──────────────────────────────────────────────
# 5. SQL ANALYSIS (SQLite)
# ──────────────────────────────────────────────
def run_sql_analysis(daily_returns, df_metrics):
    """Store data in SQLite and run analytical queries."""
    print("\n" + "=" * 60)
    print("STEP 5: SQL Analysis")
    print("=" * 60)

    db_path = os.path.join(DATA_DIR, 'sector_analysis.db')
    conn = sqlite3.connect(db_path)

    # Store daily returns in SQL
    returns_long = daily_returns.reset_index().melt(
        id_vars='Date', var_name='Sector', value_name='Daily_Return'
    )
    returns_long.to_sql('daily_returns', conn, if_exists='replace', index=False)

    # Store metrics
    df_metrics.reset_index().to_sql('sector_metrics', conn, if_exists='replace', index=False)

    print(f"  ✓ Data loaded into SQLite: {db_path}")

    # Execute and display key queries
    queries = {
        'Top sectors by Sharpe Ratio': """
            SELECT Sector,
                   "Ann. Return (%)" AS Annual_Return,
                   "Ann. Volatility (%)" AS Volatility,
                   "Sharpe Ratio",
                   "Max Drawdown (%)" AS Max_Drawdown
            FROM sector_metrics
            ORDER BY "Sharpe Ratio" DESC
            LIMIT 5;
        """,
        'Monthly average returns (top 3 defensive)': """
            SELECT Sector,
                   strftime('%Y-%m', Date) AS Month,
                   ROUND(AVG(Daily_Return) * 252 * 100, 2) AS Avg_Ann_Return_Pct
            FROM daily_returns
            WHERE Sector IN (
                SELECT Sector FROM sector_metrics
                ORDER BY "Sharpe Ratio" DESC LIMIT 3
            )
            GROUP BY Sector, Month
            ORDER BY Sector, Month
            LIMIT 15;
        """,
        'Correlation proxy — days both sectors were negative': """
            SELECT a.Sector AS Sector_A,
                   b.Sector AS Sector_B,
                   COUNT(*) AS Joint_Negative_Days,
                   ROUND(COUNT(*) * 100.0 / (
                       SELECT COUNT(DISTINCT Date) FROM daily_returns
                   ), 2) AS Pct_of_Trading_Days
            FROM daily_returns a
            JOIN daily_returns b
              ON a.Date = b.Date AND a.Sector < b.Sector
            WHERE a.Daily_Return < 0 AND b.Daily_Return < 0
              AND a.Sector != 'S&P 500' AND b.Sector != 'S&P 500'
            GROUP BY a.Sector, b.Sector
            ORDER BY Joint_Negative_Days DESC
            LIMIT 10;
        """,
    }

    for title, query in queries.items():
        print(f"\n  📊 {title}:")
        result = pd.read_sql_query(query, conn)
        print(result.to_string(index=False))

    conn.close()
    return queries


# ──────────────────────────────────────────────
# 6. EXPORT SQL QUERIES
# ──────────────────────────────────────────────
def export_sql_files():
    """Save standalone SQL query files for the portfolio."""
    print("\n" + "=" * 60)
    print("STEP 6: Exporting SQL Query Files")
    print("=" * 60)

    queries = {
        '01_sector_risk_ranking.sql': """-- Rank sectors by risk-adjusted performance
-- Uses Sharpe Ratio as primary metric with volatility and drawdown context

SELECT
    Sector,
    "Ann. Return (%)"      AS Annual_Return_Pct,
    "Ann. Volatility (%)"  AS Annual_Volatility_Pct,
    "Sharpe Ratio",
    "Sortino Ratio",
    "Max Drawdown (%)"     AS Max_Drawdown_Pct,
    "Beta",
    "Alpha (%)"            AS Alpha_Pct,
    RANK() OVER (ORDER BY "Sharpe Ratio" DESC) AS Sharpe_Rank,
    RANK() OVER (ORDER BY "Beta" ASC)           AS Beta_Rank,
    RANK() OVER (ORDER BY ABS("Max Drawdown (%)") ASC) AS Drawdown_Rank
FROM sector_metrics
ORDER BY "Sharpe Ratio" DESC;
""",
        '02_monthly_sector_returns.sql': """-- Calculate monthly aggregated returns per sector
-- Useful for seasonality and trend analysis

SELECT
    Sector,
    strftime('%Y', Date)   AS Year,
    strftime('%m', Date)   AS Month,
    COUNT(*)               AS Trading_Days,
    ROUND(AVG(Daily_Return) * 100, 4)   AS Avg_Daily_Return_Pct,
    ROUND(AVG(Daily_Return) * 252 * 100, 2) AS Annualized_Return_Pct,
    ROUND(MIN(Daily_Return) * 100, 2)   AS Worst_Day_Pct,
    ROUND(MAX(Daily_Return) * 100, 2)   AS Best_Day_Pct
FROM daily_returns
WHERE Sector != 'S&P 500'
GROUP BY Sector, Year, Month
ORDER BY Sector, Year, Month;
""",
        '03_downside_correlation.sql': """-- Identify which sector pairs tend to decline together
-- Measures joint negative days as a proxy for tail-risk correlation

SELECT
    a.Sector  AS Sector_A,
    b.Sector  AS Sector_B,
    COUNT(*)  AS Joint_Negative_Days,
    ROUND(COUNT(*) * 100.0 / (
        SELECT COUNT(DISTINCT Date) FROM daily_returns
    ), 2) AS Pct_of_Trading_Days
FROM daily_returns a
JOIN daily_returns b
  ON a.Date = b.Date AND a.Sector < b.Sector
WHERE a.Daily_Return < 0
  AND b.Daily_Return < 0
  AND a.Sector != 'S&P 500'
  AND b.Sector != 'S&P 500'
GROUP BY a.Sector, b.Sector
ORDER BY Joint_Negative_Days DESC;
""",
        '04_defensive_vs_benchmark.sql': """-- Compare defensive sector portfolio vs S&P 500 benchmark
-- Defensive sectors defined as top 3 by Sharpe Ratio

WITH defensive AS (
    SELECT Sector
    FROM sector_metrics
    ORDER BY "Sharpe Ratio" DESC
    LIMIT 3
),
daily_comparison AS (
    SELECT
        d.Date,
        AVG(CASE WHEN d.Sector IN (SELECT Sector FROM defensive)
            THEN d.Daily_Return END) AS Defensive_Avg_Return,
        AVG(CASE WHEN d.Sector = 'S&P 500'
            THEN d.Daily_Return END) AS SP500_Return
    FROM daily_returns d
    GROUP BY d.Date
)
SELECT
    strftime('%Y-%m', Date) AS Month,
    ROUND(AVG(Defensive_Avg_Return) * 252 * 100, 2) AS Defensive_Ann_Return_Pct,
    ROUND(AVG(SP500_Return) * 252 * 100, 2) AS SP500_Ann_Return_Pct,
    ROUND((AVG(Defensive_Avg_Return) - AVG(SP500_Return)) * 252 * 100, 2) AS Spread_Pct
FROM daily_comparison
GROUP BY Month
ORDER BY Month;
""",
        '05_volatility_regime.sql': """-- Classify market into volatility regimes and compare sector behavior
-- Uses 20-day rolling volatility of S&P 500 as regime indicator

WITH sp500_vol AS (
    SELECT
        Date,
        Daily_Return,
        AVG(Daily_Return * Daily_Return) OVER (
            ORDER BY Date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS Var_20d
    FROM daily_returns
    WHERE Sector = 'S&P 500'
),
regimes AS (
    SELECT
        Date,
        CASE
            WHEN SQRT(Var_20d * 252) > 0.20 THEN 'High Vol'
            WHEN SQRT(Var_20d * 252) > 0.12 THEN 'Medium Vol'
            ELSE 'Low Vol'
        END AS Vol_Regime
    FROM sp500_vol
)
SELECT
    r.Vol_Regime,
    d.Sector,
    COUNT(*) AS Days,
    ROUND(AVG(d.Daily_Return) * 252 * 100, 2) AS Ann_Return_Pct,
    ROUND(SQRT(AVG(d.Daily_Return * d.Daily_Return) * 252) * 100, 2) AS Ann_Vol_Pct
FROM daily_returns d
JOIN regimes r ON d.Date = r.Date
WHERE d.Sector != 'S&P 500'
GROUP BY r.Vol_Regime, d.Sector
ORDER BY r.Vol_Regime, Ann_Return_Pct DESC;
"""
    }

    for filename, sql in queries.items():
        path = os.path.join(SQL_DIR, filename)
        with open(path, 'w') as f:
            f.write(sql)
        print(f"  ✓ Saved: sql/{filename}")


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
def main():
    print("\n" + "▓" * 60)
    print("  S&P 500 SECTOR ETF ANALYSIS")
    print("  Identifying Defensive Sectors with Superior Risk-Adjusted Returns")
    print("▓" * 60)

    # Pipeline
    prices = fetch_data()
    daily_returns, cumulative_returns, rolling_returns = compute_returns(prices)
    df_metrics = compute_risk_metrics(daily_returns)
    top_3, df_ranked = identify_defensive_sectors(df_metrics)

    # Visualizations
    plot_correlation_heatmap(daily_returns)
    plot_rolling_returns(rolling_returns, top_3)
    plot_risk_return_scatter(df_metrics, top_3)
    plot_dashboard(prices, cumulative_returns, df_metrics, top_3)

    # SQL
    run_sql_analysis(daily_returns, df_metrics)
    export_sql_files()

    # Final summary
    print("\n" + "▓" * 60)
    print("  ANALYSIS COMPLETE")
    print("▓" * 60)
    print(f"\n  Top 3 Defensive Sectors: {', '.join(top_3)}")
    print(f"\n  Output files:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"    📄 output/{f}")
    print(f"\n  Data files:")
    for f in os.listdir(DATA_DIR):
        print(f"    📄 data/{f}")
    print(f"\n  SQL queries:")
    for f in sorted(os.listdir(SQL_DIR)):
        print(f"    📄 sql/{f}")

    return top_3, df_metrics


if __name__ == '__main__':
    top_3, metrics = main()
