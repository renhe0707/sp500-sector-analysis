-- Classify market into volatility regimes and compare sector behavior
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
