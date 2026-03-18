-- Compare defensive sector portfolio vs S&P 500 benchmark
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
