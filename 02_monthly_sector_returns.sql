-- Calculate monthly aggregated returns per sector
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
