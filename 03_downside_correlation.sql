-- Identify which sector pairs tend to decline together
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
