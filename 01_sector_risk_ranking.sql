-- Rank sectors by risk-adjusted performance
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
