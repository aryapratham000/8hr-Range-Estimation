***This project implements a robust modeling pipeline to estimate the 8-hour range of the ES futures contract using engineered market structure features, statistical regressors, and ensemble learning techniques. It explains over 62% of the variation in range behavior (R² = 0.624), with Gradient Boosting achieving a 49.8% improvement in MAE over a naive 3rd-bar persistence baseline (10.59 vs. 21.07), demonstrating the value of momentum features and explicit session context.***

***It is designed to support systematic trading, risk management, and intraday range forecasting based on session-level behavior.***

---

### 0. Data Loading & Cleaning
- Raw 8-hour ES candle data (2015–2025) is loaded and preprocessed
- Columns include Open, High, Low, Close, Volume, and timestamp metadata
- Null or malformed rows are dropped
- DateTime is set as the index for time-aware modeling

---

### 1. Core Feature Engineering  
The model uses market structure, session context, and volatility dynamics to model not just range magnitude but its conditional drivers and timing

- **range:** Measures the high–low spread of the bar. This is the target variable being predicted
- **side:** Binary label for candle direction: 1 if close ≥ open, else 0. Helps gauge directional skew
- **dayofweek:** Encodes weekly seasonality — e.g., trend continuation early week, mean-reversion into Friday
- **session:** Labels each bar as "AS" (Asia), "AM" (US Morning), or "PM" (US Afternoon). Captures distinct volatility regimes based on global market participation
- **stoch_k (Stochastic %K):** Positions the current close within the recent 111-bar high-low range. Useful for detecting sentiment extremes and overbought/oversold conditions
- **pattern (Candle structure classification):** Two-part code combining range structure and close location
  - Example: OU = outside bar with a bullish close (expansion with directional intent)
- **n_highs_taken_prev & n_lows_taken_prev:** Count how many structural highs/lows were broken in the recent past. Helps quantify breakout pressure or trend exhaustion behavior

---

### 2. Data Exploration
This section enables interactive exploration of the ES 8-hour range distribution across time windows, sessions, candle directions, and pattern types. Users can select filters using dropdown widgets for:

- Look-back window (in months)
- Session (AM, PM, AS)
- Candle direction (Side = 1 for up, Side = 0 for down)
- Candle pattern (IH, DL, etc.)

Based on the selected filters, the notebook dynamically generates:
- A descriptive statistics table for the range, including percentiles (25th, median, 75th, 90th)
- A histogram plot showing distribution of ranges within the selected segment
- Contextual insights for each selection to guide pattern recognition and hypothesis formation

⚠️ Note: The dropdowns and interactive widgets use ipywidgets, which do not render on GitHub. To use the interactive filtering tools, please open this notebook locally in JupyterLab, VS Code, or Google Colab
  
🔍 Key Observations
- The range distribution is right-skewed — most candles are small, but large outliers pull the mean above the median
- Recent (24-month) windows show increased volatility, reflecting macro regime shifts
- AM sessions show balanced ranges; PM sessions exhibit higher variance due to U.S. close activity; AS sessions remain muted and range-bound
- Down candles (Side = 0) tend to have larger ranges than up candles, suggesting stronger downside momentum

---

### 3. One-Hot Encoding
Categorical features (session, pattern, dayofweek) are one-hot encoded for model compatibility using pandas.get_dummies

---

### 4. News Event Flags 
High-impact U.S. economic events (CPI, FOMC, NFP) are flagged per 8-hour bar using a timestamp-aligned CSV to capture news-driven volatility effects

---

### 5. Lag Features
Adds lagged features for range, direction, and patterns to provide the model short-term memory of recent volatility and structure; aligns stochastic oscillator to prevent look-ahead bias

---

### 6. Train-Test Split
- Top 5% of extreme range values are excluded to reduce heteroskedastic distortion and improve model stability
- Date filtering: Only data from 2021 onward is used to reflect recent market regimes
- Chronological split: Dataset is split at Jan 2024 to preserve forward-looking temporal causality

---

### 7. Linear Models
Multiple linear regressors were tested, including:
- OLS (Ordinary Least Squares): baseline model with intercept
- Ridge: L2 regularization to stabilize coefficient magnitudes
- Lasso: L1 penalty for feature selection and sparsity
- Huber: robust regression that downweights outliers and adapts to heteroskedasticity

OLS results:
- Model explains 62% of the variation in 8-hour range (R² = 0.624)
- F-statistic (29.15, p < 1e-179) confirms statistical significance of the model
- Residuals show tight, homoskedastic spread, validating noise reduction via outlier trimming

Key signal insights:
- Lagged range features (range_m3, range_m6) show highest predictive power
- Stochastic momentum (stoch_k) has strong inverse correlation with next-bar range (suggesting mean-reversion)
- Session timing (sess_AS, sess_PM) and Sunday overnight (dow_6) show dampening effects
- CPI events trend toward significance, hinting at news-driven volatility clusters

---

### 8. Random Forest
A Random Forest Regressor with 333 estimators was trained to capture potential non-linear interactions  
- Performance: MAE = 10.77 pts, slightly worse than OLS (10.30 pts)

Interpretation:
- The target variable (8-hour range) exhibits strong linear structure, which linear models like OLS capture effectively
- Despite RF’s ability to model complex interactions, it doesn’t offer a performance edge here, possibly due to the signal being predominantly additive and low in nonlinearity

---

### 9. Gradient Boosting (Point & Quantile Forecasting) 
Performance:
Gradient Boosting achieves a MAE of 10.59 pts, showing stable generalization and outperforming most linear and tree-based baselines

Quantile Estimation:
Three quantile regressors (τ = 0.5, 0.75, 0.9) are trained to model the distributional range boundaries using pinball loss, allowing for interval estimation

Coverage Calibration:
Empirical coverage falls short of target quantiles:
- τ = 0.5 → 45.3% (vs. 50%)
- τ = 0.75 → 67.3% (vs. 75%)
- τ = 0.9 → 78.5% (vs. 90%)  
→ Indicates systematic undercoverage, especially in high-volatility conditions

Residual Behavior:
Residuals widen as predicted ranges increase, suggesting conditional heteroskedasticity

Model Fit Diagnostics:
- Actual vs. Predicted: Captures center mass well, but increasingly underpredicts as true ranges exceed ~50 pts
- Residuals vs. Predicted: Residual spread increases with prediction magnitude — hinting at potential non-linear limits
- Residuals Over Time: Displays stable error spread, with no temporal drift, supporting model robustness

---

### 10. SARIMAX 
SARIMAX MAE = 10.38 pts, matching OLS — confirming that most of the temporal structure is already captured by engineered lag features used as exogenous regressors.

---

### 11. Final Benchmarking

***Model Insights***
- OLS and SARIMAX tie for best point-wise accuracy at 10.38 MAE, confirming that engineered lagged-range, session, and weekday features already absorb most serial dependence
- Ridge and Huber models follow closely (+0.1 pts), indicating that the core signal is robust across regularized linear variants
- Lasso underperforms (12.67 MAE) due to forced sparsity — dropping informative lags and weakening predictive structure
- Gradient Boosting (GBR) achieves 10.59 MAE, just ~2% off linear models, while outperforming naive baselines significantly:
  -49.8% lower MAE vs “Naive: 3rd Last Bar” (10.59 vs 21.07)
- ATR(10) shrinks error compared to raw last-bar persistence, but still trails GBR by ~28%, reaffirming the value of context-aware modeling

***Permutation Importance (GBR)***
Top drivers of predictive power:
- sess_AS (Asian session), stoch_k (momentum), range_m3, and dow_6 (Sunday) dominate signal
- Liquidity metrics like n_lows_taken_prev add valuable structure detail
- Candle pattern lags (pat_UH_m1, side_m1) add weak but non-trivial signal enrichment

***Action Path & Next Steps***
- Anchor on OLS for transparent linear baseline
- Deploy GBR in dashboard to support confidence bands and probabilistic overlays
- Improve calibration: use post-hoc tuning (e.g. isotonic, conformal) to fix quantile undercoverage
- Integrate adaptive forecasting into a Streamlit dashboaord, continuously updating based on real-time session dynamics

























