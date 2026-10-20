## Macroeconomic GDP Forecasting Using Ensemble Learning and XAI

![Python](https://img.shields.io/badge/python-3%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble-orange)
![Econometrics](https://img.shields.io/badge/Econometrics-Time%20Series-success)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)

## Project Overview
This repository implements an end-to-end machine learning and econometric pipeline for forecasting macroeconomic trends. Utilizing annual time-series data from the World Bank (1991–2024), the project addresses the inherent challenges of modeling small-sample economic panels ($N=34$). By combining rigorous statistical preprocessing with strictly regularized ensemble models, the framework mitigates spurious regression and prevents extreme overfitting, ultimately providing transparent, non-parametric insights via SHapley Additive exPlanations (SHAP).

## Dataset
The dataset comprises $N=34$ annual observations (1991–2024) of key macroeconomic indicators sourced from the World Bank. 

**Core Variables:**
* `UNEMP`: Unemployment Rate (%)
* `GDPG`: Gross Domestic Product Growth (%)
* `INF`: Inflation Rate (%)
* `INT`: Real Interest Rate (%)
* `LABF`: Labor Force Total (Persons)
* `FDI`: Foreign Direct Investment (USD)
* `EXR`: Exchange Rate (Local Currency per USD)
* `GEXP`: Government Expenditure (% of GDP)
* `TO`: Trade Openness (% of GDP)
* `POPG`: Population Growth Rate (%)

## Methodology Workflow
Our hybrid pipeline spans classical econometrics and modern machine learning:

1. **Exploratory Data Analysis (EDA):**
   * **Outlier Management:** Interquartile Range (IQR) bounds.
   * **Stationarity:** Augmented Dickey-Fuller (ADF) testing to identify and difference $I(1)$ series (e.g., Labor Force, Trade Openness) into stationary $I(0)$ inputs.
   * **Multicollinearity:** Variance Inflation Factor (VIF) filtering to ensure all predictors maintain a VIF $< 5.0$.
2. **Feature Engineering:** Hybrid lag selection utilizing AIC, BIC, and HQIC to capture short-run temporal transmissions while maintaining model parsimony.
3. **Ensemble Modeling:** Evaluating SVM, Decision Trees, Random Forest, ExtraTrees, LightGBM, and XGBoost.
4. **Validation:** Expanding-window walk-forward cross-validation (`TimeSeriesSplit`) to completely eliminate look-ahead bias.
5. **Explainable AI (XAI):** TreeSHAP integration for global dimensionality reduction and feature attribution.

## Key Findings & Results
* **Optimal Architecture:** Unconstrained non-linear models natively overfit the idiosyncratic $I(0)$ shocks of this small sample. **XGBoost** proved superior ($RMSE = 1.6353$, $R^2 = -0.0995$).
* **Economic Insights via SHAP:** 
  * The model organically pruned noisy transmission channels (like Exchange Rates and FDI) to exactly zero impact ($\phi_i = 0$).
  * Lagged labor market dynamics (`LABF_DIFF_lag1` and `UNEMP_DIFF_lag1`) emerged as the dominant structural drivers.
  * We identified a distinct **capital-labor dilution effect**, where rapid exogenous labor force expansions in the preceding year place downward pressure on contemporaneous GDP growth.

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/macroeconomic-forecasting.git
cd macroeconomic-forecasting
pip install -r requirements.txt
```

## Usage
*Provide brief instructions or code snippets on how to run your data pipeline, train the models, and generate the SHAP plots.*

```bash
# Example: Run the full EDA, training, and evaluation pipeline
python main.py --config configs/xgboost_config.yaml
```

## Repository Structure
```text
├── data/                  # Raw and processed World Bank datasets
├── notebooks/             # Jupyter notebooks for EDA and SHAP visualizations
├── src/
│   ├── preprocessing.py   # ADF tests, differencing, VIF checks
│   ├── features.py        # Lag generation and selection
│   ├── models.py          # Model architectures and Walk-Forward CV
│   └── evaluate.py        # Metrics calculations and SHAP plots
├── requirements.txt       # Project dependencies
└── README.md
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
