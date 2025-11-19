# Survival Analysis Report (Simulated Data)
Generated: 2025-11-19T07:45:07.606431 UTC

## Dataset
- Rows: 1000
- Columns: 6
- Key columns: tenure (duration), churn (event), monthly_charges, contract_type, internet_service, gender

## EDA Summary (auto-generated)
- Numeric summaries saved in `eda_summary.txt`.
- Monthly charges: mean=73.37, median=72.53

## Kaplan-Meier
- Plots: `km_by_contract_type.png`, `km_by_charge_quartile.png`
- Compare survival across `contract_type` and monthly charge quartiles.

## Cox PH model
- Script fits a Cox model with predictors: monthly_charges + contract_type + internet_service + gender.
- Summary saved: `cox_summary.csv`
- Top predictors saved: `top_predictors.csv`

## Notes
This package uses a simulated dataset. Replace `dataset.csv` with the real file and run `run_analysis.py` or the included notebook to reproduce results on real data.
