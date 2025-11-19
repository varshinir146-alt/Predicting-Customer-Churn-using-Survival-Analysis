"""
Completed project script: Survival Analysis (Kaplan-Meier & Cox PH)
Save this file as run_analysis.py and run in an environment with pandas, numpy, matplotlib, lifelines installed.
Outputs: km plots, cox summary CSVs, eda_summary.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('dataset.csv')

# Basic cleaning and mapping
DURATION = 'tenure'
EVENT = 'churn'
df = df.dropna(subset=[DURATION, EVENT])

# EDA summary
eda_lines = []
eda_lines.append(f"Dataset shape: {df.shape}")
missing = df.isna().sum()
for col, m in missing.items():
    if m>0:
        eda_lines.append(f"{col}: {m} missing")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    s = df[c].dropna()
    eda_lines.append(f"Numeric {c}: mean={s.mean():.2f}, median={s.median():.2f}, std={s.std():.2f}")

with open('eda_summary.txt','w') as f:
    f.write('\\n'.join(eda_lines))

# Kaplan-Meier by contract_type
kmf = KaplanMeierFitter()
plt.figure(figsize=(8,6))
for grp, gdf in df.groupby('contract_type'):
    kmf.fit(gdf[DURATION], gdf[EVENT], label=grp)
    kmf.plot_survival_function(ci_show=True)
plt.title('Kaplan-Meier by contract_type')
plt.xlabel('Time')
plt.ylabel('Survival probability')
plt.grid(True)
plt.savefig('km_by_contract_type.png')
plt.close()

# Kaplan-Meier by monthly_charges quartile
df['_charge_bin'] = pd.qcut(df['monthly_charges'], q=4, labels=['Q1','Q2','Q3','Q4'])
plt.figure(figsize=(8,6))
for grp, gdf in df.groupby('_charge_bin'):
    kmf.fit(gdf[DURATION], gdf[EVENT], label=str(grp))
    kmf.plot_survival_function(ci_show=False)
plt.title('Kaplan-Meier by monthly_charges quartile')
plt.xlabel('Time')
plt.ylabel('Survival probability')
plt.grid(True)
plt.savefig('km_by_charge_quartile.png')
plt.close()

# Prepare data for Cox model
predictors = ['monthly_charges','contract_type','internet_service','gender']
model_df = df[[DURATION, EVENT] + predictors].copy()
model_df = pd.get_dummies(model_df, drop_first=True)
model_df = model_df.rename(columns={DURATION:'duration', EVENT:'event'})

# Fit Cox PH model
cph = CoxPHFitter()
cph.fit(model_df, duration_col='duration', event_col='event', step_size=0.1)
cph.print_summary()
cph.summary.to_csv('cox_summary.csv')

# Check PH assumption
cph.check_assumptions(model_df, p_value_threshold=0.05, show_plots=False)
results = proportional_hazard_test(cph, model_df, time_transform='rank')
with open('ph_test.txt','w') as f:
    f.write(str(results))

# Save top predictors
top = cph.summary.reset_index().sort_values(by='exp(coef)', ascending=False)
top[['covariate','coef','exp(coef)','p']].head(10).to_csv('top_predictors.csv', index=False)

print('Analysis complete. Files saved: eda_summary.txt, km_by_contract_type.png, km_by_charge_quartile.png, cox_summary.csv, top_predictors.csv, ph_test.txt')
