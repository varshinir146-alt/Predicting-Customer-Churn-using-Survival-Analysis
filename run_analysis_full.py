"""
run_analysis_full.py
Complete pipeline: load dataset.csv, EDA, Kaplan-Meier plots, Cox PH model,
Schoenfeld residuals check, save outputs (requires: pandas, numpy, matplotlib, lifelines, seaborn)
"""

import pandas as pd, numpy as np, os, json
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv('dataset.csv')
DURATION = 'tenure'; EVENT = 'churn'
df = df.dropna(subset=[DURATION, EVENT]).copy()
df[DURATION] = pd.to_numeric(df[DURATION], errors='coerce')
df[EVENT] = pd.to_numeric(df[EVENT], errors='coerce')
df = df.dropna(subset=[DURATION, EVENT])

# EDA
eda = {'shape': df.shape, 'missing': df.isna().sum().to_dict()}
with open(os.path.join(OUT_DIR,'eda_summary.json'),'w') as f:
    json.dump(eda, f, indent=2)
with open(os.path.join(OUT_DIR,'eda_summary.txt'),'w') as f:
    f.write(str(eda))

# KM by contract_type
kmf = KaplanMeierFitter()
if 'contract_type' in df.columns:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    for grp,gdf in df.groupby('contract_type'):
        kmf.fit(gdf[DURATION], event_observed=gdf[EVENT], label=grp)
        kmf.plot_survival_function(ci_show=True)
    plt.title('KM by contract_type')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'km_by_contract_type.png')); plt.close()

# KM by monthly charge quartile
if 'monthly_charges' in df.columns:
    df['_charge_bin'] = pd.qcut(df['monthly_charges'].fillna(df['monthly_charges'].median()), q=4, labels=['Q1','Q2','Q3','Q4'])
    plt.figure(figsize=(8,6))
    for grp,gdf in df.groupby('_charge_bin'):
        kmf.fit(gdf[DURATION], event_observed=gdf[EVENT], label=str(grp))
        kmf.plot_survival_function(ci_show=False)
    plt.title('KM by charge quartile'); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'km_by_charge_quartile.png')); plt.close()

# Log-rank example
logrank = {}
if 'contract_type' in df.columns:
    groups = df['contract_type'].unique()
    if len(groups)>=2:
        a = df[df['contract_type']==groups[0]]
        b = df[df['contract_type']==groups[1]]
        res = logrank_test(a[DURATION], b[DURATION], event_observed_A=a[EVENT], event_observed_B=b[EVENT])
        logrank['%s_vs_%s'%(groups[0],groups[1])] = {'p': float(res.p_value), 'stat': float(res.test_statistic)}
with open(os.path.join(OUT_DIR,'logrank.json'),'w') as f:
    json.dump(logrank,f,indent=2)

# Cox model
predictors = [c for c in ['monthly_charges','contract_type','internet_service','gender'] if c in df.columns]
model_df = df[[DURATION, EVENT] + predictors].copy()
model_df = pd.get_dummies(model_df, drop_first=True)
model_df = model_df.rename(columns={DURATION:'duration', EVENT:'event'}).dropna()

cph = CoxPHFitter()
cph.fit(model_df, duration_col='duration', event_col='event')
cph.summary.to_csv(os.path.join(OUT_DIR,'cox_summary.csv'))
cph.summary.reset_index().to_json(os.path.join(OUT_DIR,'cox_summary.json'),orient='records')
cph.print_summary()

# PH tests
try:
    cph.check_assumptions(model_df, p_value_threshold=0.05, show_plots=False)
except Exception as e:
    with open(os.path.join(OUT_DIR,'check_assumptions_error.txt'),'w') as f:
        f.write(str(e))
from lifelines.statistics import proportional_hazard_test
ph = proportional_hazard_test(cph, model_df, time_transform='rank')
with open(os.path.join(OUT_DIR,'ph_test.txt'),'w') as f:
    f.write(str(ph))

# Top predictors
top = cph.summary.reset_index().sort_values(by='exp(coef)', ascending=False)
top[['covariate','coef','exp(coef)','p']].head(10).to_csv(os.path.join(OUT_DIR,'top_predictors.csv'), index=False)

print('Done. Check outputs/ for files.')