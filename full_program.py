
# Survival Analysis Project Code Template
# Includes KM and Cox model using lifelines (install required)

import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

km = KaplanMeierFitter()
km.fit(df['tenure'], event_observed=df['churn'])
km.plot_survival_function()
plt.savefig("km_plot.png")

df_encoded = pd.get_dummies(df, drop_first=True)
cph = CoxPHFitter()
cph.fit(df_encoded, 'tenure', 'churn')
cph.print_summary()
