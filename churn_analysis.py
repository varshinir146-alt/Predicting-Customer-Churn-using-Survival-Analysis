
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

# Simulate dataset
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'tenure': np.random.exponential(scale=12, size=n).astype(int),
    'churn': np.random.binomial(1, 0.4, n),
    'age': np.random.randint(18, 70, n),
    'monthly_charges': np.random.randint(300, 1500, n),
    'gender': np.random.choice(['Male', 'Female'], size=n),
    'contract_type': np.random.choice(['Monthly', 'Yearly'], size=n),
    'internet_service': np.random.choice(['Fiber', 'DSL', 'None'], size=n)
})

df_encoded = pd.get_dummies(df, drop_first=True)

# Kaplan-Meier Model
km = KaplanMeierFitter()
km.fit(df['tenure'], event_observed=df['churn'])
km.plot_survival_function()
plt.title("Kaplan-Meier Survival Curve")
plt.xlabel("Tenure")
plt.ylabel("Survival Probability")
plt.savefig("km_plot.png")

# Cox Proportional Hazards Model
cph = CoxPHFitter()
cph.fit(df_encoded, duration_col='tenure', event_col='churn')
cph.print_summary()

# Save dataset
df.to_csv("simulated_churn_dataset.csv", index=False)
