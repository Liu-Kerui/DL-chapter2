import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Sample data generation (replace this with your actual data)
# np.random.seed(42)

# Simulated data for patients (replace with actual values)
data = {
    'tumor_volume': np.random.uniform(10, 100, 100),  # Tumor volume in cm³
    'SUVmean': np.random.uniform(2, 6, 100),  # SUVmean values
    'survival_time': np.random.uniform(0, 10, 100),  # Time until event (in years)
    'event_observed': np.random.choice([1, 0], 100)  # 1 if the event (death) occurred, 0 if censored
}

# Create DataFrame
df = pd.DataFrame(data)

# Define tumor volume and SUVmean cutoffs
tumor_volume_cutoff = 50  # Tumor volume > 50 cm³ considered high
SUVmean_cutoff = 4.5  # SUVmean > 4.5 considered high

# Create groups for tumor volume and SUVmean
df['tumor_volume_group'] = np.where(df['tumor_volume'] > tumor_volume_cutoff, 'High', 'Low')
df['SUVmean_group'] = np.where(df['SUVmean'] > SUVmean_cutoff, 'High', 'Low')

# Kaplan-Meier Fitter
kmf = KaplanMeierFitter()

# Plot Kaplan-Meier for Tumor Volume
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
for group in ['High', 'Low']:
    mask = df['tumor_volume_group'] == group
    kmf.fit(df['survival_time'][mask], event_observed=df['event_observed'][mask], label=group)
    kmf.plot()

plt.title("Kaplan-Meier Curve for Tumor Volume")
plt.xlabel("Time (years)")
plt.ylabel("Survival Probability")
plt.legend(title="Tumor Volume")

# Plot Kaplan-Meier for SUVmean
plt.subplot(1, 2, 2)
for group in ['High', 'Low']:
    mask = df['SUVmean_group'] == group
    kmf.fit(df['survival_time'][mask], event_observed=df['event_observed'][mask], label=group)
    kmf.plot()

plt.title("Kaplan-Meier Curve for SUVmean")
plt.xlabel("Time (years)")
plt.ylabel("Survival Probability")
plt.legend(title="SUVmean")

# Show plots
plt.tight_layout()
plt.show()
