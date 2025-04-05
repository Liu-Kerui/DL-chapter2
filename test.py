import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import seaborn as sns

# 生成模拟数据
np.random.seed(42)

# 假设我们有两个组：High SUVmean (>3) 和 Low SUVmean (<=3)
# 每组有50个数据点，模拟生存时间和是否存活
n_samples = 80
high_suvmean = np.random.normal(10, 2, n_samples)  # High SUVmean组，代谢活性较高的患者
low_suvmean = np.random.normal(2, 0.5, n_samples)  # Low SUVmean组，代谢活性较低的患者

# 生存时间（以月为单位）及是否生存（1为存活，0为死亡）
# 假设高SUVmean组生存期较短，低SUVmean组生存期较长
high_suvmean_survival = np.random.normal(10, 3, n_samples)
low_suvmean_survival = np.random.normal(20, 5, n_samples)

# 添加生存状态，假设高SUVmean组更多人死亡
high_suvmean_event = np.random.binomial(1, 0.8, n_samples)  # 80%死亡
low_suvmean_event = np.random.binomial(1, 0.3, n_samples)   # 30%死亡

# 创建数据框架
high_suvmean_df = pd.DataFrame({
    'survival_time': high_suvmean_survival,
    'event': high_suvmean_event,
    'group': 'High SUVmean'
})

low_suvmean_df = pd.DataFrame({
    'survival_time': low_suvmean_survival,
    'event': low_suvmean_event,
    'group': 'Low SUVmean'
})

# 合并为一个数据集
data = pd.concat([high_suvmean_df, low_suvmean_df])

# Kaplan-Meier 生存曲线分析
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))

# 绘制 Kaplan-Meier 曲线
for group in data['group'].unique():
    group_data = data[data['group'] == group]
    kmf.fit(group_data['survival_time'], event_observed=group_data['event'], label=group)
    kmf.plot(ci_show=True)

plt.title('Kaplan-Meier Survival Curves for High and Low SUVmean')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid(True)
plt.show()

# SUVmean与治疗反应的箱线图
# 模拟治疗反应：Responders 和 Non-Responders
response_groups = ['Responders', 'Non-Responders']
responders = np.random.normal(2, 0.5, 30)  # 响应者组，SUVmean较低
non_responders = np.random.normal(6, 1.5, 30)  # 非响应者组，SUVmean较高

# 合并数据
response_data = pd.DataFrame({
    'SUVmean': np.concatenate([responders, non_responders]),
    'Response': ['Responders'] * 30 + ['Non-Responders'] * 30
})

# 绘制箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(x='Response', y='SUVmean', data=response_data)
plt.title('SUVmean in Responders vs Non-Responders')
plt.xlabel('Treatment Response')
plt.ylabel('SUVmean')
plt.grid(True)
plt.show()
