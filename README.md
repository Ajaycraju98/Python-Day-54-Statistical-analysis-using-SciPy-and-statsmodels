# Python-Day-54-Statistical-analysis-using-SciPy-and-statsmodels
This project explains about the Statistical analysis using SciPy and stats models in python
pip install scipy statsmodels
# to deactivate warnings
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
#Import Libraries
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

Statistical Analysis with SciPy
Descriptive Statistics

data = np.array([10, 20, 30, 40, 50])
# Mean, Median, Standard Deviation
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)
print(f"Mean: {mean}, Median: {median}, Standard Deviation: {std_dev}")

Hypothesis Testing
One-Sample t-Test:
t_stat, p_val = stats.ttest_1samp(data, popmean=25)
print(f"t-statistic: {t_stat}, p-value: {p_val}")

Two-Sample t-Test:
group1 = [5, 15, 25, 35]
group2 = [10, 20, 30, 40]
t_stat, p_val = stats.ttest_ind(group1, group2)
print(f"t-statistic: {t_stat}, p-value: {p_val}")

Chi-Square Test
observed = np.array([50, 30, 20])
expected = np.array([40, 35, 25])
chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
print(f"Chi-square statistic: {chi2_stat}, p-value: {p_val}")

ANOVA Test
group1 = [5, 10, 15]
group2 = [20, 25, 30]
group3 = [35, 40, 45]
f_stat, p_val = stats.f_oneway(group1, group2, group3)
print(f"F-statistic: {f_stat}, p-value: {p_val}")

Statistical Analysis with statsmodels
Linear Regression Using statsmodels.api:
X = sm.add_constant([1, 2, 3, 4, 5])  # Add constant for intercept
y = [2, 4, 5, 4, 5]

model = sm.OLS(y, X).fit()
print(model.summary())

Using Formula API:
data = pd.DataFrame({'X': [1, 2, 3, 4, 5], 'y': [2, 4, 5, 4, 5]})
model = smf.ols('y ~ X', data=data).fit()
print(model.summary())

Logistic Regression

data = pd.DataFrame({'X': [1, 2, 3, 4, 5], 'y': [0, 1, 1, 0, 1]})
model = smf.logit('y ~ X', data=data).fit()
print(model.summary())

Time Series Analysis
# ARIMA Model
from statsmodels.tsa.arima.model import ARIMA

data = [266, 145, 183, 119, 180, 169, 232, 148, 123]
model = ARIMA(data, order=(1, 1, 1))
results = model.fit()
print(results.summary())

Hypothesis Testing with statsmodels

from statsmodels.stats.weightstats import ztest

data1 = [10, 20, 30]
data2 = [15, 25, 35]
z_stat, p_val = ztest(data1, value=25)
print(f"Z-statistic: {z_stat}, p-value: {p_val}")

Plotting Regression Results

import matplotlib.pyplot as plt

data = pd.DataFrame({'X': [1, 2, 3, 4, 5], 'y': [2, 4, 5, 4, 5]})
model = smf.ols('y ~ X', data=data).fit()

plt.scatter(data['X'], data['y'], color='blue', label='Data')
plt.plot(data['X'], model.predict(), color='red', label='Regression Line')
plt.legend()
plt.show()
