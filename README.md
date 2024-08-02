# CO2 Emission Forecast with Python and Seasonal ARIMA

## Project Overview

This project demonstrates how to forecast CO2 emissions using a Seasonal ARIMA (SARIMA) model. The steps include data retrieval, transformation to a time series, stationarity testing, model parameter optimization, and forecasting future emissions.

## Files in the Repository

-\`co2-emission-forecast-with-python-seasonal-arima.ipynb\`: This notebook contains the code for the entire project including data loading, preprocessing, model building, and forecasting.

-\`CO2\_emission.ipynb\`: This notebook contains additional code and output that might be useful for understanding the process and results.

## Dependencies

To run the notebooks, you need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- itertools
- warnings

You can install these dependencies using pip:

```
pip install pandas numpy matplotlib seaborn statsmodels
```

# Usage

* ​**Loading the Dataset**​: Load the CO2 emissions dataset in CSV format.
* ​**Exploratory Data Analysis**​: Visualize the time series data to understand its characteristics.
* ​**Checking Stationarity**​: Use the Augmented Dickey-Fuller test to check if the time series is stationary.
* ​**Seasonal Decomposition**​: Decompose the time series into trend, seasonal, and residual components.
* ​**Building the SARIMA Model**​:
  * Define parameter ranges for the SARIMA model.
  * Use grid search to find the optimal parameters.
  * Fit the SARIMA model with the optimal parameters.
* ​**Forecasting**​:
  * Generate in-sample predictions and plot them.
  * Generate out-of-sample forecasts and plot them.
* ​**Conclusion**​: Summarize the findings and potential future work.

# Detailed Steps

### 1. Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings
warnings.filterwarnings("ignore")
```

### 2. Loading the dataset

```python
data = pd.read_csv('path_to_csv')
data.head()
```

### 3. Exploratory Data Analysis

#### Plotting the Data

```python
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['CO2'])
plt.title('CO2 Emissions Over Time')
plt.xlabel('Date')
plt.ylabel('CO2 Emissions')
plt.show()
```

#### Checking Stationarity

```python
result = adfuller(data['CO2'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

#### Differencing to Achieve Stationarity

```python
data['CO2_diff'] = data['CO2'].diff().dropna()
result = adfuller(data['CO2_diff'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

### 4. Seasonal Decomposition

```python
decomposition = sm.tsa.seasonal\_decompose(data['CO2'], model='additive', period=12) decomposition.plot() plt.show()
```

### 5. Building the SARIMA Model

#### Defining Parameter Ranges

```python
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
pdq_x_QDQs = [(x[0], x[1], x[2], 12) for x in pdq]
print('Examples of Seasonal ARIMA parameter combinations:')
print('SARIMAX: {} x {}'.format(pdq[1], pdq_x_QDQs[1]))
print('SARIMAX: {} x {}'.format(pdq[2], pdq_x_QDQs[2]))
```

#### Grid Search

```python
for param in pdq:
    for param_seasonal in pdq_x_QDQs:
        try:
            mod = sm.tsa.statespace.SARIMAX(data['CO2'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

```

### 6. Fitting the Optimal Model

```python
mod = sm.tsa.statespace.SARIMAX(data['CO2'],
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary())

```

### 7. Forecasting

#### In-sample prediction

```python
pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = data.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
plt.xlabel('Date')
plt.ylabel('CO2 Emissions')
plt.legend()
plt.show()

```

#### Out-of-sample Forecasting

```python
forecast = results.get_forecast(steps=120)
forecast_ci = forecast.conf_int()
ax = data.plot(label='observed', figsize=(14, 7))
forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Emissions')
plt.legend()
plt.show()

```

### 8. Conclusion

This notebook demonstrated the process of forecasting CO2 emissions using a Seasonal ARIMA model. We explored the dataset, checked for stationarity, decomposed the time series, optimized the model parameters using grid search, and made forecasts. The results indicate a continued increase in CO2 emissions.







