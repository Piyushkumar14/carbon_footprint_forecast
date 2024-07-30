import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle

# Load data
def load_data():
    data = pd.read_csv('data/MER_T12_06.csv', usecols=[1, 2], names=['Date', 'CO2_Emissions'], skiprows=1)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m', errors='coerce')
    data.set_index('Date', inplace=True)
    return data

data = load_data()

# Load forecast data
def load_forecast_data():
    forecast_data = pd.read_csv('data/forecast_ci.csv')
    if 'Date' not in forecast_data.columns:
        raise KeyError("The 'Date' column is missing from the forecast data.")
    if 'index' in forecast_data.columns:
        forecast_data.rename(columns={'index': 'Date'}, inplace=True)
    try:
        forecast_data['Date'] = pd.to_datetime(forecast_data['Date'], errors='coerce')
    except Exception as e:
        raise ValueError(f"Error converting 'Date' column to datetime: {e}")

    forecast_data = forecast_data.loc[:, ~forecast_data.columns.str.contains('^Unnamed')]
    return forecast_data

forecast_data = load_forecast_data()
forecast_data.set_index('Date', inplace=True)

# Ensure data types are numeric
data['CO2_Emissions'] = pd.to_numeric(data['CO2_Emissions'], errors='coerce')
forecast_data = forecast_data.apply(pd.to_numeric, errors='coerce')

# Handle missing values
data.dropna(inplace=True)
forecast_data.dropna(inplace=True)

# Debug: Print data to verify
print(data.head())
print(forecast_data.head())

# Display data
st.title('CO2 Emission Forecasting')
st.write('## Original Data')
st.line_chart(data, height=300)

# Load model
def load_model():
    with open('sarima_model.pkl', 'rb') as pkl_file:
        loaded_model = pickle.load(pkl_file)
    return loaded_model

loaded_model = load_model()

# Forecast
forecast_steps = st.slider('Select number of months to forecast', 1, 36, 12)
forecast = loaded_model.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='M')[1:]
forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['Forecast'])
forecast_ci = forecast.conf_int()

# Display forecast
st.write('## Forecasted Data')
st.line_chart(forecast_df)

# Plot original and forecasted data with confidence intervals
fig, ax = plt.subplots(figsize=(20, 15))
data.plot(ax=ax, label='Observed')
forecast_df.plot(ax=ax, label='Forecast')
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='g', alpha=.4)
ax.set_xlabel('Time (year)')
ax.set_ylabel('NG CO2 Emission level')
plt.legend()
st.pyplot(fig)