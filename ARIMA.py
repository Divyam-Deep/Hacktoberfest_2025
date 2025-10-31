import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Create synthetic monthly sales data
np.random.seed(42)
months = pd.date_range(start='2020-01-01', periods=36, freq='M')
sales = 200 + np.random.randn(36).cumsum() * 10
df = pd.DataFrame({'Month': months, 'Sales': sales}).set_index('Month')

# Train-test split
train = df.iloc[:-6]
test = df.iloc[-6:]

# Build ARIMA model
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=6)
plt.plot(train.index, train['Sales'], label='Train')
plt.plot(test.index, test['Sales'], label='Actual')
plt.plot(test.index, forecast, label='Forecast', linestyle='--')
plt.legend()
plt.title("Sales Forecast using ARIMA")
plt.show()

print("Forecasted Values:\n", forecast)
