import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error

# Load data
# data = pd.read_csv('./data/GBPUSD=X.csv')
data = pd.read_csv('./data/GBPUSD=X.csv')

# Calculate daily returns
data['returns'] = abs(100 * (data['Close'].pct_change()))


# Calculate the current 20-day standard deviation of daily returns
data['current_vol'] = data.returns.rolling(20).std()

# Calculate the 20-day standard deviation of daily returns for 20 days ahead
# Used to evaluate the forecasts
data['target_vol'] = data.returns.shift(-20).rolling(20).std()
data.dropna(inplace=True)


# Get the indexes and values in the test set
test_size = 365
test_idx = data.index[len(data) - test_size:]
y_test = data.target_vol[test_idx]

# Calculate EWMA of 20-day standard deviations
data['EWMA12'] = data['current_vol'].ewm(halflife=0.97).mean()

# Plot the forecasts afainst the 20-day standard deviations
fig,ax = plt.subplots(figsize=(10,4))
plt.plot(data['target_vol'][-test_size:], alpha=0.5, label="Realised Volatility")
plt.plot(data['EWMA12'][-test_size:], color = 'black', label="EWMA (\u03BB = 0.97)", alpha = 0.75)
plt.title('EWMA Forecast of GBP/USD Volatility')
plt.legend(loc="upper right")
# plt.show()
plt.savefig('./plots/EWMA_GBPUSD.png')


# Calculate RMSE and MAE of forecasts
RMSE = math.sqrt(np.square(np.subtract(data['EWMA12'].tail(test_size),data['target_vol'].tail(test_size))).mean() )
MAE = mean_absolute_error(data['EWMA12'].tail(test_size),data['target_vol'].tail(test_size))

print("Root Mean Square Error:")
print(RMSE)
print()

print("Mean Absolute Error:")
print(MAE)
print()