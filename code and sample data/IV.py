import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_absolute_error


# Load data
# Apologies that this is a little messy but the two data files 
# are sorted in different ways so the following is necssary to 
# match up the dates
currencies = pd.read_csv('./data/GBPUSD=X.csv')
currencies['Date']= pd.to_datetime(currencies['Date'], infer_datetime_format=True)
currencies = currencies.fillna(method='ffill')

iv = pd.read_csv('./data/IV_DATA.csv')
iv['Date']= pd.to_datetime(iv['Date'], infer_datetime_format=True)

iv = iv.iloc[::-1]
iv = iv.set_index('Date')
iv = iv.fillna(method='ffill')
currencies = currencies.set_index('Date')

currencies = currencies[currencies.index.isin(iv.index)]
iv = iv[iv.index.isin(currencies.index)]


df = pd.DataFrame()

df['returns'] = abs(100 * (currencies['Close'].pct_change()))

# Calculate the current 20-day standard deviation of daily returns
df['current_vol'] = df.returns.rolling(20).std()

# Calculate the 20-day standard deviation of daily returns for 20 days ahead
# Used to evaluate the forecasts
df['target_vol'] = df.returns.shift(-20).rolling(20).std()
df.dropna(inplace=True)

# Get the scaled 20 day implied standard deviation
df['implied_vol'] = np.sqrt(20/252) * iv['GBP/USD']
# data['implied_vol'] = np.sqrt(20/252) * iv['EUR/GBP']
df = df.dropna()


test_size = 365

# Train model 1 (which only takes implied standard deviation as input)
# to predict 20-day ahead volatility
X = df[['implied_vol']].iloc[:-test_size].to_numpy()
X = sm.add_constant(X)
model1 = sm.OLS(df['target_vol'].iloc[:-test_size],X, hasconst = True)
results_model1 = model1.fit()
print(results_model1.summary())

# Use Model 1 to predict future volatility
X_test = df[['implied_vol']].tail(test_size).to_numpy()
X_test= sm.add_constant(X_test)
pred_model1 = results_model1.predict(X_test) 



# Train model 2 (which takes implied standard deviation and the most
# recent observed 20-day standard deviation as input) to predict the
# 20-day ahead volatility
X = df[['implied_vol', 'current_vol']].iloc[:-test_size].to_numpy()
X = sm.add_constant(X)
model2 = sm.OLS(df['target_vol'].iloc[:-test_size],X, hasconst = True)
results_model2 = model2.fit()
print(results_model2.summary())

X_test = df[['implied_vol', 'current_vol']].tail(test_size).to_numpy()
X_test= sm.add_constant(X_test)
pred_model2 = results_model2.predict(X_test) 


# Plot the forecasts against the observed 20-day standard deviation
fig,ax = plt.subplots(figsize=(10,4))
plt.plot(df['target_vol'].tail(test_size).to_numpy(), color = 'green', alpha = 0.5, label='Realised Volatility')
plt.plot(pred_model1, color = 'black', alpha = 0.75, label='IV-OLS 1')
plt.plot(pred_model2, color = 'red', alpha = 0.75, label='IV-OLS 2')
plt.legend(loc="upper right")
plt.title('Implied Volatility-based OLS Regression GBP/USD Forecast')
plt.show()


# Calculate RMSE and MAE of forecasts
model1_rmse = math.sqrt(np.square(np.subtract(pred_model1,df['target_vol'].tail(test_size))).mean())
model2_rmse = math.sqrt(np.square(np.subtract(pred_model2,df['target_vol'].tail(test_size))).mean())

model1_mae = mean_absolute_error(pred_model1,df['target_vol'].tail(test_size))
model2_mae = mean_absolute_error(pred_model2,df['target_vol'].tail(test_size))

print("Root Mean Square Error:")
print(model1_rmse)
print()


print("Mean Absolute Error:\n")
print(model1_mae)
print()


print("Root Mean Square Error:")
print(model2_rmse)
print()

print("Mean Absolute Error:")
print(model2_mae)
print()

