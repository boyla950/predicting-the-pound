import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import math
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv('./data/GBPUSD=X.csv')
# df = pd.read_csv('./data/EURGBP=X.csv')

# Calculate daily returns
df['returns'] = 100 * df['Close'].pct_change().dropna()
df.dropna(inplace=True)

# Calculate the 20-day standard deviation of daily returns for 20 20-days ahead
# Used to evaluate the forecasts
df['target_vol'] = df.returns.shift(-20).rolling(20).std()
df.dropna(inplace=True)

# Get the indexes and values in the test set
test_size = 365
test_idx = df.index[len(df) - test_size:]
y_test = df.target_vol[test_idx]

# Set random seed so to make random processes deterministic
seed = 42
np.random.seed(seed)

# Generating 20-day Rolling Window Forecast
forecasts = []

# Iterate over each index in the test set
for i in range(len(test_idx)):

    # Get indexs from t-200 to t
    idx = test_idx[i]
    train = df.returns[(idx - 200):idx]
    # This can be adapted to generate expanding window forecast by using:
    #       train = df.returns[:idx]
    
    # Train GARCH(1,1) model
    model = arch_model(train, vol='GARCH', p=1, q=1, dist = 'studentst') 
    model_fit = model.fit(disp='off')

    # Predict variances over next 20 days
    variance = model_fit.forecast(horizon=20, 
                             reindex=False).variance.values

    # Calculate the square root of the mean of the varainces
    pred = np.sqrt(np.mean(variance))

    # Append value to list of forecasts
    forecasts.append(pred)

garch11_preds = pd.Series(forecasts, index=test_idx)

# Reset forecasts
forecasts = []

# Repeat process for the GARCH(2,1,2) model
for i in range(len(test_idx)):

    # Get indexs from t-200 to t
    idx = test_idx[i]
    train = df.returns[(idx - 200):idx]
    # This can be adapted to generate expanding window forecast by using:
    #       train = df.returns[:idx]
    
    # Train GARCH(1,1) model
    model = arch_model(train, vol='GARCH', p=2, q=2, dist = 'studentst') 
    model_fit = model.fit(disp='off')

    # Predict variances over next 20 days
    variance = model_fit.forecast(horizon=20, 
                             reindex=False).variance.values

    # Calculate the square root of the mean of the varainces
    pred = np.sqrt(np.mean(variance))

    # Append value to list of forecasts
    forecasts.append(pred)

garch22_preds = pd.Series(forecasts, index=test_idx)


# Plot forecasts against realised 20-day standard deviations
fig,ax = plt.subplots(figsize=(10,4))
plt.plot(y_test, label="Realised Volatility", alpha=0.5)
plt.plot(garch11_preds, color = "black", label="GARCH(1,1)",  alpha=0.75)
plt.plot(garch22_preds, color = "blue", label="GARCH(2,2)",  alpha=0.75)
plt.legend(loc="upper right")
plt.title('GARCH Model Rolling Window Forecasts')
plt.show()



# calculate RMSE and MAE of forecasts
garch11_rmse = math.sqrt(np.square(np.subtract(garch11_preds, y_test)).mean())
garch22_rmse = math.sqrt(np.square(np.subtract(garch22_preds, y_test)).mean())

garch11_mae = mean_absolute_error(garch11_preds, y_test)
garch22_mae = mean_absolute_error(garch22_preds, y_test)


print("GARCH(1,1) Root Mean Square Error:")
print(garch11_rmse)
print()

print("GARCH(2,2) Root Mean Square Error:")
print(garch22_rmse)
print()

print("GARCH(1,1) Mean Absolute Error:")
print(garch11_mae)
print()

print("GARCH(2,2) Mean Absolute Error:")
print(garch22_mae)
print()
