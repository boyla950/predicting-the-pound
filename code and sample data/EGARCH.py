from turtle import color
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

# Calculate the 20-day standard deviation of daily returns for 20-days ahead
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

# Generating 20-day Expanding Window Forecast using EGARCH(1,1,1)
forecasts = []

# Iterate over each index in the test set
for i in range(len(test_idx)):

    # Get indexs from from 0 to t
    idx = test_idx[i]
    train = df.returns[:idx]
    
    # Train EGARCH(1,1,1) model
    model = arch_model(train, vol='EGARCH', p=1, o=1, q=1, dist = 'studentst') 
    model_fit = model.fit(disp='off')

    # Predict variances over next 20 days
    variance = model_fit.forecast(horizon=20, 
                             reindex=False, method='bootstrap').variance.values
                            #  egarch forecasts cant be generated analytically for n > 1
                            # therefore the bootstrapping method is used

    # Calculate the square root of the mean of the varainces
    pred = np.sqrt(np.mean(variance))

    # Append value to list of forecasts
    forecasts.append(pred)

egarch111_preds = pd.Series(forecasts, index=test_idx)


# Generating 20-day Expanding Window Forecast using EGARCH(3,1,1)
forecasts = []

# Iterate over each index in the test set
for i in range(len(test_idx)):

    # Get indexs from from 0 to t
    idx = test_idx[i]
    train = df.returns[:idx]

    # Train EGARCH(3,1,1) model
    model = arch_model(train, vol='EGARCH', p=3, o=1, q=1, dist = 'studentst') 
    model_fit = model.fit(disp='off')

    # Predict variances over next 20 days
    variance = model_fit.forecast(horizon=20, 
                             reindex=False, method='bootstrap').variance.values
                            #  egarch forecasts cant be generated analytically for n > 1
                            # therefore the bootstrapping method is used

    # Calculate the square root of the mean of the varainces
    pred = np.sqrt(np.mean(variance))

    # Append value to list of forecasts
    forecasts.append(pred)

egarch311_preds = pd.Series(forecasts, index=test_idx)



# Plot forecasts against realised 20-day standard deviations
fig,ax = plt.subplots(figsize=(10,4))
plt.plot(y_test, label="Realised Volatility", alpha=0.5, color='green')
plt.plot(egarch111_preds, color = "black", label="EGARCH(1,1,1)",  alpha=0.75)
plt.plot(egarch311_preds, color = "blue", label="EGARCH(3,1,1)",  alpha=0.75)
plt.legend(loc="upper right")
plt.title('EGARCH Model Expanding Window Forecasts')
# plt.show()
plt.savefig('./plots/egarch_fc_ew_eurgbp.png')

# calculate RMSE and MAE of forecasts
egarch111_rmse = math.sqrt(np.square(np.subtract(egarch111_preds, y_test)).mean())

egarch111_mae = mean_absolute_error(egarch111_preds, y_test)


egarch311_rmse = math.sqrt(np.square(np.subtract(egarch311_preds, y_test)).mean())

egarch311_mae = mean_absolute_error(egarch311_preds, y_test)


print("EGARCH(1,1,1) Root Mean Square Error:")
print(egarch111_rmse)
print()

print("EGARCH(1,1,1) Mean Absolute Error:")
print(egarch111_mae)
print()



print("EGARCH(3,1,1) Root Mean Square Error:")
print(egarch311_rmse)
print()

print("EGARCH(3,1,1) Mean Absolute Error:")
print(egarch311_mae)
print()