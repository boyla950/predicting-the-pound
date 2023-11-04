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

# Generating 20-day Rolling Window Forecast using GJR-GARCH(1,1,1)
forecasts = []

# Iterate over each index in the test set
for i in range(len(test_idx)):

    # Get indexs from t-200 to t
    idx = test_idx[i]
    train = df.returns[(idx - 200):idx]
    # This can be adapted to generate expanding window forecast by using:
    #       train = df.returns[:idx]
    
    # Train TGARCH(1,1,1) model
    model = arch_model(train, vol='GARCH', p=1, o=1, q=1, power=1.0, dist = 'studentst') 
    model_fit = model.fit(disp='off')

    # Predict variances over next 20 days
    variance = model_fit.forecast(horizon=20, 
                             reindex=False, method='bootstrap').variance.values
                            #  tgarch forecasts cant be generated analytically for n > 1
                            # therefore the bootstrapping method is used

    # Calculate the square root of the mean of the varainces
    pred = np.sqrt(np.mean(variance))

    # Append value to list of forecasts
    forecasts.append(pred)

tgarch111_preds = pd.Series(forecasts, index=test_idx)


# Plot forecasts against realised 20-day standard deviations
fig,ax = plt.subplots(figsize=(10,4))
plt.plot(y_test, label="Realised Volatility", alpha=0.5)
plt.plot(tgarch111_preds, color = "black", label="TGARCH(1,1,1)",  alpha=0.75)
plt.legend(loc="upper right")
plt.title('TGARCH Model Rolling Window Forecasts')
plt.show()

# calculate RMSE and MAE of forecasts
tgarch111_rmse = math.sqrt(np.square(np.subtract(tgarch111_preds, y_test)).mean())

tgarch111_mae = mean_absolute_error(tgarch111_preds, y_test)

print("TGARCH(1,1,1) Root Mean Square Error:")
print(tgarch111_preds)
print()

print("TGARCH(1,1,1) Mean Absolute Error:")
print(tgarch111_preds)
print()
