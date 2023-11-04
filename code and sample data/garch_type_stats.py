# PRINTS STATISTICAL PROPERTIES OF EACH OF THE GARCH TYPE MODELS

import numpy as np
import pandas as pd
from arch import arch_model

# df = pd.read_csv('./data/GBPUSD=X.csv')
df = pd.read_csv('./data/EURGBP=X.csv')

# Calculate daily returns
df['returns'] = 100 * df['Close'].pct_change().dropna()
df.dropna(inplace=True)

# Get the indexes and values in the test set
test_size = 365
train_idx = df.index[:len(df) - test_size]
r_train = df.returns[train_idx]

seed = 42
from arch import arch_model
np.random.seed(seed)

garch11 = arch_model(r_train, p=1, q=1, vol='GARCH', dist='normal')
result_garch11 = garch11.fit(disp='off')
print(result_garch11.summary())
print()
            
garch22 = arch_model(r_train, p=2, q=2, vol='GARCH', dist='normal')
result_garch22 = garch22.fit(disp='off')
print(result_garch22.summary())
print()
            
egarch111 = arch_model(r_train, p=1, o=1, q=1, vol='EGARCH', dist='normal')
result_egarch11 = egarch111.fit(disp='off')
print(result_egarch11.summary())
print()

egarch311 = arch_model(r_train, p=3, o=1, q=1, vol='EGARCH', dist='normal')
result_egarch11 = egarch311.fit(disp='off')
print(result_egarch11.summary())
print()     

gjr111 = arch_model(r_train, p=1, o=1, q=1, vol='GARCH', dist='normal')
result_gjr111 = gjr111.fit(disp='off')
print(result_gjr111.summary())
print()
            
gjr212 = arch_model(r_train, p=2, o=1, q=2, vol='GARCH', dist='normal')
result_gjr212 = gjr212.fit(disp='off')
print(result_gjr212.summary())
print()
            
tgarch111 = arch_model(r_train, p=1, o=1,q=1 ,power=1.0, vol='GARCH', dist='normal')
result_tgarch111 = tgarch111.fit(disp='off')
print(result_tgarch111.summary())
print()
            