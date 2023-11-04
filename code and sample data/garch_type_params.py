# TESTS AIC AND BIC OF DIFFERENT PARAMETER SETS FOR GARCH TYPE MODELS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import seaborn as sns

# Load data
df = pd.read_csv('./data/GBPUSD=X.csv')
# df = pd.read_csv('./data/EURGBP=X.csv')

# Calculate daily returns
df['returns'] = 100 * df['Close'].pct_change().dropna()
df.dropna(inplace=True)

# Calculate the 20-day standard deviation of daily returns for 20-days ahead
# Used to evaluate the forecasts
df['target_vol'] = df.returns.rolling(20).std().shift(-20)
# df.dropna(inplace=True)

# Get the indexes and values in the training set
test_size = 365
train_idx = df.index[:len(df) - test_size]
r_train = df.returns[train_idx]

# Set random seed so to make random processes deterministic
seed = 42
np.random.seed(seed)

# Initialise variables for comparing AIC and BIC
best_p_aic, best_q_aic, best_p_bic, best_q_bic = 0, 0, 0, 0
best_aic = 99999
best_bic = 99999
best_res_aic = None
best_res_bic = None

# Initialise matrix to store AIC and BIC values
aic_mat = np.zeros((5,5))
bic_mat = np.zeros((5,5))

# Iterate over values between 1 and 5 for p and q
for p in range(1,6):
    for q in range(1,6):
       
        # Generate and fit GARCH(1,1)
        model = arch_model(r_train, p=p, q=q, vol='GARCH') # GARCH
        # Parameters can be changed to implement different GARCH-type models
        # e.g
        # model = arch_model(r_train, p=p, o=1, q=q, vol='EGARCH') # EGARCH
        # model = arch_model(r_train, p=p, o=1, q=q, vol='GARCH') # GJR-EGARCH
        # model = arch_model(r_train, p=p, o=1, q=q, power=1.0, vol='GARCH') # TGARCH
        result_1 = model.fit(disp='off')
        
        # Store AIC and BIC scores
        aic_mat[p-1,q-1] = result_1.aic
        bic_mat[p-1,q-1] = result_1.bic

        # Compare to current best AIC score and replace if new AIC is lower
        if result_1.aic < best_aic:

            best_p_aic, best_q_aic = p, q
            best_aic = result_1.aic
            best_res_aic = result_1

        # Compare to current best BIC score and replace if new AIC is lower
        if result_1.bic < best_bic:

            best_p_bic, best_q_bic = p, q
            best_bic = result_1.bic
            best_res_bic = result_1

# Print statistics of the best performing parameter set in terms of AIC
print('***AIC***')
print(best_res_aic.summary())


# Print statistics of the best performing parameter set in terms of BIC
print('***BIC***')
print(best_res_bic.summary())


# Plot heatmap of AIC scores
heat_map = sns.heatmap(aic_mat, annot = True, fmt=".1f", xticklabels=['1', '2', '3', '4', '5'], yticklabels=['1', '2', '3', '4', '5'])
plt.title('AIC scores of GARCH(p,q)')
plt.xlabel('p')
plt.ylabel('q')
plt.show()
plt.clf()

# Plot heatmap of BIC scores
heat_map = sns.heatmap(bic_mat, annot = True, fmt=".1f", xticklabels=['1', '2', '3', '4', '5'], yticklabels=['1', '2', '3', '4', '5'])
plt.title('BIC scores of GARCH(p,q)')
plt.xlabel('p')
plt.ylabel('q')
plt.show()