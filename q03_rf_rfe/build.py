# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):

    X = data[[col for col in data.columns if col != 'SalePrice']]
    y = data['SalePrice']

    RFC_model = RandomForestClassifier(random_state=9)

    RFE_model = RFE(estimator=RFC_model,n_features_to_select=None)
    RFE_model.fit(X,y)

#     print(X.columns)
#     print(np.array(RFE_model.support_))
#     print(np.array(RFE_model.ranking_))
    return list(X.columns[RFE_model.support_])
rf_rfe(data)

