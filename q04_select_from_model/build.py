# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

# Your solution code here
def select_from_model(df):
    X = data[[col for col in data.columns if col != 'SalePrice']]
    y = data['SalePrice']
    
    RFC_model = RandomForestClassifier(random_state=9)
    RFC_model.fit(X,y)
    
    SFM_model = SelectFromModel(RFC_model,prefit=True)
    #X_new = SFM_model.transform(X)  #Not needed here.

    return  list(X.columns[SFM_model.get_support()])

select_from_model(data)

