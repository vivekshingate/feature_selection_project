# %load q02_best_k_features/build.py
# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k = 20):
    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1] 
    
    selector = SelectPercentile(f_regression, k).fit(X, y)
    #X_new = selector.transform(X)         # Not needed here as we need the features and their scores.
    
    #We could have directly provided the list of columns selected by using the line below but the test case fails. 
    #Column list must be in descending order of scores.
    ColumnsSelected = X.columns[selector.get_support()]
    
    
    #Hence, the below trouble.
    feature_score = selector.scores_       # This gives scores of each feature
    feature_selected = X.columns.values    # This gives the name of all original features/columns
    bool_index = selector.get_support()    # Boolean values viz. True/False against each feature of column. True indicating the feature being selected. 
    
    #Zipping all the 3 values into a list
    zipped_list = list(zip(feature_score,feature_selected,bool_index))
    
    #Putting the list in dataframe after sorting it in descending order based on scores.
    Final_df = pd.DataFrame(zipped_list).sort_values(by=0,ascending=False) 
    
    return list(Final_df.loc[Final_df.iloc[:,2]][1])


percentile_k_features(df = data,k = 20)


