import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# Read the data
train_data = pd.read_csv('../train.csv')
test_data = pd.read_csv('../test.csv')
y = train_data.Target              
train_data.drop(['Target'], axis=1, inplace=True)

numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

def get_score(n_estimators):
    my_pipeline = Pipeline(steps = [('preprocessing',SimpleImputer()),('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))])
    MAE = -1 * cross_val_score (my_pipeline,X, y, cv=3,scoring= 'neg_mean_absolute_error' )
    AVG_MAE =  MAE.mean()
    print ("Average AME of for n_estimators=",n_estimators,"in 3 CV=",AVG_MAE) 
    return AVG_MAE

results = [get_score(i) for i in range(50,400, 50)]


plt.plot(results)
plt.show()
