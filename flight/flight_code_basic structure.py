import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from datetime import datetime, timedelta
import holidays
import numpy as np
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#########################Cleaning Train data#########################################
df = pd.read_excel('Data_Train.xlsx')
# df = pd.read_excel('dummy_train.xlsx')
df.dropna(inplace=True)
# ####$$$$$$$######
df=df[df['Price']<=60000]
# ####$$$$$$$######
df = df.apply(lambda x: x.astype(str).str.upper())
df['Price'] = df['Price'].astype(int)

df.loc[df['Source']=='DELHI','Source'] = 'NEW DELHI'
df.loc[df['Destination']=='DELHI','Destination'] = 'NEW DELHI'

df['Arrival_Time'] = df['Arrival_Time'].apply(lambda x: re.sub('\s\d\d\s\w+','',x))

df['Duration'] = pd.to_timedelta(df['Duration'])
df['Duration'] =df['Duration'].dt.total_seconds().div(60).astype(int)

df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
df['Month'] = df['Date_of_Journey'].dt.month
df['Date'] = df['Date_of_Journey'].dt.day
df['Day_of_week'] = df['Date_of_Journey'].dt.dayofweek
df['Month'] = df['Month'].astype(str)
df['Day_of_week'] = df['Day_of_week'].astype(str)
# df['Dep_Time'] =  df['Dep_Time'].apply(lambda x: x.split(':')[0])
# df['Dep_Time'] = df['Dep_Time'].astype(int)
# df['Arrival_Time'] =  df['Arrival_Time'].apply(lambda x: x.split(':')[0])
# df['Arrival_Time'] = df['Arrival_Time'].astype(int)

df = df.drop(['Date_of_Journey','Route'],axis=1)

X = df[['Airline','Source','Destination','Duration','Total_Stops','Additional_Info','Month','Date','Day_of_week']]
X = pd.get_dummies(X,drop_first=True)
Y = df[['Price']]

# X = np.append(arr=X,values = np.ones((len(X),1)).astype(int),axis =1)
# def backwardElimination(x, sl):
#     numVars = len(x[0])
#     for i in range(0, numVars):
#         regressor_OLS = sm.OLS(Y, x).fit()
#         maxVar = max(regressor_OLS.pvalues).astype(float)
#         if maxVar > sl:
#             for j in range(0, numVars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                     x = np.delete(x, j, 1)
#     print(regressor_OLS.summary())
#     return x
#
# SL = 0.05
# X_Modeled = backwardElimination(X.values, SL)
################################ scaling  #################################

# sc_X = StandardScaler()
# X_Modeled = sc_X.fit_transform(X_Modeled)
# x_test = sc_X.transform(X_test)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=11)


#########################Cleaning Test data#########################################
# test_data = pd.read_excel('Data_Train.xlsx')
# test_data.dropna(inplace=True)
# test_data = test_data.apply(lambda x: x.astype(str).str.upper())
# test_data['Price'] = test_data['Price'].astype(int)
# test_data.loc[test_data['Source']=='DELHI','Source'] = 'NEW DELHI'
# test_data.loc[test_data['Destination']=='DELHI','Destination'] = 'NEW DELHI'
#
# test_data['Arrival_Time'] = test_data['Arrival_Time'].apply(lambda x: re.sub('\s\d\d\s\w+','',x))
#
# test_data['Duration'] = pd.to_timedelta(test_data['Duration'])
# test_data['Duration'] =test_data['Duration'].dt.total_seconds().div(60).astype(int)
#
# test_data['Date_of_Journey'] = pd.to_datetime(test_data['Date_of_Journey'])
# test_data['Month'] = test_data['Date_of_Journey'].dt.month
# test_data['Date'] = test_data['Date_of_Journey'].dt.day
# test_data['Day_of_week'] = test_data['Date_of_Journey'].dt.dayofweek
# test_data['Month'] = test_data['Month'].astype(str)
# test_data['Day_of_week'] = test_data['Day_of_week'].astype(str)
# test_data = test_data.drop(['Date_of_Journey','Route'],axis=1)
# X_test = test_data[['Airline','Source','Destination','Duration','Total_Stops','Additional_Info','Month','Date','Day_of_week']]
# X_test = pd.get_dummies(X_test,drop_first=True)
# Y_test = test_data[['Price']]
#
# X_test = np.append(arr=X_test,values = np.ones((len(X_test),1)).astype(int),axis =1)
# def backwardElimination(x, sl):
#     numVars = len(x[0])
#     for i in range(0, numVars):
#         regressor_OLS = sm.OLS(Y, x).fit()
#         maxVar = max(regressor_OLS.pvalues).astype(float)
#         if maxVar > sl:
#             for j in range(0, numVars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                     x = np.delete(x, j, 1)
#     print(regressor_OLS.summary())
#     return x
#
# SL = 0.05
# X_opt = X_test[:,:]
# X_Modeled_Test = backwardElimination(X_opt, SL)

################################ linear regression  #################################
# model = LinearRegression()
# q = model.fit(x_train,y_train)
# predictions = q.predict(x_test)
#
# print('MSE-> ',mean_squared_error(y_test,predictions))
# print('RMSE-> ',math.sqrt(mean_squared_error(y_test,predictions)))
#
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,predictions)
# print('r2_score-> ',r2)
#
# def rmsle(Y_test,y_pred) :
#     assert len(Y_test) == len(y_pred)
#     return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+Y_test))**2))
# w = rmsle(y_test,predictions)
# print('Accuracy---> ',(1-w.values[0])*100)

#################################### random forest ###########################################
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
# n_estimators =10000
q = model.fit(x_train,y_train.values.ravel())
predictions = q.predict(x_test)
#
# # X_grid = np.arange(min([0.0]),max([1000.0]),0.1)
# # X_grid = X_grid.reshape(len(X_grid),1)
# # plt.scatter(x_train,Y,color='red')
# # plt.plot(X_grid,predictions,color='blue')
# # plt.title('flight predictions')
# # plt.xlabel('several factors')
# # plt.y_label('price')
# # plt.show()
print('MSE-> ',mean_squared_error(y_test,predictions))
print('RMSE-> ',math.sqrt(mean_squared_error(y_test,predictions)))

from sklearn.metrics import r2_score
r2 = r2_score(y_test,predictions)
print('r2_score-> ',r2)

def rmsle(Y_test,y_pred) :
    Y_test = Y_test.values
    assert len(Y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+Y_test))**2))
w = rmsle(y_test,predictions)
print(w)
print('feature importances--->',q.feature_importances_)
print('----------------------------------------------------------------------------------------')
param_grid={'bootstrap': [True], 'n_estimators': [4000,5000,6000],
            'max_depth': [30, 40, 50, 60],'max_features': [2, 3],'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12]}

if __name__ == '__main__':
    grid_search = GridSearchCV(estimator = q, param_grid=param_grid, cv=10, n_jobs=3, verbose=2)
    grid_search = grid_search.fit(x_train,y_train.values.ravel())
    print('best_accuracy',grid_search.best_score_)
    print('best_parameters', grid_search.best_params_)
    print('----------------------------------------------------------------------------------------')
    # accuracies = cross_val_score(estimator=q,X=x_train,y=y_train.values.ravel(),cv=10,n_jobs=3,verbose=1)
    # print(accuracies.mean())
    # print(accuracies.std())

#################################### XGBoost ###########################################
# print('starting XGBoost')
# import time
# from tqdm import tqdm_gui
#
# from xgboost import XGBClassifier
# classifier = XGBClassifier(verbose=True)
# print('1')
# classifier.fit(x_train,y_train.values.ravel())
# print('2')
# predictions = classifier.predict(x_test)
# print('3')
# print('MSE-> ',mean_squared_error(y_test,predictions))
# print('RMSE-> ',math.sqrt(mean_squared_error(y_test,predictions)))
#
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,predictions)
# print('r2_score-> ',r2)
#
# def rmsle(Y_test,y_pred) :
#     assert len(Y_test) == len(y_pred)
#     return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+Y_test))**2))
# w = rmsle(y_test,predictions)
# print('Accuracy---> ',(1-w.values[0])*100)

# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator=classifier,X =x_train,Y=y_train,cv=10)
# print(accuracies.mean())
# print(accuracies.std())