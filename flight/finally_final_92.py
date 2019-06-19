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
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# reading train and test data into dataframes
df = pd.read_excel('Data_Train.xlsx')
test_df = pd.read_excel('Test_set.xlsx')
test_df['Price'] = 0
df.dropna(inplace=True)
df.drop_duplicates(keep='first', inplace=True)
# *************things to be done for df *******************
df['Arrival_Time'] = df['Arrival_Time'].apply(lambda x: re.sub('\s\d\d\s\w+','',x))
df['Duration'] = pd.to_timedelta(df['Duration'])
df['Duration'] = df['Duration'].dt.total_seconds().div(60).astype(int)
df.loc[df['Duration']<60,'Duration']+= 1440
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
df['Month'] = df['Date_of_Journey'].dt.month
df['Date'] = df['Date_of_Journey'].dt.day
df['Day_of_week'] = df['Date_of_Journey'].dt.dayofweek
df['Month'] = df['Month'].astype(str)
df['Day_of_week'] = df['Day_of_week'].astype(str)
df['Date'] = df['Date'].astype(str)
df['Arrival_Hour'] = df['Arrival_Time'].apply(lambda x: x.split(':')[0])
df['Arrival_Minute'] = df['Arrival_Time'].apply(lambda x: x.split(':')[1])
df['Dep_Hour'] = df['Dep_Time'].apply(lambda x: x.split(':')[0])
df['Dep_Minute'] = df['Dep_Time'].apply(lambda x: x.split(':')[1])
df = df.drop(['Date_of_Journey'],axis=1)
df = df.apply(lambda x: x.astype(str).str.upper())
df.loc[df['Source']=='DELHI','Source'] = 'NEW DELHI'
df.loc[df['Destination']=='DELHI','Destination'] = 'NEW DELHI'
df.loc[df['Airline']=='Jet Airways Business','Additional_Info']= 'Business class'
df['Price'] = df['Price'].astype(int)
df['Duration'] = df['Duration'].astype(int)
df['Route'] = df['Route'].apply(lambda x: ''.join(''.join(x.split('→')[1:-1]).split(' ')))
df.loc[df['Route']=='','Route']='XXX'
df.drop_duplicates(subset=['Airline', 'Source', 'Destination', 'Route',
                           'Duration', 'Total_Stops',
                           'Additional_Info', 'Month', 'Date', 'Day_of_week',
                           'Arrival_Hour','Arrival_Minute','Dep_Hour','Dep_Minute'],keep=False, inplace=True)
# Q1 = df['Price'].quantile(0.25)
# Q3 = df['Price'].quantile(0.75)
# IQR = Q3 - Q1
# upper_limit = Q3+3*IQR
# df['Price'] = df['Price'].clip(0, upper_limit, axis=0)
# df.to_csv(r'C:\Users\madhur_yadav\Desktop\only_mike.csv',sep='|',index=False)
# *************things to be done for test_df *******************
test_df['Date_of_Journey'] = pd.to_datetime(test_df['Date_of_Journey'])
test_df['Arrival_Time'] = test_df['Arrival_Time'].apply(lambda x: re.sub('\s\d\d\s\w+','',x))
test_df['Duration'] = pd.to_timedelta(test_df['Duration'])
test_df['Duration'] = test_df['Duration'].dt.total_seconds().div(60).astype(int)
test_df.loc[test_df['Duration']<60,'Duration']+= 1440

test_df['Date_of_Journey'] = pd.to_datetime(test_df['Date_of_Journey'])
test_df['Month'] = test_df['Date_of_Journey'].dt.month
test_df['Date'] = test_df['Date_of_Journey'].dt.day
test_df['Day_of_week'] = test_df['Date_of_Journey'].dt.dayofweek
test_df['Month'] = test_df['Month'].astype(str)
test_df['Day_of_week'] = test_df['Day_of_week'].astype(str)
test_df['Date'] = test_df['Date'].astype(str)
test_df['Arrival_Hour'] = test_df['Arrival_Time'].apply(lambda x: x.split(':')[0])
test_df['Arrival_Minute'] = test_df['Arrival_Time'].apply(lambda x: x.split(':')[1])
test_df['Dep_Hour'] = test_df['Dep_Time'].apply(lambda x: x.split(':')[0])
test_df['Dep_Minute'] = test_df['Dep_Time'].apply(lambda x: x.split(':')[1])
test_df = test_df.drop(['Date_of_Journey'],axis=1)
test_df = test_df.apply(lambda x: x.astype(str).str.upper())
test_df.loc[test_df['Source']=='DELHI','Source'] = 'NEW DELHI'
test_df.loc[test_df['Destination']=='DELHI','Destination'] = 'NEW DELHI'
test_df.loc[test_df['Airline']=='Jet Airways Business','Additional_Info']= 'Business class'
test_df['Price'] = test_df['Price'].astype(int)
test_df['Duration'] = test_df['Duration'].astype(int)
test_df['Route'] = test_df['Route'].apply(lambda x: ''.join(''.join(x.split('→')[1:-1]).split(' ')))
test_df.loc[test_df['Route']=='','Route']='XXX'
#########################################
# special route thing --> others
# airline_train = list(df['Airline'].unique())
# additional_info_train = list(df['Additional_Info'].unique())
# total_stops_train = list(df['Total_Stops'].unique())
# route_train = list(df['Route'].unique())

airline_test = list(test_df['Airline'].unique())
additional_info_test = list(test_df['Additional_Info'].unique())
total_stops_test = list(test_df['Total_Stops'].unique())
route_test = list(test_df['Route'].unique())

df = df[df['Airline'].isin(airline_test)]
df = df[df['Additional_Info'].isin(additional_info_test)]
df = df[df['Total_Stops'].isin(total_stops_test)]
df = df[df['Route'].isin(route_test)]

# ***************************************
df = pd.concat([df,test_df])
# ***************************************

# df=df[df['Price']<=65000]
categories = ['Airline','Source','Destination','Total_Stops','Additional_Info','Month','Date','Day_of_week','Route','Arrival_Hour','Arrival_Minute','Dep_Hour','Dep_Minute']

for i in categories:
    df[i] = df[i].astype(str)


df.to_csv(r'C:\Users\madhur_yadav\Desktop\mike_testing.csv',sep='|',index=False)
print(df.info())
print(test_df.info())

# ********************segregate *********************
df = pd.get_dummies(df,drop_first=True)
X_test = df[df['Price'] == 0]
X_test = X_test.drop('Price',axis=1)
X = df[df['Price'] != 0]
Y = X[['Price']]
X = X.drop('Price',axis=1)
# X,Y
# X_test
print('X,X_test,Y',X.shape,X_test.shape,len(Y))
# *********************************************
sc_X = StandardScaler()
sc_Y = StandardScaler()
x_train = sc_X.fit_transform(X)
x_test = sc_X.transform(X_test)
y_train = sc_Y.fit_transform(Y)
# *********************************************
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators =20000, n_jobs=3,verbose=2)
q = model.fit(x_train,y_train)
predictions = q.predict(x_test)
print('MSE-> ',mean_squared_error(y_train,q.predict(x_train)))
print('RMSE-> ',math.sqrt(mean_squared_error(y_train,q.predict(x_train))))
from sklearn.metrics import r2_score
r2 = r2_score(y_train,q.predict(x_train))
print('r2_score-> ',r2)
predictions = sc_Y.inverse_transform(predictions)
predictions = pd.DataFrame(predictions)
predictions = predictions.apply(lambda x: x.astype(float))
predictions[0] = predictions[0].apply(lambda x: '%.2f' %x)
print('----------------------- predictions are -------------------------------------->',len(predictions))
# temp = math.sqrt(mean_squared_log_error(y_train,q.predict(x_train)))
# print('RMSLE-> ',temp)
# print('Accuracy-> ',1-temp)
predictions.to_csv('foo.csv',index=False)
predictions.to_excel('foo.xlsx',index=False)