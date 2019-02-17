import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# Loading JSON into Dataframe
with open('ultimate_data_challenge.json') as f:
    data  = json.load(f)

df = pd.DataFrame(data)


# Converting Date data into datetime
df.last_trip_date = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in df.last_trip_date]
df.signup_date = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in df.signup_date]

# Generating Min Date
min_date = df.last_trip_date.max() - datetime.timedelta(days=30)

# Determining Active Users
active_user = []
for x in df.last_trip_date:
    if x >= min_date:
        active_user.append(1)
    else:
        active_user.append(0)

df['Active_Users'] = active_user

print("% of Active Users: ", len(df.Active_Users[df.Active_Users == 1])/len(df.Active_Users)*100)

# Cleaning up Features and converting into ML usable features
df.fillna(0,inplace=True)
df['signup_date'] = df['signup_date'].astype('int64')
df['last_trip_date'] = df['last_trip_date'].astype('int64')
df['ultimate_black_user'] = df['ultimate_black_user'].astype('int64')
df = df.drop('phone',axis=1)
df = df.drop('last_trip_date',axis=1)

df_dumb = pd.get_dummies(df.select_dtypes(include=['object']), dummy_na=True)
df = pd.concat([df.select_dtypes(exclude=['object']), df_dumb], axis=1)
df.info()

# Normalizing Dataframe
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df)
df.loc[:, :] = scaled_values


df_features = df.loc[:, df.columns != 'Active_Users']
y = df.Active_Users
x_train,x_test,y_train,y_test = train_test_split(df_features,y,test_size=.2)

# Machine Learning Model
scores = []
#Train Classifier: Random Forest
clf = RandomForestClassifier(n_estimators=8)

#Fit classifier to Train data set
clf.fit(x_train,y_train)

#Predict Proba

feat_importance = clf.feature_importances_

data = {'column_names': df_features.columns, 'feat_importance': feat_importance}
feat_table = pd.DataFrame(data=data).sort_values(by='feat_importance',ascending=False)

print(feat_table)
pred = clf.predict(x_test)

scores = metrics.f1_score(y_test,pred)
print("F1 score: ", scores)


