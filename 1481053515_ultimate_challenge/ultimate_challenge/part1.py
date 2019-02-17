import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime

with open('ultimate_data_challenge.json') as f:
    data  = json.load(f)

log_json = pd.read_json('logins.json')
df = pd.DataFrame(data)

# print(df.describe())

log_json['Interval'] = ((log_json['login_time'] - pd.TimedeltaIndex(log_json['login_time'].dt.minute % 15, 'm')) -
                        pd.TimedeltaIndex(log_json['login_time'].dt.second, 's')).dt.time

log_json['Month'] = [i.month for i in log_json.login_time]
log_json['weekday'] = log_json['login_time'].dt.dayofweek
login_intervals = log_json.groupby(['Interval','weekday']).count()
print(log_json)

fig, ax = plt.subplots(figsize=(15,7))
log_json.groupby(['Interval','weekday']).count()['login_time'].unstack().plot(ax=ax)
ax.legend(loc=2, prop={'size': 20})
plt.xlabel('Time Intervals on 15 Minutes')
plt.ylabel('Count of Timestamps')
plt.title('Distribution of Time Stamps')
plt.show()
#
for i in range(0,6):
    plt.subplot(3, 3, i+1)
    plt.hist(datestr(log_json[log_json['weekday']==i]['Interval']))
plt.show()
