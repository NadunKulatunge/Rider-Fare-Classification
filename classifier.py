#importing packages
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pyproj import Geod
from catboost import CatBoostClassifier

wgs84_geod = Geod(ellps='WGS84')

def Distance(lat1,lon1,lat2,lon2):
    az12,az21,dist = wgs84_geod.inv(lon1,lat1,lon2,lat2)
    return dist

def AddDuration(df):
    pickup_time = df["pickup_time"].astype('datetime64')
    drop_time = df["drop_time"].astype('datetime64')
    trip_duration = drop_time-pickup_time
    trip_duration = trip_duration.dt.total_seconds()
    return trip_duration

def AddDurationWaitingFare(df):
    duration_waiting_fare = (df["duration"] - df["meter_waiting"])/df["fare"]
    return duration_waiting_fare

def AddFarePerDistance(df):
    fare_per_distance = (df["fare"])/df["distance"]
    return fare_per_distance

def AddFarePerDuration(df):
    fare_per_duration= (df["fare"])/df["duration"]
    return fare_per_duration

def AddSpeed(df):
    speed = df["distance"]/(df["duration"] - df["meter_waiting"])
    return speed

def AddAdditionalFarePerDistance(df):
    additional_fare_per_distance = (df["additional_fare"])/df["distance"]
    return additional_fare_per_distance

def AddAdditionalFareAsFactor(df):
    additional_fare_as_factor = df["additional_fare"] / df["fare"]
    return additional_fare_as_factor


#reading the train dataset
df_train = pd.read_csv("train.csv")

#Add output label as int
df_train['output_label'] = (df_train['label'] == 'correct').astype('int')

#Add Duration to train data
df_train['trip_duration'] = AddDuration(df_train)

#AddDistance to train data
df_train['distance'] = Distance(df_train['pick_lat'].tolist(), df_train['pick_lon'].tolist(), df_train['drop_lat'].tolist(), df_train['drop_lon'].tolist())

#Add Duration Waiting Fare to train data
df_train['duration_waiting_fare'] = AddDurationWaitingFare(df_train)

#Add Fare Per Distance to train data
df_train['fare_per_distance'] = AddFarePerDistance(df_train)

#Add Fare Per Duration to train data
df_train['fare_per_duration'] = AddFarePerDuration(df_train)

#Add Speed train data
df_train['speed'] = AddSpeed(df_train)

#Add Additional Fare Per Distance to train data
df_train['additional_fare_per_distance'] = AddAdditionalFarePerDistance(df_train)

#Removing unnsessary columns
x_train = df_train.drop(columns=["tripid","pickup_time","drop_time","pick_lat","pick_lon","drop_lat","drop_lon","label","output_label"])

#Getting Values as Array
x_train = x_train.iloc[:,:].values
y_train = df_train['output_label'].values




#reading the test dataset
df_test = pd.read_csv("test.csv")

#Add Duration to test data
df_test['trip_duration'] = AddDuration(df_test)

#AddDistance to test data
df_test['distance'] = Distance(df_test['pick_lat'].tolist(), df_test['pick_lon'].tolist(), df_test['drop_lat'].tolist(), df_test['drop_lon'].tolist())

#Add Duration Waiting Fare to test data
df_test['duration_waiting_fare'] = AddDurationWaitingFare(df_test)

#Add Fare Per Distance to test data
df_test['fare_per_distance'] = AddFarePerDistance(df_test)

#Add Fare Per Duration to test data
df_test['fare_per_duration'] = AddFarePerDuration(df_test)

#Add Speed test data
df_test['speed'] = AddSpeed(df_test)

#Add Additional Fare Per Distance to test data
df_test['additional_fare_per_distance'] = AddAdditionalFarePerDistance(df_test)

#Add Additional Fare As Factor to test data
df_test['additional_fare_as_factor'] = AddAdditionalFareAsFactor(df_test)

#Removing unnsessary columns
x_test = df_test.drop(columns=["tripid","pickup_time","drop_time","pick_lat","pick_lon","drop_lat","drop_lon"])

#Getting Values as Array
x_test = x_test.iloc[:,:].values




#Train the dataset
model = CatBoostClassifier()
model.fit(x_train, y_train)


#Prediction output
y_test_pred = model.predict(x_test)



#Pre Output csv
tripid_test = np.asarray(df_test.iloc[:, 0].values) #Trip IDs of Test data

data = np.column_stack([tripid_test, y_test_pred])
label = ["tripid", "prediction"]
frame = pd.DataFrame(data, columns=label)

#Save CSV
file_path = "./output_catboost.csv"
with open(file_path, mode='w', newline='\n') as f:
    frame.to_csv(f, float_format='%.2f', index=False, header=True)



