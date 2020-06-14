#importing packages
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#reading the train dataset
df=pd.read_csv("train.csv")
df['output_label'] = (df['label'] == 'correct').astype('int')

train, test = train_test_split(df, test_size=0.2, random_state=1)

x_train=train.drop(columns=["tripid","pickup_time","drop_time","pick_lat","pick_lon","drop_lat","drop_lon","label","output_label"])
x_train = x_train.iloc[:,:].values

y_train=train['output_label'].values
#y_train = y_train.iloc[:,:].values

x_test=test.drop(columns=["tripid","pickup_time","drop_time","pick_lat","pick_lon","drop_lat","drop_lon","label","output_label"])
x_test = x_test.iloc[:,:].values

y_test=test['output_label'].values
#y_test = y_test.iloc[:,:].values

model=xgb.XGBClassifier(learning_rate=0.405)
model.fit(x_train, y_train)

a = model.predict(x_test)
print(f1_score(y_test,a))
