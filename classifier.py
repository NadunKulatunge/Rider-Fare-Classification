#importing packages
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#reading the train dataset
df_train=pd.read_csv("train.csv")
df_train['output_label'] = (df_train['label'] == 'correct').astype('int')

train, test = train_test_split(df_train, test_size=0.2, random_state=1)

x_train = df_train.drop(columns=["tripid","pickup_time","drop_time","pick_lat","pick_lon","drop_lat","drop_lon","label","output_label"])
x_train = x_train.iloc[:,:].values

y_train=df_train['output_label'].values



#reading the test dataset
df_test = pd.read_csv("test.csv")

x_test = df_test.drop(columns=["tripid","pickup_time","drop_time","pick_lat","pick_lon","drop_lat","drop_lon"])
x_test = x_test.iloc[:,:].values

#y_test = test['output_label'].values


#Train the dataset
model=xgb.XGBClassifier(learning_rate=0.405)
model.fit(x_train, y_train)


#Prediction output
y_test_pred = model.predict(x_test)

#Check F1 Score
#print(f1_score(y_test,a))

#Pre Output csv
tripid_test = np.asarray(df_test.iloc[:, 0].values) #Trip IDs of Test data

data = np.column_stack([tripid_test, y_test_pred])
label = ["tripid", "prediction"]
frame = pd.DataFrame(data, columns=label)

#Save CSV
file_path = "./xgb_output.csv"
with open(file_path, mode='w', newline='\n') as f:
    frame.to_csv(f, float_format='%.2f', index=False, header=True)

# Parameters used by current classifier
print('Parameters currently in use:\n')
print(model.get_params())
