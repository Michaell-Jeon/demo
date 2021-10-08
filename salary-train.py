## salary-train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json

load_file  = "/bd-fs-mnt/project_repo/data/Salary_Data.csv"
save_model = "/bd-fs-mnt/project_repo/models/salary_model.pkl"
dataset = pd.read_csv(load_file)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 777)

salary_model = LinearRegression()
salary_model.fit(X_train, y_train)
#X_test = X_test.reshape(-1)
print(X_test)

for x in X_test:
    y_pred = salary_model.predict(x.reshape(1,-1))
    print("Input(ex): ", x, "Predict(salary): ", y_pred)

pickle.dump(salary_model, open(save_model,'wb'))
print("model saved at: ", save_model)