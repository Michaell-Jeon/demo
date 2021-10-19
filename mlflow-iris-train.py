
import tensorflow as tf
import keras
from keras.models import Model, save_model, load_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import mlflow.keras
import mlflow

ex_name = "demo3"
print("Running mlflow_tracking.py as experiment : ", ex_name)
mlflow.set_tracking_uri("http://10.192.0.182:5000")
mlflow.set_experiment(ex_name)
experiment = mlflow.get_experiment_by_name(ex_name)

mlflow.keras.autolog()

load_file = "/bd-fs-mnt/project_repo/data/iris.csv"
iris_data = pd.read_csv(load_file,encoding="utf-8")
y_labels = iris_data.loc[:, "species"]
x_data = iris_data.loc[:,["sepalLength","sepalWidth","petalLength","petalWidth"]]

labels = {'Setosa': [1,0,0], 'Versicolor' : [0,1,0], 'Virginica' : [0,0,1] }
y_nums = np.array(list(map(lambda v : labels[v], y_labels)))
x_data = np.array(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_nums, train_size=0.8)

Dense = keras.layers.Dense
model = keras.models.Sequential()
model.add(Dense(10,activation='relu', input_shape=(4,)))
model.add(Dense(3,activation='softmax'))

model.compile( loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
with mlflow.start_run() as run:
   model.fit(x_train, y_train, batch_size=20, epochs=100)


score = model.evaluate(x_test, y_test, verbose=1)
print('Accuracy = ', score[1], ' loss = ',score[0])


print("=================================================================")
print("Connecting Tracking Server URI: ", mlflow.get_tracking_uri())
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Active run_id: {}".format(run.info.run_id))


