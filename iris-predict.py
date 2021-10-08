import os
import sys 
import json
import requests
import tensorflow as tf
import numpy as np
from time import sleep

'''

Example input 
{ 
	"use_scoring": true, 
	"scoring_args": {
		"sepalLength" : 5.1, 
		"sepalWidth" : 3.5, 
		"petalLength" : 1.4, 
		"petalWidth" : 0.2
	}
}

'{"sepalLength":5.1,"sepalWidth":3.5,"petalLength":1.4,"petalWidth":0.2 }'

'''

## Set the project repo 
def ProjectRepo(path):
    ProjectRepo = "/bd-fs-mnt/project_repo/models"
    return ProjectRepo + '/' + path

irisType = ["Setosa","Versicolor","Virginica"]
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#cli_input = json.loads('{"sepalLength":5.1,"sepalWidth":3.5,"petalLength":1.4,"petalWidth":0.2 }')
cli_input = json.loads(sys.argv[1])

input_data = [] 
col_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth"]
for i in col_names:
	input_data.append(cli_input[i])

model = tf.keras.models.load_model(ProjectRepo('/iris/irisModel.h5'), compile = False)
predictions = model.predict([input_data])
iris_idx = np.argmax(predictions)
print("Iris Species: ",irisType[iris_idx])
print("Estimate: ", round(predictions[0][0],2))
