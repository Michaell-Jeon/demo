# Salary-predict.py

import numpy as np
import pickle
import sys, json
from sklearn.linear_model import LinearRegression

save_model = "/bd-fs-mnt/project_repo/models/salary_model.pkl"
model = pickle.load(open(save_model,'rb'))

json_data = json.loads(sys.argv[1])
#json_data = json.loads('{ "exp": 10 }')
prediction = model.predict([[np.array(json_data['exp'])]])
output = prediction[0]
print("Expected salary: ", output)


# Postman
# curl -i -X POST 'https://gateway.hpe.local:10055/salary/1/predict' -H "Content-Type:application/json" -H "X-Auth-Token: b22479e9da07cc0d3b9d9e8199753c6d" -d '{ "exp": 10}'

# haproxy
# curl -i -X POST http://10.192.1.20:32700/salary/1/predict -H "Content-Type:application/json" -d '{"exp": 10}'

# REST API 
# curl -i -X POST http://10.192.1.21:10001/salary/1/predict -H "Content-Type:application/json" -d '{"exp": 10}'
