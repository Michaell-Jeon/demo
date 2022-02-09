#!/usr/bin/python3

import tensorflow as tf
from datetime import datetime
import time
import os
import sys
import json
import logging


#Fill in your student initials here
STUDENT = ""

#Disable TensorFlow warnings.  
tf.get_logger().setLevel(logging.ERROR)

mnist = tf.keras.datasets.mnist
from tensorflow.python.client import device_lib

# Project repo path function - file system mount available to all app containers
def ProjectRepo(path):
    ProjectRepo = "/bd-fs-mnt/project_repo"
    return str(ProjectRepo + '/' + path)

# Location of TensorFlow model in h5 format
IMAGE_MODEL_LOC = ProjectRepo("/models/" + STUDENT + "_MNIST/mnist_digits.h5")

# Locations of Image (mnist)  dataset   
MNIST_LOC = ProjectRepo("/data/" + STUDENT + "_MNIST/mnist.npz")

# load json file to see if any additional attributes were passed
json_attributes = json.loads(sys.argv[1])

if 'imageId' in json_attributes:
# load image from test_images
    (train_images, train_labels),(test_images, test_labels) = mnist.load_data(MNIST_LOC)
    test_images = test_images / 255.0

# load tensorflow model.     
    loaded_model = tf.keras.models.load_model(IMAGE_MODEL_LOC)

# If ExecCount attribute exists execute the specified number of times 
# before returning

    if 'ExecCount' in json_attributes:
        exec_count = json_attributes["ExecCount"]
    else:
        exec_count = 1

    count = 0
    TotalPredTime = 0

# Lookup imageId passed in and evaluate
    while (count < exec_count):
        PredStartTime = time.time()
        json_trans = json.loads(sys.argv[1])
        imageid = json_trans["imageId"]
        image = test_images[imageid] 
        image = image.reshape(1,28,28)
        prediction =loaded_model.predict_classes(image)

        PredEndTime = time.time()
        TotalPredTime = TotalPredTime + (PredEndTime - PredStartTime)
        count = count + 1

    print("The Number is a:", prediction[0])
    print ("Avg Predict Time: {:.4f} ".format( (TotalPredTime / count )))
