# import the necessary packages
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras import initializers, layers, models
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import callbacks
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import load_model

from io import StringIO
from io import BytesIO 


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from PIL import Image
import numpy as np
import flask
from flask import Flask, send_file,render_template, request, url_for

import pyimagesearch.imutils as imutils
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = load_model('final_model.h5')
graph = tf.get_default_graph()

def scan_image(model,image,stepSize,proba):
    (winW, winH) = (80, 80)
    output = []
    cnn_windows = []
    cnn_coords = []
    i = 0
    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        cnn_window = np.expand_dims(window, axis=0)
        with graph.as_default():
        	y_pred = model.predict(cnn_window)
        y_pred_val = list(y_pred[0])[1]
        
        if y_pred_val > proba :
            cnn_coords.append((x,y))
            cnn_windows.append(window)
        i+=1
    output.append(cnn_windows)
    output.append(cnn_coords)
    return output

def plot_found(cnn_coords,image):
	fig,ax = plt.subplots(1)
	ax.imshow(image)

	for i in cnn_coords:
	    rect = patches.Rectangle(i,80,80,linewidth=4,edgecolor='r',facecolor='none')
	    ax.add_patch(rect)
	
	strIO = BytesIO()
	plt.savefig(strIO, dpi=fig.dpi)
	strIO.seek(0)
	return strIO

@app.route('/predict', methods=["GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": []}

    # ensure an image was properly uploaded to our endpoint
    # path = flask.request.files["image"]
    path = 'test.png'
    image = mpimg.imread(path)
    result = scan_image(model,image,12,.7)
    plot_found(result[1],image)
    data['success'].append(1)
    return data
    

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()