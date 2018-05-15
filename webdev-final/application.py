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

import re
import sys
import base64
import pylab


import json


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


from PIL import Image
import numpy as np
import flask
from flask import Flask, send_file,render_template, request, url_for, make_response
from flask.ext.cors import CORS, cross_origin


import pyimagesearch.imutils as imutils
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = load_model('final_model.h5')
graph = tf.get_default_graph()

cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})


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


@app.route('/predict', methods=["POST"])
@cross_origin()
def simple():
    # initialize the data dictionary that will be returned from the
    # view

    # ensure an image was properly uploaded to our endpoint
    print(request, file=sys.stdout)
    try:
        data = request.form
        bit64 = data['data']
        tag = data['tag']
        image_data = re.sub('^data:image/png;base64,', '', bit64)
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        

        if im.mode == 'RGBA':
            im = im.convert('RGB')
            im = np.asarray(im)

    except KeyError:
        flask.abort(400, 'Expected a file with key "image", not found')

    result = scan_image(model,im,10,.6)

    fig=Figure()

    fig,ax = plt.subplots(1)
    ax.imshow(im)

    for i in result[1]:
        rect = patches.Rectangle(i,80,80,linewidth=4,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')

    plt.savefig('buf-save'+tag+'.png',transparent=True, format='png')
    
    canvas=FigureCanvas(fig)
    png_output =BytesIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    
    # response.headers['Content-Type'] = 'image/png'
    return response



if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(host= '0.0.0.0')

