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

from PIL import Image
import numpy as np
import flask
import io

import pyimagesearch.imutils as imutils
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = load_model('final_model.h5')
graph = tf.get_default_graph()

    

def predict_image(path):
    image = Image.open(path)
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)
    with graph.as_default():
    	response = model.predict(image_array)
    print(response)
    return int(response[0][1])

@app.route('/predict', methods=["POST"])
def predict():

    data = {"success": []}

    image = flask.request.files["image"]
    result = predict_image(image)
    data['success'].append(result)



    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()