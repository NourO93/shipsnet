# shipsnet

This project uses labeled thumbnails from Planet's Open California dataset to train a  Convolutional Neural Network (CNN). This CNN is then implemented in a sliding window for object detection over larger satellite scenes. Final model and sliding window are then implemented in a Flask app with larger Planet mosaic scenes from various dates.


## Data
Shipsnet dataset posted in Kaggle [here](https://www.kaggle.com/rhammell/ships-in-satellite-imagery). 

![Shipsnet example](blog-images/shipsnet-data.jpg)

Dataset contains:

* 2800 80x80 labeled RGB images 
* 700 ships
* 2100 no ships 

In addition, images were flipped/rotated using a Keras ImageDataGenerator to create an additional 1000 images.

## Methods

### Step 1
Non-neural net models were tested on the dataset. Classifiers tested:

- SVM (poly and rbf)
- Logistic regression
- Decision trees & random forest
- KNN

### Step 2
Best performing model (logistic regression) tested with sliding window on complete scene

### Step 4
CNN model (using Keras) trained and tested on thumbnails, as well as full image with sliding window.

Note: all CNN models were trained on amazon EC2 instance w/ GPU- they will take a very long time to run on a local machine w/ CPU. 

Final model performed with:

- Test accuracy: .97 
- Precision:  0.97
- Recall: 0.99
- Specificity: 0.92

![CNN Results](blog-images/CNN-sliding.png)

{Link here to Flask app when staged}

