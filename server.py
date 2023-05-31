import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import PIL
import cv2
import sys

import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.utils as kerasImage
from keras.models import Sequential

import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from glob import glob
from random import shuffle

import datetime
from PIL import Image
import requests
from io import BytesIO
import cv2


def resultsToStep(testResultsArray):
    # Redondear si la precisión es de más de 0.99
    for i, image in enumerate(testResultsArray):
        for j, result in enumerate(image):
            if (1 - result) <= (1 - 0.99):
                testResultsArray[i][j] = 1
    return testResultsArray


def printResults(testResultsArray):
    HowMany = np.shape(testResultsArray)
    resultList = np.zeros(HowMany[0])
    resultListString = []
    for i, image in enumerate(testResultsArray):
        for j, result in enumerate(image):
            if result == 1:
                resultList[i] = int(j)

    for i, element in enumerate(resultList):
        match element:
            case 0:
                print(f"Image {i+1}: Banana")
                resultListString.append("Banana")
            case 1:
                print(f"Image {i+1}: Blue Morph Butterfly")
                resultListString.append("Blue Morph Butterfly")
            case 2:
                print(f"Image {i+1}: Boston Terrier")
                resultListString.append("Boston Terrier")
            case 3:
                print(f"Image {i+1}: Iceberg")
                resultListString.append("Iceberg")
            case 4:
                print(f"Image {i+1}: Koala")
                resultListString.append("Koala")
            case 5:
                print(f"Image {i+1}: Monarch Butterfly")
                resultListString.append("Monarch Butterfly")
            case 6:
                print(f"Image {i+1}: Mountain")
                resultListString.append("Mountain")
            case 7:
                print(f"Image {i+1}: Penguin")
                resultListString.append("Penguin")
            case 8:
                print(f"Image {i+1}: Strawberry")
                resultListString.append("Strawberry")
    return resultListString

#PATH TO IMAGES
path= "images"
value = 9

data = tf.keras.utils.image_dataset_from_directory(
    path,
    label_mode='categorical',
    image_size=(64, 64),
    color_mode='rgb',
    batch_size=50)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()[0]

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(value, activation='softmax'))

model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.01), loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
model.compile('nadam', loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])

checkpoint_path = "checkpoints/checks"
log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cp_callbacks = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

hist = model.fit(train, epochs=12, validation_data=val, callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

LowPoint = 3
HowMany = 10

testdata = batch[0][LowPoint:LowPoint + HowMany]

# PROBAR RED
testResultsArray = model.predict(testdata)
# print(testResultsArray)
testResultsArray = resultsToStep(testResultsArray)
resultArrayString = printResults(testResultsArray)

fig, ax = plt.subplots(ncols=HowMany, figsize=(35, 20))
for idx, img in enumerate(batch[0][LowPoint:LowPoint + HowMany]):
    ax[idx].imshow(img.astype(int))
    ax[idx].set_title(resultArrayString[idx], fontsize=16)
    ax[idx].set_axis_off()

plt.savefig("static/results/testSubplots.png", transparent=True)

import cv2
img = cv2.imread("static/results/testSubplots.png", cv2.IMREAD_UNCHANGED)
crop_img = img[750:1275, 385:3175]
cv2.imwrite("static/results/croppedTestSubplots.png",crop_img)




from flask import Flask, redirect, url_for, send_from_directory, request
from flask import render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.static_folder = 'static'
UPLOAD_FOLDER = "static/results/received_images/"
HTML_IMAGES_FOLDER = "./../static/results/received_images/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test", methods=["POST","GET"])
def test():
    if request.method == "POST":
        ALLOWED_EXTENSIONS = set(['.png','.jpg','.jpeg',])

        DELETE_FOLDER = 'static/results/received_images/'
        for _filename in os.listdir(DELETE_FOLDER):
            file_path = os.path.join(DELETE_FOLDER, _filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        file = request.files['testImage']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

        path2Image = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        a=1
        for i in range(10000):
            a = a + a

        print(os.listdir("static/results/received_images/"))

        data4testImage = tf.keras.utils.image_dataset_from_directory(
            "static/results/",
            label_mode='categorical',
            image_size=(64, 64),
            color_mode='rgb',
            batch_size=1)

        data4testImage_iterator = data4testImage.as_numpy_iterator()
        batch4testImage = data4testImage_iterator.next()
        testImageResults = model.predict(batch4testImage[0])
        testImageResults = resultsToStep(testImageResults)
        testResultArrayString = printResults(testImageResults)
        print(testResultArrayString)

        return render_template("test.html",testImagePath= os.path.join(HTML_IMAGES_FOLDER,filename),testResultString=testResultArrayString[0])

if __name__ == "__main__":
    app.run(debug=True,port=9989,use_reloader=False)