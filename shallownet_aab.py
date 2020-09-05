from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset")
args = vars(ap.parse_args())

# grab the list paths of images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(128, 128)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors = [sp, iap])
(data, labels) = sdl.load(imagePaths, verbose = "50")
data = data.astype("float")/255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.2, random_state = 42)

#convert the labels to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

#initialize the optimizer and model
print("[INFO] compling model...")
opt = SGD(lr=0.005)
model. = ShallowNet.build(width = 128, height = 128, depth = 3, classes = 3)
model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 32, epochs = 25, verbose = 1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1),
target_names=["ak47", "american-flag", "baseball-bat"]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()