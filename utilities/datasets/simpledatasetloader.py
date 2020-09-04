import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors = None):
		"""
		arguments:
		preprocessors -- a python list contain the preprocess methods
		"""

		self.preprocessors = preprocessors

		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose = -1):
		"""
		load dataset from disk

		arguments:
		imagePaths -- a list of the file paths to the images in our dataset residing on disk
		verbose -- a flag to specify to print updates to a console or not
		"""

		# initialize the list of features and labels
		data = []
		labels = []

		for (i, imagePath) in enumerate(imagePaths):
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2] # assumming that our path has the following format: 
													# /dataset_name/class/image.jpg
			if self.preprocessors is not None:
				for p in self.preprocessors:
					image = p.preprocess(image)

			data.append(image)
			labels.append(label)

			if verbose > 0 and (i+1)%verbose == 0:
				print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))

		return (np.array(data), np.array(labels))

