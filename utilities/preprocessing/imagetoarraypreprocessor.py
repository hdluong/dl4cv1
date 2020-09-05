from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat = None):
		"""
		arguments:
		dataFormat -- a string specify image data format, can be either "channels_first" or "channels_last". 
		"""

		# store the image data format
		self.dataFormat = dataFormat

	def preprocess(self, image):
		"""
		convert image to keras's format
		"""
		return img_to_array(image, data_format=self.dataFormat)