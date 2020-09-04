import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter = cv2.INTER_AREA):
		"""
		arguments:
		width -- width of the image, integer
		height -- height of the image, integer
		inter -- the interpolation method used when resizing, the flag of cv2
		"""

		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		"""
		resize the image to a fixed size, ignore the ratio aspect

		arguments:
		image -- the source image load by cv2.imread, numpy array
		"""

		return cv2.resize(image, (self.width, self.height), interpolation = self.inter)