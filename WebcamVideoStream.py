# import the necessary packages
import numpy as np
import imutils
import cv2
import sys
import os

# import matplotlib
# import matplotlib.pyplot as plt
import copy
import tensorflow as tf
# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# from threading import Thread

class SingleMotionDetector:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
	def read(self):
		# return the frame most recently read
		return self.frame
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
