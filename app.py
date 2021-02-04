# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from WebcamVideoStream import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask, render_template, request, redirect, jsonify, url_for, flash

import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import imutils
import sys
import os

import matplotlib
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()
# --------------------------- -----------------------------------
# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
vs = SingleMotionDetector(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def predict(image_data, label_lines, sess, softmax_tensor):
		predictions = sess.run(softmax_tensor, \
				{'DecodeJpeg/contents:0': image_data})
		# Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		max_score = 0.0
		res = ''
		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			if score > max_score:
				max_score = score
				res = human_string
		return res, max_score

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# initialize the total number of framestostr
	# read thus far
	total = 0
  label_lines = [line.rstrip() for line
					in tf.io.gfile.GFile("logs/trained_labels.txt")]
	with tf.compat.v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
		graph_def = tf.compat.v1.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')
	with tf.compat.v1.Session() as sess:
		# Feed the image_data as input to the graph and get first prediction
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    c = 0
		res, score = '', 0.0
		i = 0
		mem = ''
		consecutive = 0
		sequence = ''
		# loop over frames from the video stream
		while True:
			# read the next frame from the video stream,
			frame = vs.read()
			frame = cv2.flip(frame,1)
			x1, y1, x2, y2 = 100, 100, 300, 300
			img_cropped = frame[y1:y2, x1:x2]
			image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()	#tostrings
                        # if the total number of frames has reached a sufficient
			# number to construct a reasonable background model, then
			# continue to process the frame
			if total > frameCount:
				# detect sign language in the image
				res, score = predict(image_data, label_lines, sess, softmax_tensor)
				cv2.putText(frame, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
				cv2.putText(frame, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
      total += 1
      with lock:
				outputFrame = frame.copy()
        
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')
      
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
# check to see if this is the main thread of execution
if __name__ == '__main__':
	# app.secret_key = 'super_secret_key'
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=16,
		help="# of frames used to construct the background model")
  args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
  
vs.stop()
