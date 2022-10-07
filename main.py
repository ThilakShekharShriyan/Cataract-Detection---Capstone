from email.mime import image
from fileinput import filename
from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import os
import random, re, math

app = Flask(__name__)
IMG_SIZE = 224
SEED = 42

@app.route('/', methods=['GET'])
def page():
    return render_template('index.html'  )
@app.route('/', methods=['POST'])
def predict():
	print(request)
	imageFile = request.files['imagefile']

	image_path = "static/uploads/" + imageFile.filename
	imageFile.save(image_path)
	model = load_model("model/cataract.h5")

	image = cv2.imread(image_path,cv2.IMREAD_COLOR)
	image = cv2.resize(image , (IMG_SIZE , IMG_SIZE))

	y = model.predict((np.array(image)).reshape(-1,IMG_SIZE,IMG_SIZE,3) > 0.5).astype("int32")
	if y == 0:
		ans  = "cataract"
	else :
		ans = "Healthy"

	return render_template('index.html', filename = image_path ,  prediction = y)
    
if __name__ == '__main__':
    app.run(port=6000, debug=True)
