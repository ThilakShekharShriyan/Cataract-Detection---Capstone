from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import random, re, math

app = Flask(__name__)
img_size = 224
SEED = 42

def decode_image(filename, label=None, image_size=(img_size,img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3) 
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    if label is None:
        return image
    else:
        return image, label
    
def preprocess(df,test=False):
    paths = df.filename.apply(lambda x: './ODIR-5K/ODIR-5K/Training Images/' + x).values
    labels = df.loc[:, ['N', 'C', 'G','H']].values
    if test==False:
        return paths,labels
    else:
        return paths
    
def data_augment(image, label=None, seed=SEED):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
           
    if label is None:
        return image
    else:
        return image, label

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    
    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    
    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )
    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))


@app.route('/', methods=['GET'])
def page():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
	imageFile = request.files['imagefile']
  

	image_path = "static/uploads/" + imageFile.filename
	imageFile.save(image_path)
	model = load_model("model/cataract.h5")
	model.predict()
   
	return render_template('index.html')
    
	


if __name__ == '__main__':
    app.run(port=3000, debug=True)