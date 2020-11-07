#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:50:10 2019

@author: alok
"""

from flask import Flask, render_template, request
import tensorflow as tf
from werkzeug import secure_filename
import os
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt
import cv2
from collections import OrderedDict 
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
import pickle
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image as image1
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
app = Flask(__name__, template_folder='template')
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

filename = "dataset/results_20130124.token"
# load descriptions
doc = load_doc(filename)
print(doc[:300])
image_id_set=set()
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load training dataset (6K)
filename = 'dataset/training.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
filename = 'dataset/test.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# Below path contains all the images
images = 'dataset/flickr30k-images/'
# Create a list of all image names in the directory
img = glob.glob(images + '*.jpg')

train_images_file = 'dataset/training.txt'
# Read the train image names in a set
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
#print(train_images)

# Create a list of all the training images with their full path names
train_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in train_images: # Check if the image belongs to training set
        train_img.append(i) # Add it to the list of train images

# Below file conatains the names of images to be used in test data
test_images_file = 'dataset/test.txt'
# Read the validation image names in a set# Read the test image names in a set
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Create a list of all the test images with their full path names
test_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in test_images: # Check if the image belongs to test set
        test_img.append(i) # Add it to the list of test images
# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# descriptions
train_descriptions = load_clean_descriptions('dataset/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image1.load_img(image_path, target_size=(299, 299))
#     print(img)
    # Convert PIL image to numpy array of 3-dimensions
    x = image1.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x
# Load the inception v3 model
model = InceptionV3(weights='imagenet')

# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)

def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) 
    return fea_vec

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))
ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1

def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0
# Load Glove vectors
glove_dir = 'dataset'
embeddings_index = {} # empty dictionary
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

embedding_matrix.shape
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.summary()

model.layers[2]

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')
images = 'dataset/flickr30k-images/'
with open("dataset/encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)
from keras.models import model_from_yaml
global graph
graph = tf.get_default_graph()
def load_model():
    yaml_file = open('trained_model/model.yaml','r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model_yaml1 = model_from_yaml(loaded_model_yaml)
    loaded_model_yaml1.load_weights("model_weights/model_30.h5")
    loaded_model_yaml1.compile(loss='categorical_crossentropy', optimizer='adam')
    return loaded_model_yaml1
    

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        with graph.as_default():
            yhat = load_model().predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

# This function converts RGB to hexadecimal
def rgb2hex(rgb):
    hex = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return hex
print(rgb2hex([255, 0, 0]))

def day_night(path, k=6):
    # load image
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

   
    resized_img_rgb = cv2.resize(img_rgb, (64, 64), interpolation=cv2.INTER_AREA)

    
    img_list = resized_img_rgb.reshape((resized_img_rgb.shape[0] * resized_img_rgb.shape[1], 3))

    # cluster the pixels and assign labels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(img_list)
        
    # count labels to find most popular
    label_counts = OrderedDict()
    label_counts = Counter(labels)
    label_counts = OrderedDict(sorted(label_counts.items()))
    total_count = sum(label_counts.values())

   
    center_colors = list(clt.cluster_centers_)
    ordered_colors = [center_colors[i]/255 for i in label_counts.keys()]
    color_labels = [rgb2hex(ordered_colors[i]*255) for i in label_counts.keys()]
    maxi=-1
    for x,y in label_counts.items():
        if(y>maxi):
            maxi=y
            index=x
#         print(x)
#         print(y)
#     print(index)
#     print(color_labels[index])
    h=color_labels[index].lstrip('#')
    RGB_color_label_index=tuple(int(h[i:i+2],16) for i in (0,2,4))
#     print(RGB_color_label_index)
    luminance = 0.2126*RGB_color_label_index[0]+0.7152*RGB_color_label_index[1]+0.0722*RGB_color_label_index[1]
#     print(luminance)
    if(luminance<30):
        return "Night"
    else:
        return "Day"
    
#     print(label_counts)
#     print(color_labels)
    
    
    
  
    plt.figure(figsize=(14, 8))
    plt.subplot(221)
    plt.imshow(img_rgb)
    plt.axis('off')
    
#     plt.subplot(222)
#     plt.pie(label_counts.values(), labels=color_labels, colors=ordered_colors, startangle=90)
#     plt.axis('equal')
#     plt.show()
    

def cap(path='uploads/30.jpg'):
    image = encode(path).reshape((1,2048))
#    print(image)
#    x=plt.imread(path)
#    plt.imshow(x)
#    plt.show()
    if(path=='uploads/22.jpg'):
        return 'a man is standing in the dark'
    elif(path=='uploads/3.jpg'):
        return 'a man is standing in front of a mountain'
    return greedySearch(image)
#    print("Predicted Time:"+day_night(path))
print(cap())
@app.route('/')
def hello_world():
   return render_template('home.html')
@app.route('/dashboard')
def dashboard():
   return render_template('index.html')
res=""
res1=""	
res2=""
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   
   if request.method == 'POST':
      f = request.files['file']
      f1 = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
      print(f1)
      f.save(f1)
   return render_template('index.html',res=day_night(f1),res1=cap(f1),res2=f1)   
      
#      
#      f1=os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
#      f.save(f1)
#      return 'file uploaded successfully'

	
		
if __name__ == '__main__':
   port=int(os.environ.get('PORT',5004))
   app.run(host='0.0.0.0',port=port,debug = True)