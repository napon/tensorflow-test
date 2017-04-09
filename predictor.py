import glob
import os
import sys
import librosa
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2


'''
Credits: 
- https://serv.cusp.nyu.edu/projects/urbansounddataset/
- https://github.com/jaron/deep-listening.git
- https://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/
'''

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features_array(filename, bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    sound_clip,s = librosa.load(filename)        
    for (start,end) in windows(sound_clip,window_size):
        if(len(sound_clip[start:end]) == window_size):
            signal = sound_clip[start:end]
            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            logspec = librosa.logamplitude(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features)

def build_model():
    # input: 60x41 data frames with 2 channels => (60,41,2) tensors
    frames = 41
    bands = 60
    num_channels = 2
    num_labels = 10
    model = Sequential()

    # filters of size 3x3 - paper describes using 5x5, but their input data is 128x128
    f_size = 3

    # Layer 1 - 24 filters with a receptive field of (f,f), i.e. W has the
    # shape (24,1,f,f).  This is followed by (4,2) max-pooling over the last
    # two dimensions and a ReLU activation function
    model.add(Convolution2D(24, f_size, f_size, border_mode='same', init="normal", input_shape=(bands, frames, num_channels)))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Layer 2 - 48 filters with a receptive field of (f,f), i.e. W has the 
    # shape (48, 24, f, f). Like L1 this is followed by (4,2) max-pooling 
    # and a ReLU activation function.
    model.add(Convolution2D(48, f_size, f_size, init="normal", border_mode='same'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Layer 3 - 48 filters with a receptive field of (f,f), i.e. W has the
    # shape (48, 48, f, f). This is followed by a ReLU but no pooling.
    model.add(Convolution2D(48, f_size, f_size, border_mode='valid'))
    model.add(Activation('relu'))

    # flatten output into a single dimension, let Keras do shape inference
    model.add(Flatten())

    # Layer 4 - a fully connected NN layer of 64 hidden units, L2 penalty of 0.001
    model.add(Dense(64, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Layer 5 - an output layer with one output unit per class, with L2 penalty, 
    # followed by a softmax activation function
    model.add(Dense(num_labels, W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))

    # compile and fit model, reduce epochs if you want a result faster
    # the validation set is used to identify parameter settings (epoch) that achieves 
    # the highest classification accuracy
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="adamax")
    
    return model

#####################
sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling", "gun shot","jackhammer","siren","street music"]

def setup_model(pretrained_model):
	model = build_model()
	model.load_weights(pretrained_model)
	return model, tf.get_default_graph()

def predict_unwrapped(graph, model, filename):
	features = extract_features_array(filename)
	# https://www.tensorflow.org/versions/r0.11/api_docs/python/framework/utility_functions#get_default_graph
	with graph.as_default():
		predictions = model.predict(features)

	if len(predictions) == 0:
	    print "No prediction"
	else:
	    ind = np.argpartition(predictions[0], -2)[-2:]
	    ind[np.argsort(predictions[0][ind])]
	    ind = ind[::-1]
	    print "PREDICTION: ", sound_names[ind[0]]

def predict(args):
	predict_unwrapped(*args)
#####################
