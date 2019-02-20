import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

import os
import re

import tensorflow as tf

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pickle

import cv2

import itertools
    
def run():
    model_dir = 'model/'

    with open('Final_Model.pkl', 'rb') as fid:
        final_model = pickle.load(fid)


    ###### CÓDIGO PARA USUÁRIO ##########
    test_dir='test_images/'
    list_images = [test_dir+f for f in os.listdir(test_dir) if re.search('jpg|JPG', f)]####

    features_test = extract_features(list_images)

    y_pred = final_model.predict_proba(features_test)

    image_id = [i.split('/')[-1] for i in list_images]#####

    submit = open('submit.SVM.csv','w')
    submit.write('image;BupInferior;BupSuperior;Formicidae_Inferior;Formicidae_Superior;PentInferior;PentSuperior\n')

    for idx, name in enumerate(list_images):
        probs=['%s' % p for p in list(y_pred[idx, :])]
        submit.write('%s;%s\n' % (str(image_id[idx]),';'.join(probs)))####
        img_name = name.split('/')
        i = probs.index(max(probs))
        print("Imagem    :", img_name[1])
        print("Categoria :", final_model.classes_[i])
        print("Precisão  : %.2f" % (float(max(probs))*100), "%") 
        image = name
        img=cv2.imread(image,1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    submit.close()

    print("FIM")

def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    create_graph()
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for ind, image in enumerate(list_images):
            print('Processing >> %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor,
            {'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
        return features

# setup tensorFlow graph initiation
def create_graph():
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')