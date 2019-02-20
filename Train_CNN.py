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

    # all training images
    images_dir = 'training_images/'
    list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]####

    #features,labels = extract_features(list_images)

    #pickle.dump(features, open('features', 'wb'))
    #pickle.dump(labels, open('labels', 'wb'))

    features = pickle.load(open('features', 'rb'))
    labels = pickle.load(open('labels', 'rb'))

    # run a 10-fold CV SVM using probabilistic outputs. 
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.10)

    clf = svm.SVC(kernel='linear', C=0.1,probability=True)
    final_model = CalibratedClassifierCV(clf,cv=10,method='sigmoid')
    final_model = clf.fit(Xtrain, ytrain)

    #Nome e ordem das categorias da CNN
    cat = final_model.classes_

    ypreds = final_model.predict_proba(Xtest)
    ypred_label = final_model.predict(Xtest)

    print("Loss após treino : ", log_loss(ytest, ypreds, eps=1e-15, normalize=True))


    ######### CROSS VALIDATION ############
    CV = cross_validate(final_model,features, labels, cv=6)
    print("Test Scores após Cross validation: ", CV['test_score'])

    ######### MATRIZ CONFUSAO ################
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(ytest, ypred_label)
    np.set_printoptions(precision=2)    

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=cat, normalize=True,title='Normalized confusion matrix')

    plt.show()

    ######### ANALISE DE CLASSIFICACAO ##############
    #print(classification_report(ytest, ypred_label, target_names=cat))

    with open('Final_Model.pkl', 'wb') as fid:
        pickle.dump(final_model, fid)

# extract all features from pool layer of InceptionV3
def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []
    create_graph()
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        print(next_to_last_tensor)
        for ind, image in enumerate(list_images):
            print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor,
            {'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
            labels.append(re.split('-\d+',image.split('/')[1])[0])#####
        return features, labels

# setup tensorFlow graph initiation
def create_graph():
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')