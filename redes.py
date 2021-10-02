# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:55:20 2019

@author: aagils
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

n_cols = x_tz.shape[1]
classifier = Sequential()
classifier.add(Dense(7, activation='relu', kernel_initializer='random_normal', input_dim=n_cols,))
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
nn_ = classifier.fit(x_tz, y_train, batch_size=10, epochs=100)
y_pred = nn_.predict(x_tz.values)
roc_auc_score(x_tz, y_train)

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

classifier.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', auroc])

 y_pred = model.predict_proba(x_test)

