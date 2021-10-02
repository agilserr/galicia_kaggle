# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:48:36 2019

@author: aagils
"""
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



data = pd.concat([
       pd.read_csv("pageviews.csv", parse_dates=["FEC_EVENT"]),
       pd.read_csv("pageviews_complemento.csv", parse_dates=["FEC_EVENT"])
])

submission = pd.read_csv("sampleSubmission.csv")

X_test = []
for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns:
    print("haciendo", c)
    temp = pd.crosstab(data.USER_ID, data[c])
    temp.columns = [c + "_" + str(v) for v in temp.columns]
    X_test.append(temp.apply(lambda x: x / x.sum(), axis=1))
X_test = pd.concat(X_test, axis=1)

data = data[data.FEC_EVENT.dt.month < 10]
X_train = []
for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns:
    print("haciendo", c)
    temp = pd.crosstab(data.USER_ID, data[c])
    temp.columns = [c + "_" + str(v) for v in temp.columns]
    X_train.append(temp.apply(lambda x: x / x.sum(), axis=1))
X_train = pd.concat(X_train, axis=1)

features = list(set(X_train.columns).intersection(set(X_test.columns)))
X_train = X_train[features]
X_test = X_test[features]

y_prev = pd.read_csv("conversiones.csv")
y_train = pd.Series(0, index=X_train.index)
idx = set(y_prev[y_prev.mes >= 10].USER_ID.unique()).intersection(
        set(X_train.index))
y_train.loc[list(idx)] = 1

#### entrenamiento y validación     
def normalize(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

x_train = X_train.values
y_tr = y_train.values

X_t, X_v, y_t, y_v = train_test_split(x_train, y_train, test_size= 0.30, random_state= 25)
X_t_st, X_v_st = normalize(X_t, X_v)

def model_auc_score(model, X_t, y_t, X_v, y_v):
    model.fit(X_t, y_t)
    y_pred_t = model.predict_proba(X_t)[:,1]
    y_pred_v = model.predict_proba(X_v)[:,1]
    auc_t = roc_auc_score(y_t, y_pred_t)
    auc_v = roc_auc_score(y_v, y_pred_v)
    return auc_t, auc_v

def report_model(label, auc_t, auc_v):
    print('------- ' + label + '  --------')
    print('Training auc:   ' + str(auc_t))
    print('Validation auc: ' + str(auc_v))

 
##### Benchmark XGboost ##########

scaler = StandardScaler()
x_tz = scaler.fit_transform(X_train)     
xgb = XGBClassifier(n_estimators = 100,
            objective='binary:logistic',
            silent = 0)

params = {
        'eta' : [0.1,0.2,0.3],
        'min_child_weight': [1, 2,3,4,5, 10],
        'gamma': [0.1,0.2,0.5, 1, 1.5, 2, 5],
        'subsample': [0.5,0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6,7]
        }
grid = RandomizedSearchCV(xgb,
                      cv = 10,
                      param_distributions= params,
                      scoring='roc_auc',
                      random_state = 123,
                      verbose = 3)
xgb_ = grid.fit(x_tz, y_train)
print("Parametros:",xgb_.best_params_)
"""
Parametros: {'subsample': 0.8, 'min_child_weight': 3, 'max_depth': 3, 'gamma': 1, 'eta': 0.2, 'colsample_bytree': 1.0}
Score (AUC): 0.8559076024205946
Accuracy: 0.9499796929012844
"""
print("Score (AUC):", xgb_.best_score_)
print("Accuracy:",xgb_.score(x_tz, y_train))


xgb = XGBClassifier(n_estimators = 100,
            objective= 'binary:logistic',
            subsample = 0.6,
            min_child_weight= 10,
            max_depth= 4, 
            gamma= 1, 
            colsample_bytree= 0.6)

auc_t, auc_v = model_auc_score(xgb, X_t, y_t, X_v, y_v)
report_model('XGB', auc_t, auc_v)  
"""
Training auc:   0.9619462288659242
Validation auc: 0.8557061436303225
"""

######
"""
eval_set  = [(train,y_train), (valid,y_valid)] para controlar overfit
clf.fit(train, y_train, eval_set=eval_set,
        eval_metric="auc", early_stopping_rounds=30)
"""
xgb = XGBClassifier(n_estimators = 100,
            objective='binary:logistic')

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6,7]
        }

grid = RandomizedSearchCV(xgb,
                      cv = 10,
                      param_distributions= params,
                      scoring='roc_auc',
                      random_state = 123,
                      verbose = 3)
xgb_ = grid.fit(X_t_st, y_t)
print("Parametros:",xgb_.best_params_)
print("Score (AUC):", xgb_.best_score_)
print("Accuracy:",xgb_.score(X_t_st, y_t))

"""
{'subsample': 0.8, 
'min_child_weight': 1, 
'max_depth': 3, 
'gamma': 5, 
'colsample_bytree': 0.6}
"""
xgb = XGBClassifier(n_estimators = 100,
            objective= 'binary:logistic',
            subsample = 0.8,
            min_child_weight= 1,
            max_depth= 3, 
            gamma= 5, 
            colsample_bytree= 0.6)

auc_t, auc_v = model_auc_score(xgb, X_t_st, y_t, X_v_st, y_v)
report_model('XGB', auc_t, auc_v)  

"""
Training auc:   0.9640428975348839
Validation auc: 0.861871066457373
"""

X_test
scaler = StandardScaler()
x_test_gal = scaler.fit_transform(X_test)
x_test_gal = scaler.transform(X_test)


xgb_probs = xgb.predict_proba(x_test_gal)[:,1]
submission['SCORE'] = xgb_probs
submission.to_csv("submission.csv",index = False)

#### feature selection por correlación ###
from feature_selector import FeatureSelector #importar con directorio feature..py 

fs = FeatureSelector(X_train, y_train)
fs.identify_collinear(correlation_threshold = 0.9) #401 features with a correlation magnitude greater than 0.90.
collinear_features = fs.ops['collinear']
cols = [col for col in X_train.columns if col not in collinear_features]
X_t_co = X_train[cols]
# data frames con el feature selection por correlacion
X_t, X_v, y_t, y_v = train_test_split(X_t_co.values, y_train, test_size= 0.30, random_state= 123)
X_t_st, X_v_st = normalize(X_t, X_v)

X_test_fs = X_test[cols]
scaler = StandardScaler()
x_test_gal = scaler.fit_transform(X_test_fs)
x_test_gal = scaler.transform(X_test_fs)


 ### xbgs ###
xgb = XGBClassifier(
            objective='binary:logistic')
params = {'n_estimators' : [25,50,75,100,150],
        'min_child_weight': [1, 3,5,8,  10],
        'gamma': [0.1,0.2,0.3,0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6,7]
        }
grid = RandomizedSearchCV(xgb,
                      cv = 10,
                      param_distributions= params,
                      scoring='roc_auc',
                      random_state = 25,
                      verbose = 3,
                      n_jobs = -1)
auc_t, auc_v = model_auc_score(grid, X_t_st, y_t, X_v_st, y_v) 
report_model('XGB', auc_t, auc_v)
grid.best_params_ 

"""
data normalizada. 1625 variables

{'subsample': 0.8,
 'n_estimators': 75,
 'min_child_weight': 1,
 'max_depth': 3,
 'gamma': 0.2,
 'colsample_bytree': 1.0}

Training auc:   0.9582863441606445
Validation auc: 0.855887512979075
"""
## SVM ###
from sklearn.svm import SVC
param_grid = {'C': [i for i in range(1,10)],
              'gamma':[0.1,0.001,0.0001,0.00001], 
              'kernel':['linear','rbf','poly','sigmoid']}
grid = RandomizedSearchCV(SVC(class_weight="balanced",
                              probability = True),
                    param_grid,
                    refit = True, 
                    scoring='roc_auc',
                    cv = 5,
                    random_state =25,
                    n_jobs = -1,
                    verbose = 3)
auc_t, auc_v = model_auc_score(grid, X_t_st, y_t, X_v_st, y_v) 
report_model('SVM', auc_t, auc_v)
grid.best_params_
"""
data normalizada. 1625 variables

Training auc:   0.9081318456466967
Validation auc: 0.800920171656258
Out[21]: {'kernel': 'sigmoid', 'gamma': 1e-05, 'C': 7}
"""

## random forest 
from sklearn.ensemble import RandomForestClassifier
bosque=RandomForestClassifier(class_weight = "balanced")
params_bosque = {'criterion':['gini','entropy'],
          'n_estimators':[5,10,15,20,30,35,40,45,50,55,60,70,80],
          'max_depth': [1,2,3,4,5,6,7,8,9,10],
          'min_samples_leaf':[8,9,10,11,12,13,14,15],
          'min_samples_split':[3,4,5,6,7],
          'oob_score' : ['True','False'],
          'bootstrap' : ['True','False']}
rgs_bosque = RandomizedSearchCV(bosque,
                      cv = 10,
                      param_distributions= params_bosque,
                      scoring='roc_auc',
                      random_state = 25,
                      n_jobs = -1)
auc_t, auc_v = model_auc_score(rgs_bosque, X_t_st, y_t, X_v_st, y_v) 
report_model('Random Forest', auc_t, auc_v)
rgs_bosque.best_params_
"""
------- Random Forest  -------- ESTE ES EL DE KAGLE
Training auc:   0.9934032859104145
Validation auc: 0.8531126762724608
Out[25]: 
{'oob_score': 'True',
 'n_estimators': 60,
 'min_samples_split': 3,
 'min_samples_leaf': 15,
 'criterion': 'entropy',
 'bootstrap': 'True'} 

 ------- Random Forest  -------- CON MAX DEPTH ya esta escrito el .csv
Training auc:   0.888027180171942
Validation auc: 0.8670530678537651
Out[33]: 
{'oob_score': 'True',
 'n_estimators': 80,
 'min_samples_split': 7,
 'min_samples_leaf': 12,
 'max_depth': 3,
 'criterion': 'entropy',
 'bootstrap': 'False'} 
"""
bosque=RandomForestClassifier(oob_score= True,
                     class_weight = "balanced",        
                     n_estimators= 60,
                     min_samples_split= 3,
                     min_samples_leaf= 15,
                     criterion= 'entropy',
                     bootstrap= True)

rf = bosque.fit(X_t_st,y_t)

rf_probs = rf.predict_proba(x_test_gal)[:,1]
submission['SCORE'] = rf_probs
submission.to_csv("submission.csv",index = False)


### logistic regression
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
auc_t, auc_v = model_auc_score(logreg_cv, X_t_st, y_t, X_v_st, y_v) 
report_model('Regresión logistica', auc_t, auc_v)



### neural nets

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def auc_score(model, X_t, y_t, X_v, y_v):
    model.fit(X_t, y_t)
    y_pred_t = model.predict(X_t)
    y_pred_v = model.predict(X_v)
    auc_t = roc_auc_score(y_t, y_pred_t)
    auc_v = roc_auc_score(y_v, y_pred_v)
    return auc_t, auc_v

n_cols = X_t_st.shape[1]
classifier = Sequential()
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal', input_dim=n_cols,))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(10, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
auc_t, auc_v = auc_score(classifier, X_t_st, y_t, X_v_st, y_v)
report_model('NN', auc_t, auc_v)  
"""
------- NN  --------
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal', input_dim=n_cols,))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(10, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

------- NN  --------
Training auc:   0.860199117797927
Validation auc: 0.7519602878669306
"""
classifier = Sequential()
classifier.add(Dense(7, activation='relu', kernel_initializer='random_normal', input_dim=n_cols,))
classifier.add(Dense(33, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
classifier.fit(X_t_st, y_t, batch_size=10, epochs=100)
auc_t, auc_v = model_auc_score(classifier, X_t_st, y_t, X_v_st, y_v)
report_model('NN', auc_t, auc_v)  




