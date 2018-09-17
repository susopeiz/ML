#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 13:04:08 2018

@author: susopeiz
"""


#This program makes a prediction for the Kaggle Housing Prices competition.
#The forecast is made from several house features using different models
# comparing them with the corresponding stacked ensemble model.
#
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,cross_validate,train_test_split,GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA


dirin='/Volumes/Suso/MachineLearning/Kaggle/HousePricing'
os.chdir(dirin)

dftrain0=pd.read_csv('train.csv')
dftest0=pd.read_csv('test.csv')

#Simple preprocessing
def Preprocessing():
        
    Ytrain0=dftrain0.SalePrice
    dfdata0=dftrain0.append(dftest0,ignore_index=True,sort=False).drop('SalePrice',axis=1)
    dfdata=dfdata0.copy()
    
    #DEALING WITH NANs
    #ratio of nans per feature
    rnanTrain=dftrain0.isna().sum().sort_values()/len(dftrain0)
    rnanTest=dftest0.isna().sum().sort_values()/len(dftest0)
    rnanTotal=dfdata0.isna().sum().sort_values()/len(dfdata0)
    #print('Train \n',rnanTrain.tail(35))
    #print('Test \n', rnanTest.tail(35))
    
    
    #features with nans
    fnanTrain=(rnanTrain[rnanTrain>0]).index
    fnanTest=(rnanTest[rnanTest>0]).index
    fnanTotal=(rnanTotal[rnanTotal>0]).index
    
    ##features with more than 10% of nans
    #rnan10=rnan[rnan>0.1].sort_values(ascending=False)
    #print(rnan10.index)
    
    
    
    #Adding 'No' category in categorical features with nan's when nan's mean no feature
    fnanNum=['LotFrontage','GarageYrBlt','MasVnrArea','BsmtFinSF2','BsmtFinSF1','BsmtUnfSF','BsmtUnfSF','GarageArea','TotalBsmtSF','GarageCars']
    fnanCat=[f for f in fnanTotal if f not in fnanNum]
    for f in fnanCat:
        #dfdata[f][dfdata[f].isna()]='No'
        dfdata[f].fillna('No',inplace=True)
        
    
    #Replacing nan's for median
    for f in fnanNum:
        #dfdata[f][dfdata[f].isna()]=dfdata[f].median()
        dfdata[f].fillna(value=dfdata[f].median(),inplace=True)
        
        #dfdata[f]=np.log1p(dfdata[f])
        
        
    #Bin by neighborhood depending on the corresponding median SalePrice
    df_NeighborhoodPrice=dftrain0["SalePrice"].groupby(dftrain0["Neighborhood"]).median().sort_values()
    NeighborhoodPriceCat=pd.cut(df_NeighborhoodPrice,bins=5,labels=False)
    dfdata.Neighborhood=dfdata.Neighborhood.map(NeighborhoodPriceCat)
           
    dfdata=pd.get_dummies(dfdata)
    
    dftrain=dfdata.iloc[:1460]
    dftest=dfdata.iloc[1460:]
    
    X=dftrain.values
    X_Test=dftest.values
    Y=Ytrain0.values
    
    
    return X,Y,X_Test


#Setting up each model
def SelectingModel(model='Lasso'):
    
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor

    if model=='SVM':
        Model=SVR(C=100)
        
        parameters={'C':[0.5,1,10,30,50,100],'kernel':['rbf']}
    
    if model=='KNR':
        Model=KNeighborsRegressor(n_neighbors=20,weights='distance')
        
        parameters={'n_neighbors':[3,5,10,50],'weights':['uniform','distance']}
        
    if model=='Lasso':
        Model=Lasso(alpha=0.001,max_iter=50000)
        
        parameters={'alpha':[0.0005,0.001,0.005,0.01]}
    
    if model=='RandomForest':
        Model=RandomForestRegressor(n_estimators=150,max_depth=15)
        
        parameters={'n_estimators':[150],'max_depth':[15],'max_features':[100,150,200]}
        
    if model=='GradientBoosting':
        Model=GradientBoostingRegressor(n_estimators=150, learning_rate=0.1)
        
        parameters={'n_estimators':[50,100,150],'max_depth':[3,5],'learning_rate':[0.1]}
        
    if model=='AdaBoost':
        Model=AdaBoostRegressor(n_estimators=150, learning_rate=1)
        
        parameters={'n_estimators':[50,150],'learning_rate':[0.5,1]}
        
    
    print('Selected model: {x}'.format(x=model))
    
    return Model,parameters
    

#Running GridSearch for best model parametrization
def TuningModel(X,Y,Model,parameters,GridSearchFlag=False):
    
    
    dirin='/Volumes/Suso/MachineLearning/Kaggle/HousePricing'
    os.chdir(dirin)
    
    #We are going to work with logarithmic values of price to reduce larger 
    #influence of larger prices 
    Ylog=np.log(Y)
    
    Xtrain,Xtest,Ylogtrain,Ylogtest=train_test_split(X,Ylog,test_size=0.2)
    
    scaler=MinMaxScaler()
    #scaler=StandardScaler()
    scaler.fit(Xtrain)
    Xtrain_s=scaler.transform(Xtrain)
    Xtest_s=scaler.transform(Xtest)
    
    pca=PCA(0.9999)
    pca.fit(Xtrain_s)
    Xtrain_sp=pca.transform(Xtrain_s)
    Xtest_sp=pca.transform(Xtest_s)
    
        
    
    if GridSearchFlag:
        
        model_GS=GridSearchCV(Model,parameters,return_train_score=True,scoring='neg_mean_squared_error')
        model_GS.fit(Xtrain,Ylogtrain)
        Trainscore=np.sqrt(-model_GS.cv_results_['mean_train_score'])#.reshape(len(parameters['n_estimators']),len(parameters['max_depth']))
        Testscore=np.sqrt(-model_GS.cv_results_['mean_test_score'])#.reshape(len(parameters['n_estimators']),len(parameters['max_depth']))
        
        model_GS=GridSearchCV(Model,parameters,return_train_score=True,scoring='neg_mean_squared_error')
        model_GS.fit(Xtrain_s,Ylogtrain)
        Trainscore_s=np.sqrt(-model_GS.cv_results_['mean_train_score'])#.reshape(len(parameters['n_estimators']),len(parameters['max_depth']))
        Testscore_s=np.sqrt(-model_GS.cv_results_['mean_test_score'])#.reshape(len(parameters['n_estimators']),len(parameters['max_depth']))
        
        model_GS=GridSearchCV(Model,parameters,return_train_score=True,scoring='neg_mean_squared_error')
        model_GS.fit(Xtrain_sp,Ylogtrain)
        Trainscore_sp=np.sqrt(-model_GS.cv_results_['mean_train_score'])#.reshape(len(parameters['n_estimators']),len(parameters['max_depth']))
        Testscore_sp=np.sqrt(-model_GS.cv_results_['mean_test_score'])#.reshape(len(parameters['n_estimators']),len(parameters['max_depth']))
        
        print('Grid Search CV:')
        print('Parameters: {x}'.format(x=parameters))
        print('No scale (train/test):')
        print(Trainscore), print(Testscore)
        print('Scaled (train/test):')
        print(Trainscore_s), print(Testscore_s)
        print('Scaled and PCA (train/test):')
        print(Trainscore_sp), print(Testscore_sp)
    
    
    
#Making a prediction  
def Forecast(X,Y,X_Test,Model,PrintScoresFlag=False,LogFlag=True):
    
    #print('parameters: {x}'.format(x=Model.get_params()))


    scaler=MinMaxScaler()
    #scaler=StandardScaler()
    scaler.fit(X)
    Xs=scaler.transform(X)
    X_Test_s=scaler.transform(X_Test)
    
    if LogFlag:
        Y=np.log(Y)
    
    if PrintScoresFlag:
        scores=cross_validate(Model,Xs,Y,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
        RMSEtrain=np.sqrt(-scores['train_score'])
        RMSEtest=np.sqrt(-scores['test_score'])
        print('Trainset: ',RMSEtrain.mean(),RMSEtrain.std())
        print('Testset: ',RMSEtest.mean(),RMSEtest.std())
    
    
    Model.fit(Xs,Y)
    
    Y_Forecast=Model.predict(X_Test_s)
    
    if LogFlag:
        Y_Forecast=np.exp(Y_Forecast)
    
    return Y_Forecast


#Stacking the selected models
def StackedFeatures(X,Y,X_Test,ModelList):
    
    #1 Create empty forecast arrays for train and test dataset for each model
    Ytrain_Forecast=np.empty([Y.shape[0],len(ModelList)])
    Ytest_Forecast=np.empty([X_Test.shape[0],len(ModelList)])
    
    #2 The prediction for the train dataset is done through K folds validation 
    #to avoid dependency of forecasted Y from corresponding Y values. Training
    #folds are used to fit_predict testings folds
    #3 The prediction for the test dataset is done after training the whole 
    #training dataset for each model
    kf=KFold(n_splits=5,shuffle=True)
    for im,imodel in enumerate(ModelList):
        for itrain,itest in kf.split(X):
            Xtrain_ik=X[itrain]
            Ytrain_ik=Y[itrain]
            
            Xtest_ik=X[itest]
            #Ytest_ik=Y[itest]
        
        
            Model,parameters=SelectingModel(model=imodel)
            Ytest_Forecast_i=Forecast(Xtrain_ik,Ytrain_ik,Xtest_ik,Model)
            
            Ytrain_Forecast[itest,im]=Ytest_Forecast_i
        
        Ytest_Forecast[:,im]=Forecast(X,Y,X_Test,Model)
    
    #4 Forecasted Ys from different models are used as new features to predict Y
    #    This part is done by the StackedForecast function
   
    return Ytrain_Forecast,Ytest_Forecast

#Forecast from the forecasted features
def StackedForecast(Ytrain_Forecast,Y,Ytest_Forecast,FinalModel,PrintScoresFlag=True):

    Model,parameters=SelectingModel(model=FinalModel)
    
    Y_Forecast_Stack=Forecast(Ytrain_Forecast,Y,Ytest_Forecast,Model,PrintScoresFlag)
    
    return Y_Forecast_Stack


#RUNNING ALL THE SEQUENCE

#Trying with different models
X,Y,X_Test=Preprocessing()
Model,parameters=SelectingModel(model='SVM')
TuningModel(X,Y,Model,parameters,GridSearchFlag=True)
Y_Forecast=Forecast(X,Y,X_Test,Model,PrintScoresFlag=True)
#Printing the output
df_Forecast = pd.DataFrame(Y_Forecast, index=dftest0.Id, columns=['SalePrice'])
#df_Forecast.to_csv('output_vGradientBoosting_simple.csv', header=True, index_label='Id')


#Creating a stacked ensemble from different models
ModelList=['KNR','SVM','Lasso','RandomForest','GradientBoosting','AdaBoost']
Ytrain_Forecast,Ytest_Forecast = StackedFeatures(X,Y,X_Test,ModelList)

Model,parameters=SelectingModel(model='KNR')
TuningModel(Ytrain_Forecast,Y,Model,parameters,GridSearchFlag=True)

Y_Forecast_Stack=StackedForecast(Ytrain_Forecast,Y,Ytest_Forecast,'KNR',PrintScoresFlag=True)

#Printing the output
df_Forecast_Stack = pd.DataFrame(Y_Forecast_Stack, index=dftest0.Id, columns=['SalePrice'])
df_Forecast_Stack.to_csv('output_vStack_simple.csv', header=True, index_label='Id')

