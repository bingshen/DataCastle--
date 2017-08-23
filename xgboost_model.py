import pandas as pd
from datetime import datetime
import os
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import xgboost as xgb
import random
import scipy as sp
from numpy import *

def load_data():
    df_train=pd.read_csv('data\\train.csv')
    df_test=pd.read_csv('data\\test.csv')
    return df_train,df_test

if __name__ == '__main__':
    df_train,df_test=load_data()
    model=GradientBoostingClassifier()
    testX=df_test.values
    trainY=df_train['Attrition'].values
    del df_train['Attrition']
    trainX=df_train.values
    model.fit(trainX,trainY)
    predictY=model.predict_proba(testX)
    # params={
    #         'booster':'gbtree',
    #         'objective':'binary:logistic',
    #         'eta':0.1,
    #         'max_depth':8,
    #         'subsample':0.5,
    #         'colsample_bytree':0.5,
    #         'eval_metric':'error',
    #         'silent':1
    # }
    # trainY=df_train['Attrition'].values
    # del df_train['Attrition']
    # trainX=df_train.values
    # testX=df_test.values
    # plst=list(params.items())
    # print(plst)
    # num_rounds=500
    # xgtrain=xgb.DMatrix(trainX[:1000,:],label=trainY[:1000])
    # xgval=xgb.DMatrix(trainX[1000:,:],label=trainY[1000:])
    # watchlist=[(xgtrain,'train'),(xgval,'val')]
    # model=xgb.train(plst,xgtrain,num_rounds,watchlist,early_stopping_rounds=20)
    # model.save_model('xgboost.model')
    # predictX=xgb.DMatrix(testX)
    # DtrainX=xgb.DMatrix(trainX)
    # xgtrain_feat=model.predict(DtrainX)
    # xgtest_feat=model.predict(predictX)
    # trainFile=pd.DataFrame({'xgboost':xgtrain_feat})
    # trainFile.to_csv("data\\xgb_feat_train.csv",index=False)
    # testFile=pd.DataFrame({'xgboost':xgtest_feat})
    # testFile.to_csv("data\\xgb_feat_test.csv",index=False)
    # print(xgtrain_feat)
    # resultList=[]
    # for value in predictY:
    #     if(value>0.5):
    #         resultList.append(1)
    #     else:
    #         resultList.append(0)
    probaFile=pd.DataFrame({'0-proba':predictY[:,0],'1-proba':predictY[:,1]})
    probaFile.to_csv("gdbt.csv",index=False)
    # submissionFile=pd.DataFrame({'result':predictY})
    # submissionFile.to_csv("submission.csv",index=False)