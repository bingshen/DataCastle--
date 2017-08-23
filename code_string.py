# -*-coding:utf-8 -*-
from numpy import *
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse.csr import csr_matrix
from sklearn.linear_model import LogisticRegression
import os

def load_data():
    df_train=pd.read_csv('data\\pfm_train.csv')
    df_test=pd.read_csv('data\\pfm_test.csv')
    return df_train,df_test

def make_feat_dict(featArr):
    feat_dict={};counter=1
    for item in featArr:
        if(item not in feat_dict):
            feat_dict[item]=counter
            counter+=1
    return feat_dict

def map_featid(feat,feat_dict,featArr):
    feat_list=[]
    for item in featArr:
        feat_list.append(feat_dict[item])
    return pd.DataFrame({feat:feat_list})

if __name__ == '__main__':
    df_train,df_test=load_data()
    feats=["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","Over18","OverTime"]
    for feat in feats:
        featArr=hstack((df_train[feat].values,df_test[feat].values))
        feat_dict=make_feat_dict(featArr)
        new_train=map_featid(feat,feat_dict,df_train[feat].values)
        new_test=map_featid(feat,feat_dict,df_test[feat].values)
        del df_train[feat]
        del df_test[feat]
        df_train[feat]=new_train
        df_test[feat]=new_test
    df_train['Age']=pd.DataFrame({'Age':df_train['Age'].values//5})
    df_train['AgeDistance']=pd.DataFrame({'AgeDistance':df_train['Age'].values*100+df_train['DistanceFromHome'].values})
    df_test['AgeDistance']=pd.DataFrame({'AgeDistance':df_test['Age'].values*100+df_test['DistanceFromHome'].values})
    df_train['AgeEnvir']=pd.DataFrame({'AgeEnvir':df_train['Age'].values*10+df_train['EnvironmentSatisfaction'].values})
    df_test['AgeEnvir']=pd.DataFrame({'AgeEnvir':df_test['Age'].values*10+df_test['EnvironmentSatisfaction'].values})
    df_train['JobRoleLevel']=pd.DataFrame({'JobRoleLevel':df_train['JobRole'].values*10+df_train['JobLevel'].values})
    df_test['JobRoleLevel']=pd.DataFrame({'JobRoleLevel':df_test['JobRole'].values*10+df_test['JobLevel'].values})
    df_train['OverPer']=pd.DataFrame({'OverPer':df_train['OverTime'].values*10+df_train['PerformanceRating'].values})
    df_test['OverPer']=pd.DataFrame({'OverPer':df_test['OverTime'].values*10+df_test['PerformanceRating'].values})
    df_train['InvolvementPer']=pd.DataFrame({'InvolvementPer':df_train['JobInvolvement'].values*10+df_train['PerformanceRating'].values})
    df_test['InvolvementPer']=pd.DataFrame({'InvolvementPer':df_test['JobInvolvement'].values*10+df_test['PerformanceRating'].values})
    df_train['StockYear']=pd.DataFrame({'StockYear':df_train['StockOptionLevel'].values*10+df_train['YearsAtCompany'].values})
    df_test['StockYear']=pd.DataFrame({'StockYear':df_test['StockOptionLevel'].values*10+df_test['YearsAtCompany'].values})
    df_train.to_csv("data\\train.csv",index=False)
    df_test.to_csv("data\\test.csv",index=False)