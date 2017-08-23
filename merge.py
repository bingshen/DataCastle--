import pandas as pd

if __name__ == '__main__':
    df_lr=pd.read_csv('LogisticRegression.csv')
    df_xg=pd.read_csv('gdbt.csv')
    pred1=df_lr['1-proba'].values
    pred2=df_xg['1-proba'].values
    pred=(pred1+pred2)/2
    resultList=[]
    for value in pred:
        if(value>0.5):
            resultList.append(1)
        else:
            resultList.append(0)
    submissionFile=pd.DataFrame({'result':resultList})
    submissionFile.to_csv("submission.csv",index=False)