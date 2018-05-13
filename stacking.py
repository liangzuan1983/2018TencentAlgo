# -*- coding: utf-8 -*-
import os
import sys
import json
import pandas as pd
import numpy as np
from scipy import sparse
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import preprocess
import FeatFuns
import warnings
import gc

warnings.filterwarnings('ignore')

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print('Usage: python stacking.py all/sample [on/off(whether to preprocess)]')
    exit()

datasetflag = sys.argv[1].lower()
if len(sys.argv) == 2:  
    preprocessflag = 'on'
else:
    preprocessflag = sys.argv[2].lower()

# Here to set parameter
LGBM_params=dict(boosting_type='gbdt', num_leaves=355, reg_alpha=0.0, reg_lambda=1, max_depth=-1, n_estimators=100, 
        objective='binary', subsample=0.7, colsample_bytree=0.7, subsample_freq=1,learning_rate=0.02, min_child_weight=50, 
        random_state=2018, n_jobs=-1, silent=True)

XGB_params=dict(booster='gbtree',  reg_lambda=50, max_depth=7,  n_estimators=100, missing=0,  
        objective='binary:logistic', subsample=1, colsample_bytree=1, learning_rate=0.25, gamma=0, scale_pos_weight=1,
        random_state=2018, n_jobs=-1, silent=True)


def LGBTrain(parameter,train_x,train_y):
    print('LGBM...')
    clf = lgb.LGBMClassifier().set_params(**parameter)
    clf.fit(train_x, train_y)
    return clf


def XGBTrain(parameter,train_x,train_y):
    print('XGB...')
    bst = xgb.XGBClassifier().set_params(**parameter)
    bst.fit(train_x, train_y)
    return bst


def cutData(inputpath):
    print('Load dataset...')
    data_x = sparse.load_npz(inputpath+'train_x.npz')
    data_y = pd.read_csv(inputpath+'train_y.csv', header=None)
    print(data_x.shape[0])

    data_x, data_y = shuffle(data_x, data_y, random_state=2333)

    print('Slice dataset into 5 parts...')
    step = data_x.shape[0]//5
    slice_x = list(data_x[i*step:(i+1)*step] for i in range(4))
    slice_x.append(data_x[4*step:])
    slice_y = list(data_y[i*step:(i+1)*step] for i in range(4))
    slice_y.append(data_y[4*step:])
    print('Done!')

    return slice_x, slice_y


def model_train(train_x, train_y, test_x):
    
    clf = LGBTrain(LGBM_params,train_x,train_y)
    bst = XGBTrain(XGB_params,train_x,train_y)
    
    LGBM_score = clf.predict_proba(test_x)[:,1]
    XGB_score = bst.predict_proba(test_x)[:,1]

    return LGBM_score, XGB_score


def stacking_train(inputpath, outputpath):
	''' Slice dataset into 5 folds '''
	print('Slice dataset into 5 folds...')
	slice_x, slice_y = cutData(inputpath)

	scorels_LGBM = []
	scorels_XGB = []

	''' Train stacking '''
	for num in range(5):
		print('Begin stacking {0}...'.format(num))
		train_x = sparse.vstack([slice_x[i] for i in range(5) if i != num])
		train_y = pd.concat([slice_y[i] for i in range(5) if i != num], ignore_index=True)
		score_LGBM, score_XGB = model_train(train_x, train_y, slice_x[num])
		scorels_LGBM+=score_LGBM.tolist()
		scorels_XGB+=score_XGB.tolist()
		del train_x, train_y, score_LGBM, score_XGB
		print(len(scorels_LGBM))
		gc.collect()

	print('Saving stacking...')
	score_feat = pd.DataFrame({'LGBM':scorels_LGBM, 'XGB':scorels_XGB})
	score_feat.to_csv(outputpath+'stacking_train.csv',index=False)


def stacking_test(inputpath, outputpath):
    print('Load dataset...')
    data_x = sparse.load_npz(inputpath+'train_x.npz').tocsr()
    data_y = pd.read_csv(inputpath+'train_y.csv', header=None)
    test_x = sparse.load_npz(inputpath+'test_x.npz').tocsr()

    score_LGBM, score_XGB = model_train(data_x, data_y, test_x)

    print('Saving stacking...')
    score_feat = pd.DataFrame({'LGBM':score_LGBM, 'XGB':score_XGB})
    score_feat.to_csv(outputpath+'stacking_test.csv',index=False)


def LGBMTrain_stacking(inputpath, outputpath):
    print('Load dataset...')
    data_x = sparse.load_npz(inputpath+'train_x.npz')
    data_y = pd.read_csv(inputpath+'train_y.csv', header=None)
    test_x = sparse.load_npz(inputpath+'test_x.npz')
    data_stacking_feat = pd.read_csv(inputpath+'stacking_train.csv').drop('LGBM', axis=1)
    test_stacking_feat = pd.read_csv(inputpath+'stacking_test.csv').drop('LGBM', axis=1)

    print('Combining stacking_feature...')
    data_x = sparse.hstack((data_x, data_stacking_feat))
    test_x = sparse.hstack((test_x, test_stacking_feat))
    del data_stacking_feat, test_stacking_feat
    gc.collect()

    clf = lgb.LGBMClassifier().set_params(**LGBM_params)
    clf.fit(data_x, data_y, eval_set=[(data_x, data_y)], eval_metric='auc',early_stopping_rounds=20)

    print('Save result...')
    res=pd.read_csv(inputpath+'res.csv')
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(outputpath+'submission_LGBM.csv', index=False)


def XGBTrain_stacking(inputpath, outputpath):
    print('Load dataset...')
    data_x = sparse.load_npz(inputpath+'train_x.npz')
    data_y = pd.read_csv(inputpath+'train_y.csv', header=None)
    test_x = sparse.load_npz(inputpath+'test_x.npz')
    data_stacking_feat = pd.read_csv(inputpath+'stacking_train.csv').drop('XGB', axis=1)
    test_stacking_feat = pd.read_csv(inputpath+'stacking_test.csv').drop('XGB', axis=1)

    print('Combining stacking_feature...')
    data_x = sparse.hstack((data_x, data_stacking_feat))
    test_x = sparse.hstack((test_x, test_stacking_feat))
    del data_stacking_feat, test_stacking_feat
    gc.collect()

    bst = xgb.XGBClassifier().set_params(**XGB_params)
    bst.fit(data_x, data_y, eval_set=[(data_x, data_y)], eval_metric='auc', early_stopping_rounds=20)

    print('Save result...')
    res=pd.read_csv(inputpath+'res.csv')
    res['score'] = bst.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(outputpath+'submission_XGB.csv', index=False)


def blending(inputpath, outputpath):
    def averageBlending(a,b):
        sub1=pd.read_csv(inputpath+'submission_LGBM.csv')
        sub2=pd.read_csv(inputpath+'submission_XGB.csv')
        
        s1=sub1['score'].values
        s2=sub2['score'].values
        
        s=s1*a+s2*b

        res=sub1.drop('score',axis=1)
        res['score']=s
        
        res.to_csv(outputpath+'submission_tmp.csv',index=False) 
        
    def toExpected():
        sub=pd.read_csv(outputpath+'submission_tmp.csv')
        mean=np.mean(sub['score'].values)    
        p=0.0273/mean
        sub['score']=sub['score'].apply(lambda x:x*p)
        sub.to_csv(outputpath+'submission.csv',index=False)
        
    averageBlending(0.5,0.5)
    toExpected()


def main():
    ''' Usage: 
    1. using all data: data.csv,  data/,  res/
    2. using sample data (1%): data_s.csv,  data_s/,  res/
    '''
    if datasetflag == 'all':
        raw_data_name='data.csv'
        inputpath='data/'
        outputpath='res/' + 'Stacking_' + datetime.now().strftime("%Y%m%d_%H%M%S")+'/'
    else:
        raw_data_name='data_s.csv'
        inputpath='data_s/'
        outputpath='res_s/' + 'Stacking_' + datetime.now().strftime("%Y%m%d_%H%M%S")+'/'
    os.mkdir(outputpath)

    ''' PreProcess '''
    if preprocessflag == 'on':
        preprocess.preprocess(inputfilename=raw_data_name,outputpath=inputpath)

    print('-'*40+' stacking train '+'-'*40)
    stacking_train(inputpath, inputpath)
    gc.collect()
    
    print('-'*40+' stacking test '+'-'*40)
    stacking_test(inputpath, inputpath)
    gc.collect()

    print('-'*40+' Model1: LGBM with XGB_stacking '+'-'*40)
    LGBMTrain_stacking(inputpath, outputpath)
    gc.collect()

    print('-'*40+' Model2: XGB with LGBM_stacking '+'-'*40)
    XGBTrain_stacking(inputpath, outputpath)
    gc.collect()

    print('Merge results of two model...')
    blending(outputpath, outputpath)

if __name__=='__main__':
    main()