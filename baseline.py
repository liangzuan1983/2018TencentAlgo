# -*- coding: utf-8 -*-
# @author: Charles
# base on codes by bryan

import pandas as pd
import lightgbm as lgb
from scipy import sparse
import os
from datetime import datetime
import json
import preprocess
import FeatFuns

inputpath='data_s/'
outputpath='res_s/'

def LGBTrain(parameter,train_x,train_y,evals_x,evals_y):
	clf=lgb.LGBMClassifier(
    boosting_type=parameter['boosting_type'], num_leaves=parameter['num_leaves'], reg_alpha=parameter['reg_alpha'], reg_lambda=parameter['reg_lambda'],
    max_depth=parameter['max_depth'], n_estimators=parameter['n_estimators'], objective=parameter['objective'],
    subsample=parameter['subsample'], colsample_bytree=parameter['colsample_bytree'], subsample_freq=parameter['subsample_freq'],
    learning_rate=parameter['learning_rate'], min_child_weight=parameter['min_child_weight'], random_state=parameter['random_state'], n_jobs=parameter['n_jobs']
    )
	clf.fit(train_x, train_y, eval_set=[(evals_x, evals_y)], eval_metric='auc',early_stopping_rounds=200)
	return clf

''' PreProcess '''
preprocess.preprocess()

''' Load dataset '''
print('Load data....')
train_x=sparse.load_npz(inputpath+'train_x.npz')
evals_x=sparse.load_npz(inputpath+'evals_x.npz')
res=pd.read_csv(inputpath+'res.csv')
train_y=pd.read_csv(inputpath+'train_y.csv', header=None)
evals_y=pd.read_csv(inputpath+'evals_y.csv', header=None)
test_x=sparse.load_npz(inputpath+'test_x.npz')
print('Load data done!')


''' Begin Training '''
print("LGB test")
# Here to set parameter
parameter=dict(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,max_depth=-1, n_estimators=1000, 
	objective='binary',subsample=0.7, colsample_bytree=0.7, subsample_freq=1,learning_rate=0.05, min_child_weight=50, 
	random_state=2018, n_jobs=-1)
clf=LGBTrain(parameter,train_x,train_y,evals_x,evals_y)

# create output path
outputpath = outputpath + datetime.now().strftime("%Y%m%d_%H%M%S")+'/'
os.mkdir(outputpath)

print('save model...')
clf.booster_.save_model(outputpath+'model.txt')

print('save result...')
res['score'] = clf.predict_proba(test_x)[:,1]
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv(outputpath+'submission.csv', index=False)
# os.system('zip '+outputpath+'submission.zip '+outputpath+'submission.csv')

print('save parameter...')
with open(outputpath+'parameter.json','w') as outfile:
	json.dump(parameter,outfile)
	outfile.write('\n')

print('save feature list...')
one_hot_feature,vector_feature=FeatFuns.load_feat_list('data_s/')
save_feat_list(one_hot_feature, vector_feature, outputpath)