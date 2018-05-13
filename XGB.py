# -*- coding: utf-8 -*-

import pandas as pd
import xgboost as xgb
from scipy import sparse
import os
import sys
from datetime import datetime
import json
import preprocess
import FeatFuns
from sklearn.model_selection import train_test_split


if len(sys.argv) < 2 or len(sys.argv) > 4:
	print('Usage: python LGBM.py all/sample [on/off(whether to preprocess)] [train_size]')
	exit()

datasetflag = sys.argv[1].lower()
if len(sys.argv) == 2:	
	preprocessflag = 'on'
	train_size = 0.7
elif len(sys.argv) == 3:
	preprocessflag = sys.argv[2].lower()
	train_size = 0.7
else:
	preprocessflag = sys.argv[2].lower()
	train_size = float(sys.argv[3])
    
# Here to set parameter
parameter=dict(max_depth=7, learning_rate=0.25, n_estimators=200, silent=True, objective='binary:logistic', 
	booster='gbtree', n_jobs=-1, random_state=2018, subsample=1, colsample_bytree=1, gamma=0, scale_pos_weight=1,
	reg_lambda=50, missing=0)


def XGBTrain(train_x,train_y,evals_x,evals_y):
	bst = xgb.XGBClassifier().set_params(**parameter)
	bst.fit(train_x, train_y, eval_set=[(evals_x, evals_y)], eval_metric='auc', early_stopping_rounds=50)
	return bst


def main():
	''' Usage: 
	1. using all data: data.csv,  data/,  res/
	2. using sample data (1%): data_s.csv,  data_s/,  res/
	'''
	if datasetflag == 'all':
		raw_data_name='data.csv'
		inputpath='data/'
		outputpath='res/' + 'XGB_' + datetime.now().strftime("%Y%m%d_%H%M%S")+'/'
	else:
		raw_data_name='data_s.csv'
		inputpath='data_s/'
		outputpath='res_s/' + 'XGB_' + datetime.now().strftime("%Y%m%d_%H%M%S")+'/'
	os.mkdir(outputpath)

	''' PreProcess '''
	if preprocessflag == 'on':
		preprocess.preprocess(inputfilename=raw_data_name,outputpath=inputpath)

	''' Load dataset '''
	print('Load data....')
	data_x=sparse.load_npz(inputpath+'train_x.npz')
	res=pd.read_csv(inputpath+'res.csv')
	data_y=pd.read_csv(inputpath+'train_y.csv', header=None)
	test_x=sparse.load_npz(inputpath+'test_x.npz')
	print('Load data done!')

	''' Split dataset '''
	print('slice into train and evals....')
	train_x,evals_x,train_y,evals_y=train_test_split(data_x,data_y,train_size=train_size,shuffle=True,random_state=2333)
	del data_x,data_y
	print('slice into train and evals done')
	print('train dataset, samples:%d' %len(train_y))
	print('evals dataset, samples:%d' %len(evals_y))

	''' Begin Training '''
	print("XGBoost test")
	bst=XGBTrain(train_x,train_y,evals_x,evals_y)

	print('save model...')
	bst._Booster.save_model(outputpath+'model.txt')

	print('save result...')
	res['score'] = bst.predict_proba(test_x)[:,1]
	res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
	res.to_csv(outputpath+'submission_XGB.csv', index=False)
	# os.system('zip -pj '+outputpath+'submission.zip '+outputpath+'submission.csv')

	print('save parameter...')
	with open(outputpath+'parameter.json','w') as outfile:
		json.dump(parameter,outfile)
		outfile.write('\n')

	print('save feature list...')
	one_hot_feature,vector_feature=FeatFuns.load_feat_list(inputpath)
	FeatFuns.save_feat_list(one_hot_feature, vector_feature, outputpath)

if __name__ == "__main__":
	main()