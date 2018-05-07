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

# Here to set parameter
parameter=dict(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,max_depth=-1, n_estimators=10000, 
		objective='binary',subsample=0.7, colsample_bytree=0.7, subsample_freq=1,learning_rate=0.05, min_child_weight=50, 
		random_state=2018, n_jobs=-1)



def LGBTrain(parameter,train_x,train_y,evals_x,evals_y):
	clf=lgb.LGBMClassifier()
	clf.set_params(**parameter)
	clf.fit(train_x, train_y, eval_set=[(evals_x, evals_y)], eval_metric='auc',early_stopping_rounds=1000)
	return clf


def main():
	''' Usage: 
	1. using all data: data.csv,  data/,  res/
	2. using sample data (1%): data_s.csv,  data_s/,  res/
	'''
	raw_data_name='data.csv'
	inputpath='data/'
	outputpath='res/' + 'LGBM_' + datetime.now().strftime("%Y%m%d_%H%M%S")+'/'
	os.mkdir(outputpath)

	''' PreProcess '''
	preprocess.preprocess(inputfilename=raw_data_name,outputpath=inputpath)

	''' Load dataset '''
	print('Load data....')
	data_x=sparse.load_npz(inputpath+'train_x.npz')
	# evals_x=sparse.load_npz(inputpath+'evals_x.npz')
	res=pd.read_csv(inputpath+'res.csv')
	data_y=pd.read_csv(inputpath+'train_y.csv', header=None)
	# evals_y=pd.read_csv(inputpath+'evals_y.csv', header=None)
	test_x=sparse.load_npz(inputpath+'test_x.npz')
	print('Load data done!')

	''' Split dataset '''
	print('slice into train and evals....')
	train_x,train_y,evals_x,evals_y=FeatFuns.split_dataset(data_x,data_y)
	del data_x,data_y
	print('slice into train and evals done')
	print('train dataset, samples:%d' %len(train_y))
	print('evals dataset, samples:%d' %len(evals_y))

	''' Begin Training '''
	print("LGB test")
	clf=LGBTrain(parameter,train_x,train_y,evals_x,evals_y)

	print('save model...')
	clf.booster_.save_model(outputpath+'model.txt')

	print('save result...')
	res['score'] = clf.predict_proba(test_x)[:,1]
	res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
	res.to_csv(outputpath+'submission_LGBM.csv', index=False)
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