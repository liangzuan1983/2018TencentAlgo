# -*- coding: utf-8 -*-
# @author: Charles
# base on codes by bryan

import pandas as pd
import lightgbm as lgb
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
parameter=dict(boosting_type='gbdt', num_leaves=355, reg_alpha=0.0, reg_lambda=1,max_depth=-1, n_estimators=1500, 
		objective='binary',subsample=0.7, colsample_bytree=0.7, subsample_freq=1,learning_rate=0.02, min_child_weight=50, 
		random_state=2018, n_jobs=-1)



def LGBTrain(parameter,train_x,train_y,evals_x,evals_y,callbacks=[None]):
	clf = lgb.LGBMClassifier().set_params(**parameter)
	clf.fit(train_x, train_y, eval_set=[(evals_x, evals_y)], eval_metric='auc',early_stopping_rounds=500,callbacks=callbacks)
	return clf


def main():
	''' Usage: 
	1. using all data: data.csv,  data/,  res/
	2. using sample data (1%): data_s.csv,  data_s/,  res/
	'''
	if datasetflag == 'all':
		raw_data_name='data.csv'
		inputpath='data/'
		outputpath='res/' + 'LGBM_' + datetime.now().strftime("%Y%m%d_%H%M%S")+'/'
	else:
		raw_data_name='data_s.csv'
		inputpath='data_s/'
		outputpath='res_s/' + 'LGBM_' + datetime.now().strftime("%Y%m%d_%H%M%S")+'/'
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
	print("LGB test")
	if datasetflag == 'all':
		clf=LGBTrain(parameter,train_x,train_y,evals_x,evals_y,callbacks=[FeatFuns.save_model(500)])
	else:
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