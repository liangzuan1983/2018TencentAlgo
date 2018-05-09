# -*- coding: utf-8 -*-

import pandas as pd
import lightgbm as lgb
from scipy import sparse
import sys

if not len(sys.argv) == 3:
	print('Usage: python Predict.py modelname outputpath')
	exit()

modelname = sys.argv[1]
outputpath = sys.argv[2]

''' Load dataset '''
print('Load data....')
res=pd.read_csv('data/res.csv')
test_x=sparse.load_npz('data/test_x.npz')
print('Load data done!')

# load model to predict
print('Load model to predict')
clf = lgb.Booster(model_file='tmp/'+modelname)
res['score'] = clf.predict(test_x)
# res['score'] = clf.predict_proba(test_x)[:,1]
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv(outputpath+'/submission_LGBM.csv', index=False)