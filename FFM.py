# -*- coding: utf-8 -*-
from scipy import sparse
import os
from datetime import datetime
import json
import FeatFuns
import pandas as pd
import xlearn as xl
import sys
import preprocess

if len(sys.argv) < 2 or len(sys.argv) > 4:
    print('Usage: python FFM.py all/sample [on/off(Whether to preprocess)] [on/off(Whether to FFM_preprocess)]')
    exit()

datasetflag = sys.argv[1].lower()
if len(sys.argv) == 2:	
	preprocessflag = 'on'
	FFM_preprocessflag = 'on'
elif len(sys.argv) == 3:
	preprocessflag = sys.argv[2].lower()
	FFM_preprocessflag = 'on'
else:
	preprocessflag = sys.argv[2].lower()
	FFM_preprocessflag = sys.argv[3].lower()

# param:
parameter = {
    'task':'binary', 
    'lr':0.2,
    'lambda':0.002, 
    'metric':'auc',
    'epoch': 500,
    'stop_window': 10
    }

def main():
    '''
    前期准备工作
    小样本用 data_s/  res_s/ 全部用data/ res/
    '''
    if datasetflag == 'all':
        raw_data_name='data.csv'
        inputpath='data/'
        outputpath='res/' + 'FFM_' + datetime.now().strftime("%Y%m%d_%H%M%S")+'/'
    else:
        raw_data_name='data_s.csv'
        inputpath='data_s/'
        outputpath='res_s/' + 'FFM_' + datetime.now().strftime("%Y%m%d_%H%M%S")+'/'
    os.mkdir(outputpath)

    ''' Preprocess '''
    if preprocessflag == 'on':
        preprocess.preprocess(inputfilename=raw_data_name,outputpath=inputpath)
    if FFM_preprocessflag == 'on':
        preprocess.FFM_preprocess(inputpath)

    ''' training '''
    ffm_model = xl.create_ffm()
    ffm_model.setTrain(inputpath+'train.txt')
    ffm_model.setValidate(inputpath+'evals.txt')
    ffm_model.fit(parameter, outputpath+'model.out')
    ffm_model.setTest(inputpath+'test.txt')

    ffm_model.setSigmoid()
    ffm_model.predict(outputpath+'model.out',outputpath+'res.txt')

    ''' turn txt to final result '''
    print('turn txt to final result')
    res = pd.read_csv(inputpath + 'res.csv')

    count = 0
    tmp_list=[]
    for line in open(outputpath+'res.txt').readlines():
        if line == os.linesep:
            continue
        else:
            tmp_list.append(float(line))
            count += 1

    # print(count)
    # print(res.shape[0])

    res['score'] = tmp_list
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(outputpath + 'submission_FFM.csv', index=False)



    print('save parameter...')
    with open(outputpath + 'parameter.json', 'w') as outfile:
        json.dump(parameter, outfile)
        outfile.write('\n')

    print('save feature list...')
    one_hot_feature, vector_feature = FeatFuns.load_feat_list(inputpath)
    FeatFuns.save_feat_list(one_hot_feature, vector_feature, outputpath)

if __name__ == "__main__":
	main()