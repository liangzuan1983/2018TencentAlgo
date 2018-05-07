# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import json
import FeatFuns


def preprocess(inputfilename='data.csv',outputpath='data/'):

    # load feature list
    one_hot_feature,vector_feature=FeatFuns.load_feat_list(outputpath)
    print(one_hot_feature,vector_feature)

    # read in data
    print('Read data....')
    data=FeatFuns.get_data(inputfilename)
    print(data.shape[0])
    print('Read data done!')

    print('Drop some columns...')
    data = data.drop(['appIdAction', 'appIdInstall', 'interest3', 'interest4', 'kw3', 'topic3'], axis=1)
    vector_feature.remove('appIdAction')
    vector_feature.remove('appIdInstall')
    vector_feature.remove('interest3')
    vector_feature.remove('interest4')
    vector_feature.remove('kw3')
    vector_feature.remove('topic3')
    print('Drop some columns done!')

    ''' transform the features '''
    print('one-hot....')
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
    print('one-hot done!')

    train=data[data['label']!=-1]
    test=data[data['label']==-1]
    train_y=train.pop('label')
    test=test.drop('label',axis=1)
    res=test[['aid','uid']]

    # print('slice into train and evals....')
    # train_per=0.7
    # size=len(train)
    # end=int(size*train_per)
    # r=np.random.permutation(size)   # shuffle
    # train=train[r,:]
    # evals=train[end:size]
    # train=train[0:end]

    # train_y = train_label[:end]
    # assert len(train_y)==len(train)
    # evals_y = train_label[end:size]
    # assert len(evals_y)==len(evals)
    # del train_label
    # print('slice into train and evals done!')
    # print('%f train dataset, samples:%d' %(train_per,len(train_y)))
    # print('%f evals dataset, samples:%d' %((1-train_per),len(evals_y)))

    train_x=train[['creativeSize']]
    # evals_x=evals[['creativeSize']]
    test_x=test[['creativeSize']]

    print('Total vector feature: %d' %len(vector_feature))
    cnt=0
    cv=CountVectorizer(token_pattern=r'-?\d+')
    for feature in vector_feature:
        cv.fit(data[feature])

        train_a = cv.transform(train[feature])
        # evals_a = cv.transform(evals[feature])
        test_a = cv.transform(test[feature])

        train_x = sparse.hstack((train_x, train_a))
        # assert len(train_y)==train_x.shape[0]
        # evals_x = sparse.hstack((evals_x, evals_a))
        # assert len(evals_y)==evals_x.shape[0]
        test_x = sparse.hstack((test_x, test_a))

        cnt+=1
        print(feature+' finish. '+'Finished %d features.' %cnt)
    print('cv prepared !')

    print('Total one hot feature: %d' %len(one_hot_feature))
    cnt=0
    enc = OneHotEncoder()
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))

        train_a=enc.transform(train[feature].values.reshape(-1, 1))
        # evals_a=enc.transform(evals[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))

        train_x = sparse.hstack((train_x, train_a))
        # assert len(train_y)==train_x.shape[0]
        # evals_x = sparse.hstack((evals_x, evals_a))
        # assert len(evals_y)==evals_x.shape[0]
        test_x = sparse.hstack((test_x, test_a))

        cnt+=1
        print(feature+' finish. '+'Finished %d features.' %cnt)
    print('one-hot prepared!')

    del data
    del train
    del test

    # save
    print('save data...')
    sparse.save_npz(outputpath+'train_x.npz',train_x)
    # sparse.save_npz(outputpath+'evals_x.npz',evals_x)
    sparse.save_npz(outputpath+'test_x.npz',test_x)
    res.to_csv(outputpath+'res.csv', index=False)
    train_y.to_csv(outputpath+'train_y.csv', index=False)
    # evals_y.to_csv(outputpath+'evals_y.csv', index=False)
    print('save data done')