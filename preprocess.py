# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import sparse
import os
import json
import FeatFuns
import pickle
import gc
from xlearn import write_data_to_xlearn_format


def preprocess(inputfilename='data.csv',outputpath='data/'):

    # load feature list
    one_hot_feature,vector_feature=FeatFuns.load_feat_list(outputpath)
    print(one_hot_feature,vector_feature)
    print(len(one_hot_feature),len(vector_feature))

    # read in data
    print('Read data....')
    data=FeatFuns.get_data(inputfilename)
    print(data.shape[0],data.shape[1])
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
    gc.collect()

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

    # 定义一个field_list
    # 含义如下
    # list每一个下标对应同样下标的列 （均从1开始）
    # 不知道从0开始可不可以 为了防止bug  老实从1开始
    # 元素代表该列（或者该feature）所属的field
    # 如field_list[50] 的值 为第50列所属的 field

    field_size=[1]  # creativeSize
    # current_field = 1  #下一个将要添加到field_list中的field 是 current_field  显然第一个是1

    train_x=train[['creativeSize']]
    # evals_x=evals[['creativeSize']]
    test_x=test[['creativeSize']]

    # field_list.append(1)   #creativeSize 属于filed 1
    # current_field += 1

    # creativeSize 没有编码 直接用

    print('Total one hot feature: %d' %len(one_hot_feature))
    cnt=0
    enc = OneHotEncoder()
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))

        train_a=enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))

        field_size.append(train_a.shape[1])

        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

        data = data.drop(feature, axis = 1)
        train = train.drop(feature, axis = 1)
        test = test.drop(feature, axis = 1)
        gc.collect()
        cnt+=1
        print(feature+' finish. '+'Finished %d features.' %cnt)
    print('one-hot prepared!')

    print('Total vector feature: %d' %len(vector_feature))
    cnt=0
    cv=CountVectorizer(token_pattern=r'-?\d+')
    for feature in vector_feature:
        cv.fit(data[feature])

        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])

        field_size.append(train_a.shape[1])

        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

        data = data.drop(feature, axis = 1)
        train = train.drop(feature, axis = 1)
        test = test.drop(feature, axis = 1)
        gc.collect()
        cnt+=1
        print(feature+' finish. '+'Finished %d features.' %cnt)
    print('cv prepared !')

    del data
    del train
    del test
    gc.collect()

    # save
    print('save data...')
    sparse.save_npz(outputpath+'train_x.npz',train_x)
    sparse.save_npz(outputpath+'test_x.npz',test_x)
    res.to_csv(outputpath+'res.csv', index=False)
    train_y.to_csv(outputpath+'train_y.csv', index=False)

    with open(outputpath+'field_size.pk', 'wb+') as f:        # 卧槽！！md 要wb+  读取要rb   我可去你个大西瓜
        pickle.dump(field_size, f)

    print('save data done')

# def FFM_preprocess(inputpath='data/'):

#     '''
#     在preprocess中已经保存了
#     训练集和验证集feature  train_x.npz
#     训练集和验证集label   train_y.csv
#     测试集feature  test_x.npz
#     腾讯给的答题卡  res.csv
#     '''


#     ''' Load dataset '''
#     print('Load data....')
#     data_x = sparse.load_npz(inputpath + 'train_x.npz')
#     data_y = pd.read_csv(inputpath + 'train_y.csv', header=None)
#     data_y = np.array(data_y[0])    #  把data_y 转化成numpy的array 因为只有一列数据 用DataFrame很冗余

#     test_x = sparse.load_npz(inputpath + 'test_x.npz')
#     #res = pd.read_csv(inputpath + 'res.csv')

#     with open(inputpath+'field_size.pk','rb') as f:
#         field_size = pickle.load(f)
#     print('Load data done!')
#     # print field_list

#     ''' Prepare field list '''
#     print('Preparing field list....')
#     field_start = [0]
#     for item in field_size:
#         field_start.append(field_start[-1] + item)

#     field = 0
#     field_list = dict()
#     for column in range(field_start[-1]):
#         while field_start[field] <= column:
#             field += 1
#         field_list[column] = field - 1
#     print('Preparing field list done!')

#     ''' Split dataset '''
#     print('slice into train and evals....')
#     train_x, evals_x, train_y, evals_y = train_test_split(data_x,data_y,test_size=0.01)
#     del data_x, data_y
#     print('slice into train and evals done')
#     print('train dataset, samples:%d' % len(train_y))
#     print('evals dataset, samples:%d' % len(evals_y))


#     '''转换稀疏矩阵类型 使得row col 有序排列'''
#     train_x = train_x.tocoo()
#     evals_x = evals_x.tocoo()
#     test_x = test_x.tocsr()
#     test_x = test_x.tocoo()

#     ''' 写 第一个txt====》 训练集的txt '''
#     print('start write train.txt')
#     with open(inputpath+'train.txt', 'w') as f:
#         count = 0  # 用于遍历train_x 中的row
#         row_size = train_x.row.shape[0]

#         for row in range(train_x.shape[0]):   # row 代表当前是写第几行
#             # 先写label
#             f.write(str(train_y[row]))

#             # 接着写后面的  field:feature:value 这里我们的value全是1 除了第一个creativesize

#             while count < row_size and train_x.row[count] <= row :  # 如果后面还有 而且 依旧是当前行的
#                 f.write('\t')
#                 # 写field
#                 f.write(str(field_list[train_x.col[count]] ) + ':')   #我就是喜欢嵌套一堆[]和() 咬我啊   +1是失误
#                 # 写feature 和value  其中value就是1 除了第一个creativesize
#                 if train_x.col[count] == 0:
#                     f.write(str(train_x.col[count]) + ':' + str(train_x.data[count]))    # 为什么要+1 呢 因为 假设col是5 那么他就是第六列 写进去的应该是6
#                 else:
#                     f.write(str(train_x.col[count]) + ':' + '1')

#                 count +=1

#             f.write(os.linesep)
#     print('write train.txt  done')

#     ''' 写 第二个txt====》 测试的txt '''
#     print('start write evals.txt')
#     with open(inputpath+'evals.txt', 'w') as f:
#         count = 0
#         row_size = evals_x.row.shape[0]

#         for row in range(evals_x.shape[0]):  # row 代表当前是写第几行
#             # 先写label
#             f.write(str(evals_y[row]))

#             # 接着写后面的  field:feature:value 这里我们的value全是1 除了第一个creativesize

#             while count < row_size and evals_x.row[count] <= row:  # 如果后面还有 而且 依旧是当前行的
#                 f.write('\t')
#                 # 写field
#                 f.write(str(field_list[evals_x.col[count]]) + ':')  # 我就是喜欢嵌套一堆[]和() 咬我啊   +1是失误
#                 # 写feature 和value  其中value就是1 除了第一个creativesize
#                 if evals_x.col[count] == 0:
#                     f.write(str(evals_x.col[count]) + ':' + str(evals_x.data[count]))    # 为什么要+1 呢 因为 假设col是5 那么他就是第六列 写进去的应该是6
#                 else:
#                     f.write(str(evals_x.col[count]) + ':' + '1')

#                 count += 1

#             f.write(os.linesep)
#     print('write evals.txt done')
#     ''' 写 第三个txt====》 测试集的txt '''

#     print('start write test.txt')
#     with open(inputpath+'test.txt', "w") as f:
#         count = 0  # 用于遍历train_x 中的row
#         row_size = test_x.row.shape[0]

#         for row in range(test_x.shape[0]):  # row 代表当前是写第几行
#             # 先写label  这个随便写  占label位置即可
#             f.write('0')

#             # 接着写后面的  field:feature:value 这里我们的value全是1 除了第一个creativesize

#             while count < row_size and test_x.row[count] <= row:  # 如果后面还有 而且 依旧是当前行的
#                 f.write('\t')
#                 # 写field
#                 f.write(str(field_list[test_x.col[count]]) + ':')  # 我就是喜欢嵌套一堆[]和() 咬我啊   +1是失误
#                 # 写feature 和value  其中value就是1 除了第一个creativesize
#                 if test_x.col[count] == 0:
#                     f.write(str(test_x.col[count]) + ':' + str(test_x.data[count]))    # 为什么要+1 呢 因为 假设col是5 那么他就是第六列 写进去的应该是6
#                 else:
#                     f.write(str(test_x.col[count]) + ':' + '1')
#                 count += 1

#             f.write(os.linesep)
#     print('write test.txt done')
#     '''
#     至此 FFM_preprocess 处理完毕
#     生成了三个txt 将用于FFM脚本的使用
#     '''

def convertion(inputpath='data/'):

    ''' Load dataset '''
    print('Load data....')
    data_x = sparse.load_npz(inputpath + 'train_x.npz')
    data_y = pd.read_csv(inputpath + 'train_y.csv', header=None)
    data_y = np.array(data_y[0])
    test_x = sparse.load_npz(inputpath + 'test_x.npz').tocsr()
    test_y = np.zeros(test_x.shape[0])

    ''' Prepare field list '''
    print('Preparing field list....')
    with open(inputpath+'field_size.pk','rb') as f:
        field_size = pickle.load(f)
    field_start = [0]
    for item in field_size:
        field_start.append(field_start[-1] + item)
    field = 0
    field_list = list()
    for column in range(field_start[-1]):
        while field_start[field] <= column:
            field += 1
        field_list.append(field - 1)
    field_list = np.array(field_list).astype('int32')

    print('slice into train and evals....')
    train_x, evals_x, train_y, evals_y = train_test_split(data_x,data_y,test_size=0.01)
    del data_x, data_y
    gc.collect()
    print('train dataset, samples:%d' % len(train_y))
    print('evals dataset, samples:%d' % len(evals_y))

    print('start write train.txt....')
    write_data_to_xlearn_format(train_x, train_y, inputpath+'train.txt', fields = field_list)
    print('start write evals.txt....')
    write_data_to_xlearn_format(evals_x, evals_y, inputpath+'evals.txt', fields = field_list)
    print('start write test.txt....')
    write_data_to_xlearn_format(test_x, test_y, inputpath+'test.txt', fields = field_list)