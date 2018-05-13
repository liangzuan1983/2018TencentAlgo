# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import json
from sklearn.utils import shuffle
from scipy import sparse

# get userfeature
def get_user_feature(filepath):
    if os.path.exists(filepath+'userFeature.csv'):
        user_feature=pd.read_csv(filepath+'userFeature.csv')
    else:
        userFeature_data = []
        with open(filepath+'userFeature.data', 'r') as f:
            cnt = 0
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
                if i % 1000000 == 0:
                    user_feature = pd.DataFrame(userFeature_data)
                    user_feature.to_csv(filepath+'userFeature_' + str(cnt) + '.csv', index=False)
                    cnt += 1
                    del userFeature_data, user_feature
                    userFeature_data = []
            user_feature = pd.DataFrame(userFeature_data)
            user_feature.to_csv(filepath+'userFeature_' + str(cnt) + '.csv', index=False)
            del userFeature_data, user_feature
            user_feature = pd.concat(
                [pd.read_csv(filepath+'userFeature_' + str(i) + '.csv') for i in range(cnt + 1)]).reset_index(drop=True)
            user_feature.to_csv(filepath+'userFeature.csv', index=False)
    return user_feature

# get data
def get_data(filename='data.csv'):
    filepath='raw_data/'
    if os.path.exists(filepath+filename):
        return pd.read_csv(filepath+filename)
    else:
        ad_feature = pd.read_csv(filepath+'adFeature.csv')
        train=pd.read_csv(filepath+'train.csv')
        predict=pd.read_csv(filepath+'test1.csv')
        train.loc[train['label']==-1,'label']=0
        predict['label']=-1
        user_feature=get_user_feature(filepath)
        data=pd.concat([train,predict])
        data=pd.merge(data,ad_feature,on='aid',how='left')
        data=pd.merge(data,user_feature,on='uid',how='left')
        data=data.fillna('0')
        del user_feature
        data.to_csv(filepath+'data.csv')
        return data

# load feature list
def load_feat_list(inputpath='data/'):
    if os.path.exists(inputpath+'allfeat.json'):
        allfeat=list()
        with open(inputpath+'allfeat.json','r') as infile:
            allfeat=[json.loads(line) for line in infile]
        one_hot_feature=list(allfeat[0].values())
        vector_feature=list(allfeat[1].values())
    else:
        one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType']
        vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','os','ct','marriageStatus']
        save_feat_list(one_hot_feature,vector_feature,inputpath)

    return one_hot_feature,vector_feature


# save feature list
def save_feat_list(one_hot_feature, vector_feature, outputpath='data/'):
	oh_feat={i:one_hot_feature[i] for i in range(len(one_hot_feature))}
	vec_feat={i:vector_feature[i] for i in range(len(vector_feature))}
	with open(outputpath+'allfeat.json','w') as outfile:
	    json.dump(oh_feat,outfile)
	    outfile.write('\n')
	    json.dump(vec_feat,outfile)  
	    outfile.write('\n')


# Split dataset
# def split_dataset(data_x, data_y, train_size = 0.7):

#     # shuffle
#     data_x,data_y = shuffle(data_x,data_y,random_state = 2333)

#     if train_size == 1:
#         return data_x, data_y, data_x, data_y

#     # Split
#     data_x = data_x.tocoo()
#     rown_data = data_x.shape[0]
#     rown_train = int(train_size * rown_data)
#     n_train = len(data_x.row[data_x.row < rown_train])

#     ind=np.arange(len(data_x.data))
#     ind_train=ind[:n_train]
#     ind_evals=ind[n_train:]

#     train_x = sparse.coo_matrix(
#         (data_x.data[ind_train],(data_x.row[ind_train], data_x.col[ind_train])),
#         shape=(rown_train,data_x.shape[1]))
#     train_y = data_y[:rown_train]

#     evals_x = sparse.coo_matrix(
#         (data_x.data[ind_evals],(data_x.row[ind_evals]-rown_train, data_x.col[ind_evals])),
#         shape=(rown_data-rown_train,data_x.shape[1]))
#     evals_y = data_y[rown_train:]

#     return train_x,train_y,evals_x,evals_y 


''' save model per 1000 iter '''
def save_model(per_iter = 1000):
    def callback(env):
        iter = env.iteration - env.begin_iteration
        if iter % per_iter == 0 and not iter == 0:
            print('Save model per {0} iter...'.format(per_iter))
            env.model.save_model('tmp/model_{0}.txt'.format(iter))
    callback.before_iteration = True
    callback.order = 0
    return callback


''' Following are some funtions for feature calculation '''
def calnum(x):
	if len(x)==1 and x[0]==0:
		return 0
	return len(x)