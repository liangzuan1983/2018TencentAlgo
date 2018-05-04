# -*- coding: utf-8 -*-
import pandas as pd
import os
import json

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
def get_data(filepath, filename):
    if os.path.exists(filepath+filename):
        return pd.read_csv(filepath+filename)
    else:
        ad_feature = pd.read_csv(filepath+'adFeature.csv')
        train=pd.read_csv(filepath+'train.csv')
        predict=pd.read_csv(filepath+'test1.csv')
        train.loc[train['label']==-1,'label']=0
        predict['label']=-1
        user_feature=get_user_feature()
        data=pd.concat([train,predict])
        data=pd.merge(data,ad_feature,on='aid',how='left')
        data=pd.merge(data,user_feature,on='uid',how='left')
        data=data.fillna('-1')
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
        one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType']
        vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
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


''' Following are some funtions for feature calculation '''
def calnum(x):
	if len(x)==1 and x[0]==-1:
		return 0
	return len(x)