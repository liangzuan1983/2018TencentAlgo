# -*- coding: utf-8 -*-
import pandas as pd
import json
import FeatFuns

inputpath='raw_data/'
outputpath='raw_data/'

# load feature list
one_hot_feature,vector_feature=FeatFuns.load_feat_list()

# read in data
print('Read data....')
data=FeatFuns.get_data()
print('Read data done!')


'''Add features'''
print('Adding LBS_advertiserId....')
data['LBS_advertiserId']= data['advertiserId'] * data['LBS']
one_hot_feature.append('LBS_advertiserId')


print('Adding LBS_creativeId....')
data['LBS_creativeId']= data['LBS'] * data['creativeId']
one_hot_feature.append('LBS_creativeId')


print('Adding LBS_productId....')
data['LBS_productId']= data['LBS'] * data['productId']
one_hot_feature.append('LBS_productId')


print('Adding LBS_productType....')
data['LBS_productType']= data['LBS'] * data['productType']
one_hot_feature.append('LBS_productType')


print('Adding gender_advertiserId....')
data['gender_advertiserId']= data['gender'] * data['advertiserId']
one_hot_feature.append('gender_advertiserId')


print('Adding gender_productType....')
data['gender_productType']= data['gender'] * data['productType']
one_hot_feature.append('gender_productType')


print('Adding age_productType....')
data['age_productType']= data['age'] * data['productType']
one_hot_feature.append('age_productType')


print('Adding age_creativeId....')
data['age_creativeId']= data['age'] * data['creativeId']
one_hot_feature.append('age_creativeId')


'''save'''
print('save data...')
# save feature list
FeatFuns.save_feat_list(one_hot_feature,vector_feature)
FeatFuns.save_feat_list(one_hot_feature,vector_feature,'data_s/')
# save data
data.to_csv(outputpath+'data.csv', index=False)
print('save data done')


print('slice 0.01 samples...')
data_s=data.sample(frac=0.01)
data_s.to_csv(outputpath+'data_s.csv')
print('slice 0.01 samples done')