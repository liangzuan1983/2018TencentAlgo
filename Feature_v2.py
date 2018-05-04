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
data=pd.read_csv(inputpath+'data.csv')
print('Read data done!')


'''Add features'''
# print('Adding aid_age....')
# data['aid_age']= data['aid'] * data['age']
# one_hot_feature.append('aid_age')


# print('Adding adCate_age....')
# data['adCate_age']= data['adCategoryId'] * data['age']
# one_hot_feature.append('adCate_age')


# print('Adding aid_edu....')
# data['aid_edu']= data['aid'] * data['education']
# one_hot_feature.append('aid_edu')


# print('Adding adCate_edu....')
# data['adCate_edu']= data['adCategoryId'] * data['education']
# one_hot_feature.append('adCate_edu')


print('Adding aid_gender....')
data['aid_gender']= data['aid'] * data['gender']
one_hot_feature.append('aid_gender')


print('Adding adCate_gender....')
data['adCate_gender']= data['adCategoryId'] * data['gender']
one_hot_feature.append('adCate_gender')


print('Adding aid_LBS....')
data['aid_LBS']= data['aid'] * data['LBS']
one_hot_feature.append('aid_LBS')


print('Adding adCate_LBS....')
data['adCate_LBS']= data['adCategoryId'] * data['LBS']
one_hot_feature.append('adCate_LBS')


'''save'''
print('save data...')
# save feature list
FeatFuns.save_feat_list(one_hot_feature,vector_feature)
FeatFuns.save_feat_list(one_hot_feature,vector_feature,'data_s/')
# save data
data.to_csv(outputpath+'data.csv', index=False)
print('save data done')


print('slice 300000 samples...')
data_s=data.sample(n=300000)
data_s.to_csv(outputpath+'data_s.csv')
print('slice 300000 samples done')