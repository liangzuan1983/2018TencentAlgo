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
# print('Adding appInstallNum....')
# data['appInstallNum']= data['appIdInstall'].apply(lambda x:FeatFuns.calnum(x))
# one_hot_feature.append('appInstallNum')
# print('Adding appInstallNum done')

# print('Adding appActionNum....')
# data['appActionNum']= data['appIdAction'].apply(lambda x:FeatFuns.calnum(x))
# one_hot_feature.append('appActionNum')
# print('Adding appActionNum done')

# print('Adding adCatIdPercent....')
# group1=data.groupby(['aid','adCategoryId']).size()
# group2=group1.groupby('adCategoryId').size()
# data['adCatIdPercent']=data['adCategoryId'].apply(lambda x:group2[x]/len(group1))
# del group1, group2
# one_hot_feature.append('adCatIdPercent')
# print('Adding adCatIdPercent done')

# print('Adding prodTypePercent....')
# group1=data.groupby(['aid','productType']).size()
# group2=group1.groupby('productType').size()
# data['prodTypePercent']=data['productType'].apply(lambda x:group2[x]/len(group1))
# del group1, group2
# one_hot_feature.append('prodTypePercent')
# print('Adding prodTypePercent done')

# print('Removing appInstallNum....')
# data=data.drop('appInstallNum',axis=1)
# one_hot_feature.remove('appInstallNum')
# print('Remoing appInstallNum done')


# print('Removing adCatIdPercent....')
# data=data.drop('adCatIdPercent',axis=1)
# one_hot_feature.remove('adCatIdPercent')
# print('Remoing adCatIdPercent done')

# print('Removing appActionNum....')
# data=data.drop('appActionNum',axis=1)
# one_hot_feature.remove('prodTypePercent')
# print('Remoing appActionNum done')


# print('Removing prodTypePercent....')
# data=data.drop('prodTypePercent',axis=1)
# one_hot_feature.remove('prodTypePercent')
# print('Remoing prodTypePercent done')



'''save'''
print('save data...')
# save feature list
FeatFuns.save_feat_list(one_hot_feature,vector_feature)
FeatFuns.save_feat_list(one_hot_feature,vector_feature,'data_s/')
# save data
# data.to_csv(outputpath+'data.csv', index=False)
print('save data done')


print('slice 300000 samples...')
data_s=data.sample(n=300000)
data_s.to_csv('data_s/data_s.csv')
print('slice 300000 samples done')