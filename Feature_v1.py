# -*- coding: utf-8 -*-
import pandas as pd
import json
import FeatFuns

inputpath='raw_data/'
outputpath='raw_data/'

# load feature list
# one_hot_feature,vector_feature=FeatFuns.load_feat_list()

# read in data
print('Read data....')
data=FeatFuns.get_data()
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

# one_hot_feature.remove('os')
# one_hot_feature.remove('ct')
# one_hot_feature.remove('marriageStatus')
# vector_feature.append('os')
# vector_feature.append('ct')
# vector_feature.append('marriageStatus')

for col_name in data.columns:
	if col_name=='label':
		continue
	data[col_name]=data[col_name].replace('-1','0')


'''save'''
print('save data...')
# save feature list
# FeatFuns.save_feat_list(one_hot_feature,vector_feature)
# FeatFuns.save_feat_list(one_hot_feature,vector_feature,'data_s/')
# save data
data.to_csv(outputpath+'data.csv', index=False)
print('save data done')


print('slice 0.01 samples...')
data_s=data.sample(frac=0.01)
data_s.to_csv(outputpath+'data_s.csv')
print('slice 0.01 samples done')