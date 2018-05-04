# 说明

## 文件夹说明

* __raw_data文件夹__：放原始数据：data.csv, train.csv, test1.csv, adFeature.csv, userFeature.csv/userFeature.data。

> 其中，__data.csv__通过__第一次__运行脚本__preprocess.py__，由后4个文件拼接而成。
> 生成data.csv后，其它原始数据文件可删除。

* __data文件夹__：放训练用数据集和特征表：train_x.npz, train_y.csv, evals_x.npz, evals_y.csv, test_x.npz, res.csv, allfeat.json。

该部分文件通过运行脚本__preprocess.py__得到。

* __res文件夹__：内为一系列以时间命名的文件夹，文件夹内为对应时间节点训练出的结果__submission.csv__,所用参数__parameter.json__，及可用于提交的压缩包__submission.zip__。


## 脚本文件说明

* __FeatFuns.py__：相关辅助函数的定义。

* __Feature_v1.py__：读取__data.csv__文件，添加特征后重新保存到__data.csv__，并将添加完特征的完整特征表保存到__data__文件夹。

> 后续特征添加可仿照该脚本框架，编写新的特征添加脚本Feature_v2.py、Feature_v3.py等。

* __preprocess.py__：读取__data.csv__文件（若无该文件，则利用__raw_data__中文件生成），将特征one-hot化或向量稀疏化，并将原始训练集其拆分成训练集（70%）和验证集（30%），

生成__data__文件夹中的一系列训练用文件。

* __baseline.py__：脚本利用__data__文件夹中的数据集，调用LGB分类器进行训练，并将结果保存至__res__文件夹。