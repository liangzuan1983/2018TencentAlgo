# 说明

## 文件夹说明

* __raw_data文件夹__：放原始数据：data.csv, data_s.csv，train.csv, test1.csv, adFeature.csv, userFeature.csv/userFeature.data。

> 其中，data.csv通过第一次运行脚本 `preprocess.py` ，由后4个文件拼接而成。生成 `data.csv` 后，其它原始数据文件可删除。`data_s.csv` 文件为 `data.csv` 采样1%的结果，由 `Feature_v?.py` 系列脚本生成，用于后续特征挖掘。

* __data文件夹__：放由全部数据 `data.csv` 生成的训练用数据集和特征表：train_x.npz, train_y.csv, evals_x.npz, evals_y.csv, test_x.npz, res.csv, allfeat.json。

该部分文件通过运行脚本 `preprocess.py` 得到。

* __data_s文件夹__：放由采样数据 `data_s.csv` 生成的训练用数据集和特征表：train_x.npz, train_y.csv, evals_x.npz, evals_y.csv, test_x.npz, res.csv, allfeat.json。

该部分文件通过运行脚本 `preprocess.py` 得到。

* __res文件夹__：内为一系列以时间命名的文件夹，文件夹内为对应时间节点全部数据集训练出的结果 `submission.csv`，所用参数 `parameter.json`，最优代模型 `model.txt`，特征表 `allfeat.json`，及可用于提交的压缩包 `submission.zip`。

* __res_s文件夹__：采样数据集的训练结果，内容同上。


## 脚本文件说明

* __FeatFuns.py__：相关辅助函数的定义。

* __Feature_v1.py__：读取 `data.csv` 文件，添加特征后重新保存到 `data.csv` ，同时生成1%的采样数据 `data_s.csv`，并将添加完特征的完整特征表保存到 `data` 和 `data_s` 文件夹。

> 后续特征添加可仿照该脚本框架，编写新的特征添加脚本Feature_v2.py、Feature_v3.py等。

* __preprocess.py__：读取 `data.csv` 文件（若无该文件，则利用 `raw_data` 中文件生成），将特征独热化或向量稀疏化，并将原始训练集其拆分成训练集（70%）和验证集（30%），生成 `data` 或 `data_s` 文件夹中的一系列训练用文件。

* __baseline.py__：脚本利用 `data` 或 `data_s` 文件夹中的数据集，调用LGBM分类器进行训练，并将结果保存至 `res` 或 `res_s` 文件夹。


## 关于全部训练集和采样数据集的切换

修改baseline.py脚本 `18-20` 行代码，若要训练全部数据集，相关文件路径及文件名输入为：

```
	raw_data_name='data.csv'
	inputpath='data/'
	outputpath='res/'
```

若要训练采样数据集，相关文件路径及文件名为：

```
	raw_data_name='data_s.csv'
	inputpath='data_s/'
	outputpath='res_s/'
```