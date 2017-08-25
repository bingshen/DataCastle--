# DataCastle
员工离职预测训练赛（练习赛）

目前最好成绩是0.90857

解法并没有想太多，直接把基础特征OneHot之后，然后用LR跑一下就出结果了，提交上去大概是0.89+。简单组合了一下基础特征，得到目前最好结果。

题目链接：http://www.pkbigdata.com/common/cmpt/%E5%91%98%E5%B7%A5%E7%A6%BB%E8%81%8C%E9%A2%84%E6%B5%8B%E8%AE%AD%E7%BB%83%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html

运行步骤：
1 data文件夹下面放上原始数据
2 先执行code_string.py这个文件，先生成train.csv和test.csv，目的是把原始数据中的字符串映射成数值
3 执行logistic_regression_model.py，根据特征和组合特征独热码之后计算预测结果。

另外：xgboost_model.py尝试了一些非线性模型，不过效果都没有LR好。感兴趣的同学可以自行尝试，也可以用stacking技术融合多个模型。



