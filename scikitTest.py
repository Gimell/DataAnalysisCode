# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris  # sklearn有数据集，引入即可
from sklearn.model_selection import train_test_split #划分数据集的包
from sklearn import tree #树算法
from sklearn import metrics


def main():
    iris = load_iris()
    print(iris)
    print(len(iris["data"]))
    train_data,test_data,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.2,random_state=1)
    #因为花萼数据比较完整，所以只进行数据划分，划分为训练集和测试集，测试集占20%，random随机取


    #Model 决策树
    clf = tree.DecisionTreeClassifier(criterion="entropy") #参数是增益计算方法，此处选择熵增益
    clf.fit(train_data,train_target) #用训练集进行训练
    y_pred = clf.predict(test_data) #进行预测

    #Verify 两种方式：准确率和混淆矩阵，混淆矩阵横轴表示实际值，纵轴表示预测值。在第几行/列表示实际/预测属于第几类。
    print(metrics.accuracy_score(y_true=test_target,y_pred=y_pred))
    print(metrics.confusion_matrix(y_true=test_target,y_pred=y_pred))
    #决策树还可以输出文件
    with open("./data/test.xlsx","w") as fw:
        tree.export_graphviz(clf,out_file=fw) #将决策树结构输出到文件

if __name__=="__main__":
    main()