# -*- coding: UTF-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation #Dense表示要激活的层，Activation表示激活函数
from keras.optimizers import SGD #随机梯度下降算法
from sklearn.datasets import load_iris
from sklearn.preprocessing import  LabelBinarizer
from sklearn.model_selection import train_test_split


def main():
    iris=load_iris()
    #因为此神经网络是分类器，需要将序列标签化。类别有0，1，2这三种，所以将其转化为(1,0,0),(0,1,0),(0,0,1)
    print(LabelBinarizer().fit_transform(iris["target"]))
    #分成训练集和测试集
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.2,
                                                                        random_state=1)
    #将训练集的标签和测试集的标签序列化
    labels_train = LabelBinarizer().fit_transform(train_target)
    labels_test = LabelBinarizer().fit_transform(test_target)

    #构建神经网络层（model）
    model=Sequential(
        [
            Dense(5,input_dim=4), #第一层输出5个，输入有4个
            Activation("relu"),#激活函数选择sigmoid
            Dense(3), #输入是上一层的输出，共有5个，此处可以省略，输出有3个，是label
            Activation("sigmoid"),
        ]
    )
    #构建模型的第二种方法
    #model=Sequential()
    #model.add(Dense(5,input=4))
    sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(optimizer=sgd,loss="categorical_crossentropy")
    model.fit(train_data,labels_train,nb_epoch=200,batch_size=40)#nb_epoch指定训练次数，batch_size指定一次训练用多少数据
    print(model.predict_classes(test_data))
    model.save_weights("./data/w") #保存model的参数，下次直接使用
    model.load_weights("./data/w")



if __name__ == "__main__":
    main()
