# -*- coding: UTF-8 -*-
import numpy as np

def main():
    lst=[[1,3,5],[2,4,6]] #lst中可以同时有多种不同的数据类型
    print(type(lst))
    np_lst=np.array(lst)
    #1 Some properties
    print(type(np_lst))
    np_lst=np.array(lst,dtype=np.float) #numpy中只能有一种数据类型
    print(np_lst.shape) #维数
    print(np_lst.ndim) #
    print(np_lst.dtype) #数据类型
    print(np_lst.itemsize) #类型所占字节数
    print(np_lst.size) #元素个数

    #2 random

    #3 Array Opes
    lst=np.arange(1,11).reshape([2,-1]) #没有reshape则默认一维，reshape将其变为2行5列，后边的-1是缺省值，根据2计算出2行需要5列，当然，也可以直接将-1写成5
    print(np.exp(lst)) #指数形式
    print(np.exp2(lst)) #指数的平方
    print(np.sqrt(lst)) #开方

    lst=np.array([[[1,2,3,4],
                   [4,5,6,7]],
                  [[7,8,9,10],
                   [10,11,12,13]],
                  [[14,15,16,17],
                   [18,19,20,21]]
                  ])
    print(lst.sum(axis=0)) #axis取值范围最大维数-1。取0的意思是最外层3个元素进行操作。
    print(lst.max(axis=1))
    print(lst.min(axis=2))

    lst1=np.array([10,20,30,40])
    lst2=np.array([1,2,3,4])

    print(lst1+lst2) #加减乘除 平方
    print(np.dot(lst1.reshape([2,2]),lst2.reshape([2,2]))) #点乘运算，就是矩阵相乘
    print(np.concatenate((lst1,lst2),axis=0)) #连接两个矩阵，连接之后还是一个矩阵
    print(np.vstack((lst1,lst2))) #连接两个矩阵，原矩阵保持不变
    print(np.hstack((lst1,lst2)))
    print(np.split(lst1, 4)) #分成4个数组
    print(np.copy(lst1))

    #4 Linear Agebra
    from numpy.linalg import *
    print(np.eye(3)) #单位矩阵
    lst= np.array([[1,2],[3,4]])
    print("Inv:")
    print(inv(lst)) #逆矩阵
    print(lst.transpose()) #转置矩阵
    print(det(lst)) #行列式
    print(eig(lst)) #特征值和特征向量
    y=np.array([[5.],[7.]])
    print(solve(lst,y)) #解方程1x+2y=5，3x+4y=7

    #5 Others
    print(np.fft.fft(np.array([1,1,1,1,1,1,1,1]))) #fft是信号处理
    print(np.corrcoef([1,0,1],[0,2,1]))
    print(np.polyld([2,1,3]))

if __name__=="__main__":
    main()