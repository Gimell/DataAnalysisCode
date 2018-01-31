# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

def main():
    #Data Structure
    s=pd.Series([i*2 for i in range(1,11)])
    print(type(s))
    dates=pd.date_range('20170301',periods=8)
    #dataframe的第一种定义方式
    df=pd.DataFrame(np.random.randn(8,5),index=dates,columns=list("ABCDE"))
    print(df)
    #dataframe的第二种定义方式
    #df=pd.DataFrame({"A":1,"B":pd.Timestamp("20170301"),"C":pd.Series(1,index=list(range(4)),dtype="float32"),\
    #                 "D":np.array([3]*4,dtype="float32"),"E":pd.Categorical(["police","student","teacher","doctor"])})
    #print(df)

    #Basic
    print(df.head(3))
    print(df.tail(3))
    print(df.index)
    print(df.values)
    print(df.T) #转置
    #print(df.sort(columus="C")) #对C列升序排列 这句代码有问题
    print(df.sort_index(axis=1,ascending=False)) #对属性名称降序排列
    print(df.describe()) #所有属性值的数量、平均值、最小值、四分位数、中位数、上四分位数、最大值

    #Select Split
    print(df["A"])
    print(type(df["A"]))
    print(df[:3]) #使用下标进行切片
    print(df["20170301":"20170304"]) #使用索引进行切片
    print(df.loc[dates[0]])
    print(df.loc["20170301":"20170304",["B","D"]])
    print(df.at[dates[0],"C"])

    print(df.iloc[1:3,2:4])
    print(df.iloc[1,4])
    print(df.iat[1,4])

    print(df[df.B>0][df.A<0])
    print(df[df>0])
    print(df[df["E"].isin([1,2])])

    #Set
    s1=pd.Series(list(range(10,18)),index=pd.date_range("20170301",periods=8))
    df["F"]=s1
    print(df)
    df.at[dates[0],"A"]=0
    print(df)
    df.iat[1,1]=1
    df.loc[:,"D"]=np.array([4]*len(df))
    print(df)
    df2=df.copy()
    df2[df2>0]=-df2
    print(df2)

    #Missing Values
    df1=df.reindex(index=dates[:4],columns=list("ABCD")+["G"]) #取前四行，ABCDG属性
    df1.loc[dates[0]:dates[1],"G"]=1 #对G进行赋值，仅对第一行第二行的进行赋值
    print(df1)
    #缺失值的填充方式有两种，一是丢弃，二是填充，填充又有两种，一是填充固定值，二是插值，插值可以参考scipy中的插值算法
    print(df1.dropna()) #丢弃处理
    print(df1.fillna(value=2)) #填充固定值

    #Statistic
    print(df.mean()) #平均值,得到的放回结果就是
    print(df.var()) #方差
    s=pd.Series([1,2,4,np.nan,5,7,9,10],index=dates) #构造一个Series
    print(s)
    print(s.shift(2)) #将所有的值纵向下移两位，不是循环移位，所以用空值补齐
    print(s.diff()) #不写数字表示一阶，当前数字减前一位数字
    print(s.value_counts()) #每个值在Series中出现的次数，这个值可以用来绘制直方图
    print(df.apply((np.cumsum))) #动态应用，cumsum表示累加
    print(df.apply((lambda x:x.max()-x.min())))

    #Contact 表格拼接和类SQL的操作
    pieces=[df[:3],df[-3:]] #将表格前3行和后3行拼接起来
    print(pd.concat(pieces))
    left = pd.DataFrame({"key": ["x", "y"], "value": [1, 2]})
    right = pd.DataFrame({"key": ["x", "z"], "value": [1, 2]})
    print("LEFT:",left)
    print("RIGHT:",right)
    print(pd.merge(left,right,on="key",how="left")) #类似于SQL语句XXX，how的参数还可以是inner，outer
    df3=pd.DataFrame({"A":["a","b","c","b"],"B":list(range(4))}) #类似SQL语句GroupBy
    print(df3.groupby("A").sum())


    #Reshape 透视表 交叉分析
    import datetime
    df4=pd.DataFrame({'A':['one','one','two','three']*6, #这个表格共有4*6=24行
                      'B':['a','b','c']*8,
                      'C':['foo','foo','foo','bar','bar','bar']*4,
                      'D':np.random.randn(24),
                      'E':np.random.randn(24),
                      'F':[datetime.datetime(2017,i,1) for i in range(1,13)]+
                          [datetime.datetime(2017,i,15)for i in range(1,13)]})
    print("      ")
    print(pd.pivot_table(df4,values="D",index=["A","B"],columns=["C"]))



    #Time Series
    t_exam=pd.date_range("20170301",periods=10,freq="S")
    print(t_exam)


    #Graph
    ts=pd.Series(np.random.randn(1000),index=pd.date_range("20170301",periods=1000))
    ts=ts.cumsum()
    from pylab import *
    ts.plot()
    show()


    #File
    df6=pd.read_csv("./test.csv") #读取
    print(df6)
    df7=pd.read_excel("./test.xlsx","Sheet1")
    print("Excel",df7)
    df6.to_csv("./test1.csv") #将表格保存到test1.csv文件中
    df7.to_excel("./test1.excel")


if __name__ == '__main__':
    main()