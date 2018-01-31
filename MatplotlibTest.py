# -*- coding: UTF-8 -*-
import numpy as np

def main():

    #line
    import matplotlib.pyplot as plt
    '''
    x=np.linspace(-np.pi,np.pi,256,endpoint=True)
    c,s=np.cos(x),np.sin(x)
    plt.figure(1)
    plt.plot(x,c,color="blue",linewidth=1.0,linestyle="--",label="COS",alpha=0.5) #x自变量，c因变量，linestyle线条形式，此处是虚线
    plt.plot(x,s,"r*",label="SIN") #x自变量，s因变量,r red/right,*组成线
    plt.title("COS & SIN")
    ax=plt.gca() #轴编辑器
    ax.spines["right"].set_color("none") #隐藏右边和上边的线
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_position(("data",0)) #左边和底边的线移到中间，形成常见坐标轴
    ax.spines["bottom"].set_position(("data",0))
    plt.xticks([-np.pi,np.pi/2.0,np.pi/2,np.pi],
                [r'$-\pi$',r'$-\pi/2$',r'$0$',r'$0$',r'$+\pi/2$',r'$+\pi$']) #第一个数组指定显示位置，第二个数组指定显示内容
    plt.yticks(np.linspace(-1,1,5,endpoint=True))
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor="white",edgecolor="None",alpha=0.2))
    plt.legend(loc="upper left")
    plt.grid() #打印网格线
    #plt.axis([-1,1,-0.5,1]) #指定横轴和纵轴的显示范围
    plt.fill_between(x,np.abs(x)<0.5,c,c>0.5,color="green",alpha=0.25) #x指明横轴取值范围,c指明y轴取值范围
    t=1 #在t=1处进行添加线
    plt.plot([t,t],[0,np.cos(t)],"y",linewidth=3,linestyle="--")
    plt.annotate("cos(1)",xy=(t,np.cos(1)),xycoords="data",xytext=(+10,+30),
                 textcoords="offset points",arrowprops=dict(arrowstylr="->",connectionstyle="arc3,rad=.2")) #为刚刚添加的线加注释
    plt.show()
    '''
    


    #scatter 散点图
    fig=plt.figure()
    ax=fig.add_subplot(3,3,1) #3行3列第1个
    n=128
    X=np.random.normal(0,1,n)
    Y=np.random.normal(0,1,n)
    T=np.arctan2(Y,X)
    #plt.axes([0.025,0.025,0.95,0.95]) #作用是将图片填满空间
    ax.scatter(X,Y,s=75,c=T,alpha=.5)
    plt.xlim(-1.5,1.5),plt.xticks([])
    plt.axis()
    plt.title("scatter")


    #bar 柱状图
    ax=fig.add_subplot(332)
    n=10
    X=np.arange(n) #构建0到9的数列
    Y1=(1-X/float(n))*np.random.uniform(0.5,1.0,n)
    Y2=(1-X/float(n))*np.random.uniform(0.5,1.0,n)

    ax.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    ax.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
    for x,y in zip(X,Y1):
        ax.text(x + 0.4, y + 0.05, '%.2f'% y, ha = 'center', va = 'top')
    for x, y in zip(X, Y1):
        ax.text(x + 0.4, - y - 0.05, '%.2f'% y, ha = 'center', va = 'bottom')

    #Pie
    fig.add_subplot(333)
    n=20
    Z=np.ones(n)
    Z[-1]*=2 #
    plt.pie(Z,explode=Z*.05,colors=['%f'%(i/float(n))for i in range(n)],
            labels=['%.2f'%(i/float(n))for i in range(n)])
    #Z先把数组Z传过去，explode表示每个数离扇形的距离，colors表示颜色，%f是灰度颜色，label是标签是颜色的值
    plt.gca().set_aspect('equal')#设置成圆，否则是椭圆
    plt.xticks([]),plt.yticks([])

    #polar
    fig.add_subplot(334,polar=True)
    n=20
    theta=np.arange(0.0,2*np.pi,2*np.pi/n)
    radii=10*np.random.rand(n) #半径
    plt.polar(theta,radii) #除了下边的方法外，这也是一种画polar图的方法
    #plt.plot(theta,radii) #plot画折线图,在add_subplot函数中加入polar=True后，即可画出polar图

    #heatmap
    ax = fig.add_subplot(335)
    from matplotlib import cm #cm是colormap 上色用
    data = np.random.rand(3,3) #定义3*3的随机数
    cmap=cm.Blues
    map=plt.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=0,vmax=1)
    #插值方法，缩放auto，vmin颜色最小值 白色，vmax颜色最大值 前边设置的颜色

    #3D图
    from mpl_toolkits.mplot3d import Axes3D
    ax=fig.add_subplot(336,projection="3d")
    ax.scatter(1,1,3,s=100)

    #hot map
    fig.add_subplot(313)
    def f(x,y):
        return (1-x/2+x**5+y*3)*np.exp(-x**2-y**2)
    n=256
    x=np.linspace(-3,3,n)
    y = np.linspace(-3, 3, n)
    X,Y=np.meshgrid(x,y)
    plt.contourf(X,Y,f(X,Y),8,alpha=.75,camp=plt.cm.hot)
    plt.savefig("./data/fig.png")#将结果保存
    plt.show()

if __name__=="__main__":
    main()