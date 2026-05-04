# 2026统计建模
  这次的项目主要是关于数据开放平台对于企业创业环境(本文选取的是企业留言板上的企业诉求量指标)的一次因果实证。

项目流程:

1. 数据:初始数据为[0405]和1_插值,为全国各地各个省份的宏观指标以及政府诉求量
2. 数据清洗:对于缺失值主要采用的是线性插值方法

```python
df[target_cols] = df.groupby('地级市')[target_cols].apply(
    lambda x: x.interpolate(method='linear', limit_direction='both')
).reset_index(level=0, drop=True)
```

3. 特征工程(也就是进行特征变量的筛选,主要选取的是人力资本水平什么的):

```python
df['人力资本水平'] = df['普通高等学校在校学生数(人)'] / df['户籍人口(万人)']
df['传统基础设施'] = np.log(df['公路货运量(万吨)'] + 1) # 加1防止出现log(0)
df['互联网发展水平'] = np.log(df['电信业务收入(万元)'] + 1)
df['金融发展水平'] = df['年末金融机构各项贷款余额(万元)'] / df['地区生产总值(万元)']
df['财政压力水平'] = df['地方财政一般预算内支出(万元)'] / df['地方财政一般预算内收入(万元)']
df['科学支出水平'] = np.log(df['科学支出(万元)'] + 1)
df['城市发展水平'] = np.log(df['人均地区生产总值(元)'])
df['外商投资水平'] = np.log(df['外商投资企业数(个)'])
df['交通便捷程度'] = df['高速公路里程(公里)'] / df['户籍人口(万人)']
```

4.对于清洗好的数据使用dml进行基准回归分析 主要关注的是post这一个系数的正负(如果为正,则起到极好的效应,为负则为负效应,当然前提是显著)采用dml包进行分析

```python
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
est = LinearDML(model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                model_t=RandomForestClassifier(n_estimators=100, random_state=42),
                discrete_treatment=True,
                cv=5,
                random_state=42)
```

5.进行异质性分析(分省份分地区 ，看看经济不同的地方 这个政策是不是都有效),在project.ipynb

6.进行中介效应分析 也就是所谓的机制分析 看看通过什么进行传导(本文选的是营商环境),数据开放$\to$营商环境变好$\to$企业创业活力提升,主要采用的是温忠麟三步法,但是我们这一次两步就解决了,只要系数都显著,就说明中介效应检验有效(mediator.ipynb)

##### Learned and review

这一次项目学到了什么?

1. 因果推断方法(也是景仰已久)

   用的是双重机器学习方法,这个方法实际理解起来很简单,就是先用X去解释post,剥离掉post里面能用X解释的部分,然后再用post和X都扔进预测Y的回归分析里,这样Y的残差就是纯净的Y,第一个分类算法里的post残差就是纯净的post,然后对于这两个残差进行回归就可以得出我们想要的系数了。

2. dml包和插值函数

```python
from econml.dml import LinearDML
df.dropna(subset=[cols])
df[target_cols] = df.groupby('地级市')[target_cols].apply(
    lambda x: x.interpolate(method='linear', limit_direction='both')
```

3. 非参数、半参数、参数统计方法

   **非参数**:不假定表达式形式,全靠数据自行去拟合(也就是使用核函数等 插值什么的,不假定这些变量是一个线性的什么的)

   **半参数**:假定一部分,就比如这个,只假定post是线性的,剩下的X用$g(X)$去拟合,$g()$为选的机器学习算法
   $$
   Y = \alpha * post + g(X) + \varepsilon, \quad \mathbb{E}(\varepsilon \mid D, X) = 0
   $$
   **参数**：典型的就是线性回归,我就是假定你这些就是线性的,最终拟合出这样一个式子就行
   $$
   y = \beta +\beta_1 x_{1}+\beta_{2}x_{2}
   $$

不足:update soon
