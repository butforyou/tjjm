import pandas as pd
import numpy as np
import statsmodels.api as sm
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ==========================================
# 1. 读取数据
# ==========================================
file_path = r"【0405】左连结数据整合 - 副本.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# ==========================================
# 2. 面板数据缺失值处理（按地级市分组插值）
# ==========================================
# 提取需要插值的列：诉求量(被解释变量) 和 R列到AH列(控制变量基础数据)
# 这里以列名或列索引来提取，假设17:34对应R到AJ列
target_cols = ['诉求量'] + df.columns[17:36].tolist()

# 按照地级市分组，使用线性插值处理时间序列上的缺失值
# limit_direction='both' 确保时间序列头尾的缺失值也能向前或向后填充
df[target_cols] = df.groupby('地级市')[target_cols].apply(
    lambda x: x.interpolate(method='linear', limit_direction='both')
).reset_index(level=0, drop=True)

df.to_excel("1_插值补充后数据.xlsx", index=False)

# ==========================================
# 3. 构造文献要求的控制变量
# ==========================================
# 注意：等号右侧的 ['列名'] 需要你对照生成的 "1_插值补充后数据.xlsx" 里的实际基础表头进行替换
df['人力资本水平'] = df['普通高等学校在校学生数(人)'] / df['户籍人口(万人)']
df['传统基础设施'] = np.log(df['公路货运量(万吨)'] + 1) # 加1防止出现log(0)
df['互联网发展水平'] = np.log(df['电信业务收入(万元)'] + 1)
df['金融发展水平'] = df['年末金融机构各项贷款余额(万元)'] / df['地区生产总值(万元)']
df['财政压力水平'] = df['地方财政一般预算内支出(万元)'] / df['地方财政一般预算内收入(万元)']
df['科学支出水平'] = np.log(df['科学支出(万元)'] + 1)
df['城市发展水平'] = np.log(df['人均地区生产总值(元)'])
df['外商投资水平'] = np.log(df['外商投资企业数(个)'])
df['交通便捷程度'] = df['高速公路里程(公里)'] / df['户籍人口(万人)']
# 将构造好的控制变量放入列表
controls = ['地区生产总值(万元)','人口密度(人／平方公里)', '第三产业增加值占GDP比重(%)', '人力资本水平', '每百人公共图书馆藏书(册、件)',
            '传统基础设施', '互联网发展水平', '金融发展水平', '财政压力水平', '科学支出水平','城市发展水平','外商投资水平','生活垃圾无害化处理率(%)','交通便捷程度']

# 剔除在构造变量后仍然存在缺失值（如缺乏某城市整体面积数据导致无法插值）的样本
df.dropna(subset=['诉求量', 'post（开放数据平台时间虚拟变量）'] + controls, inplace=True)
df.to_excel("2_处理后含控制变量数据.xlsx", index=False)

# ==========================================
# 4. 双重机器学习 (DML) 基准回归
# ==========================================
df['诉求量_log'] = np.log1p(df['诉求量'])
Y = df['诉求量_log'].values
T = df['post（开放数据平台时间虚拟变量）'].values
X = df[controls].values

# 提取城市与年份固定效应，转化为虚拟变量矩阵，作为混杂因素 W
# DML第一阶段会利用机器学习算法剥离W对Y和T的影响，从而获得纯净的因果效应
W_df = pd.get_dummies(df[['地级市', '年份_final']], columns=['地级市', '年份_final'], drop_first=True)
W = W_df.values

# 初始化 DML 模型，主模型和倾向得分模型均采用随机森林
# 这与你之前使用 DML 测度公共数据开放效应的方法论一致
est = LinearDML(model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                model_t=RandomForestClassifier(n_estimators=100, random_state=42),
                discrete_treatment=True,
                cv=5,
                random_state=42)

print("正在拟合 DML 基准回归模型，可能需要几分钟时间...")
est.fit(Y, T, X=X, W=W)

# 提取回归摘要并导出
summary_df = est.summary().tables[0]
pd.DataFrame(summary_df.data).to_excel("3_DML基准回归结果.xlsx", index=False, header=False)


# ==========================================
# 5. 平行趋势检验 (事件研究法)
# ==========================================
# DML中进行平行趋势相对复杂，这里采用经典的加入双向固定效应的事件研究法
# 假设你的数据中存在标明该城市实施政策时间的列 '初次开放数据平台年份'
# 如果部分城市未开放，该列可能为空，需做替换（如替换为9999并过滤）
df_treated = df[df['初次开放数据平台年份'].notna()].copy()
df_treated['相对时间'] = df_treated['年份_final'] - df_treated['初次开放数据平台年份']

# 生成相对年份的虚拟变量，以政策实施前一期（time_-1）作为基准组，避免完全多重共线性
time_dummies = pd.get_dummies(df_treated['相对时间'], prefix='time', dtype=int)
if 'time_-1.0' in time_dummies.columns:
    time_dummies.drop(columns=['time_-1.0'], inplace=True)
elif 'time_-1' in time_dummies.columns:
    time_dummies.drop(columns=['time_-1'], inplace=True)

# 组装回归所需数据结构：被解释变量、时间虚拟变量、控制变量、固定效应虚拟变量
df_pt = pd.concat([df_treated['诉求量_log'], time_dummies, df_treated[controls], pd.get_dummies(df_treated[['地级市', '年份_final']], drop_first=True, dtype=int)], axis=1)

# 构建模型变量并加入常数项
X_pt = sm.add_constant(df_pt.drop(columns=['诉求量_log']))
Y_pt = df_pt['诉求量_log']

# 考虑到随机误差项的潜在相关性，使用聚类稳健标准误（聚类到地级市层面）
model_pt = sm.OLS(Y_pt, X_pt).fit(cov_type='cluster', cov_kwds={'groups': df_treated['地级市']})

# 整理时间虚拟变量的系数、P值及置信区间
pt_results = pd.DataFrame({
    'Coef': model_pt.params,
    'P-value': model_pt.pvalues,
    'CI Lower (2.5%)': model_pt.conf_int()[0],
    'CI Upper (97.5%)': model_pt.conf_int()[1]
})

# 仅筛选展示代表平行趋势的时间虚拟变量结果
pt_results = pt_results[pt_results.index.str.contains('time_')]
pt_results.to_excel("4_平行趋势检验结果.xlsx")

print("所有数据处理与模型检验完毕，四个Excel文件已生成。")