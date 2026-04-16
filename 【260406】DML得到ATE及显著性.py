import pandas as pd
import numpy as np
from econml.dml import LinearDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import warnings

warnings.filterwarnings('ignore')

# 1. 基础设置与数据读取
file_path = r"D:\Desktop\我的竞赛\2026统计建模\2_处理后含控制变量数据.xlsx"
print("正在读取数据...")
df = pd.read_excel(file_path)

# 2. 变量预处理
# A. 对诉求量取对数 log(Y + 1)
y_col = '诉求量_log'
df[y_col] = np.log1p(df['诉求量'])

# B. 处理时间固定效应 (年份在第E列，即索引4)
year_col_name = df.columns[4]
print(f"检测到年份列: {year_col_name}")
year_dummies = pd.get_dummies(df[year_col_name], prefix='year', drop_first=True)
year_dummy_cols = year_dummies.columns.tolist()

# C. 处理地区变量 (地区在第A列/省份)
region_col_name = df.columns[0]
print(f"检测到地区列: {region_col_name}")

# D. 确定控制变量 (X)
t_col = 'post（开放数据平台时间虚拟变量）'
controls = ['地区生产总值(万元)','人口密度(人／平方公里)', '第三产业增加值占GDP比重(%)', '人力资本水平', '每百人公共图书馆藏书(册、件)',
            '传统基础设施', '互联网发展水平', '金融发展水平', '财政压力水平', '科学支出水平','城市发展水平','外商投资水平','生活垃圾无害化处理率(%)','交通便捷程度']
# 最终控制变量 = 自定义变量 + 时间固定效应虚拟变量（保留原有的时间效应）
x_cols = controls + year_dummy_cols

# 整合数据并剔除缺失值
df_final = pd.concat([df, year_dummies], axis=1)
cols_to_use = [y_col, t_col, region_col_name] + x_cols
df_dml = df_final.dropna(subset=cols_to_use).copy()

# 3. 定义全套机器学习模型组合
# 注意：对于 Ridge 分类器，我们使用 LogisticRegressionCV(penalty='l2') 作为等效实现
models_dict = {
    "1. Lasso (L1正则化)": (
        LassoCV(cv=5),
        LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=1000)
    ),
    "2. Ridge (L2正则化)": (
        RidgeCV(cv=5),
        LogisticRegressionCV(cv=5, penalty='l2', solver='lbfgs', max_iter=1000)
    ),
    "3. RandomForest (随机森林)": (
        RandomForestRegressor(n_estimators=50, max_depth=6, n_jobs=-1, random_state=123),
        RandomForestClassifier(n_estimators=50, max_depth=6, n_jobs=-1, random_state=123)
    ),
    "4. GradientBoosting (GBDT)": (
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=123),
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=123)
    ),
    "5. XGBoost (极端梯度提升)": (
        XGBRegressor(n_estimators=50, max_depth=4, n_jobs=-1, random_state=123),
        XGBClassifier(n_estimators=50, max_depth=4, n_jobs=-1, random_state=123, use_label_encoder=False,
                      eval_metric='logloss')
    ),
    "6. LightGBM (轻量梯度提升)": (
        LGBMRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=123, verbose=-1),
        LGBMClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=123, verbose=-1)
    )
}

# ==========================================
# 第一阶段：全样本基准回归 (不分地区)
# ==========================================
print("\n" + "=" * 80)
print("第一阶段：全样本多模型 DML 估计 (控制了时间固定效应)")
print("=" * 80)

Y_all = df_dml[y_col].values
T_all = df_dml[t_col].values
X_all = df_dml[x_cols].values

scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)

global_results = []

for model_name, (mod_y, mod_t) in models_dict.items():
    print(f"正在运行全样本模型: {model_name} ...")
    try:
        est = LinearDML(model_y=mod_y, model_t=mod_t, discrete_treatment=True, cv=5, random_state=456)
        est.fit(Y_all, T_all, X=X_all_scaled, W=None)

        ate = est.const_marginal_ate(X_all_scaled).item()
        summary_table = est.summary().tables[1]
        p_val = float(summary_table.data[1][4])
        stderr = float(summary_table.data[1][2])

        # 标记显著性星号
        stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        global_results.append([model_name, f"{ate:.4f}{stars}", f"{stderr:.4f}", f"{p_val:.4f}"])
    except Exception as e:
        print(f"  [!] {model_name} 运行失败: {e}")
        global_results.append([model_name, "Error", "Error", "Error"])

print("\n【全样本基准结果汇总】")
print(tabulate(global_results, headers=["模型组合", "ATE (log效应)", "标准误", "P值"], tablefmt="grid"))

# ==========================================
# 第二阶段：分地区异质性分析
# ==========================================
print("\n\n" + "=" * 80)
print("第二阶段：分地区异质性分析")
print("=" * 80)

unique_regions = df_dml[region_col_name].unique()
print(f"共涉及 {len(unique_regions)} 个地区...\n")

regional_results = []

for region in unique_regions:
    df_sub = df_dml[df_dml[region_col_name] == region]

    # 样本量过滤：由于加入了大量时间虚拟变量，子样本量要求略高
    if len(df_sub) < 50:
        continue

    print(f"正在分析地区: {region} (有效样本量: {len(df_sub)})")

    Y_sub = df_sub[y_col].values
    T_sub = df_sub[t_col].values
    X_sub = df_sub[x_cols].values

    scaler_sub = StandardScaler()
    X_sub_scaled = scaler_sub.fit_transform(X_sub)

    for model_name, (mod_y, mod_t) in models_dict.items():
        try:
            est = LinearDML(model_y=mod_y, model_t=mod_t, discrete_treatment=True, cv=5, random_state=456)
            est.fit(Y_sub, T_sub, X=X_sub_scaled, W=None)

            ate = est.const_marginal_ate(X_sub_scaled).item()
            summary_table = est.summary().tables[1]
            p_val = float(summary_table.data[1][4])
            stderr = float(summary_table.data[1][2])

            stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            regional_results.append([region, model_name, f"{ate:.4f}{stars}", f"{stderr:.4f}", f"{p_val:.4f}"])

        except Exception as e:
            # 某些高级树模型在小样本下可能会报错（如叶子节点分裂失败），捕获异常保证循环继续
            pass

print("\n【分地区异质性结果汇总】")
print(tabulate(regional_results, headers=["地区", "模型组合", "ATE (log效应)", "标准误", "P值"], tablefmt="grid"))
print("\n注：*** p<0.01, ** p<0.05, * p<0.1。ATE 为对数处理后的效应值。")