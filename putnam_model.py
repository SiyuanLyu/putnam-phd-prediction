import pandas as pd
import numpy as np
from scipy import stats

# 读入数据
df = pd.read_excel('Putnam PHD.xlsx')

# 第一步：基本情况
print(f"总样本量: {len(df)}")
print(f"\nPHD分布:")
print(df['PHD'].value_counts())
print(f"PHD=1比例: {df['PHD'].mean():.1%}")

# 第二步：缺失值检查
features_of_interest = [
    'Participation_Count',
    'Top5_Count',
    'Middle_Count',
    'Next9_Count',
    'HM_Count',
    'Honor_Level',
    'School_Rank',
    'Year',
    'College'
]

print("\n--- 缺失值检查 ---")
for col in features_of_interest:
    missing = df[col].isna().sum()
    pct = missing / len(df) * 100
    print(f"{col:25s} 缺失: {missing:3d} ({pct:.1f}%)")

# 第三步：t检验
numeric_features = [
    'Participation_Count', 'Top5_Count', 'Middle_Count',
    'Next9_Count', 'HM_Count', 'Honor_Level', 'School_Rank', 'Year'
]

print("\n--- t检验 ---")
print(f"{'Feature':25s} {'t统计量':>10} {'p值':>10} {'显著性':>10}")
print("-" * 60)

for col in numeric_features:
    group0 = df[df['PHD']==0][col]
    group1 = df[df['PHD']==1][col]
    t_stat, p_value = stats.ttest_ind(group0, group1)
    sig = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'n.s.'))
    print(f"{col:25s} {t_stat:>10.3f} {p_value:>10.4f} {sig:>10}")

# 第四步：特征工程
# College → binary变量
college_phd_rate = df.groupby('College')['PHD'].mean()
high_phd_schools = college_phd_rate[college_phd_rate >= 0.60].index.tolist()
df['top_phd_school'] = df['College'].isin(high_phd_schools).astype(int)

# 最终特征
features = [
    'Participation_Count',
    'Top5_Count',
    'Middle_Count',
    'Next9_Count',
    'HM_Count',
    'Honor_Level',
    'top_phd_school'
]

X = df[features]
y = df['PHD']

print("\n--- 最终特征列表 ---")
for f in features:
    print(f"  {f}")
print(f"\n样本量: {len(X)}, 特征数: {len(features)}")

# 第五步：切分数据
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y    # 保证训练集和测试集PHD比例一致
)

print(f"训练集: {len(X_train)}样本")
print(f"测试集: {len(X_test)}样本")

# 第六步：Logistic Regression baseline
print("\n--- Logistic Regression ---")

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])

lr_cv = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV AUC: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

lr_pipeline.fit(X_train, y_train)
lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_prob)
print(f"Test AUC: {lr_auc:.4f}")

# 第七步：XGBoost
print("\n--- XGBoost ---")

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc'
)

xgb_cv = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV AUC: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

xgb_model.fit(X_train, y_train)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_prob)
print(f"Test AUC: {xgb_auc:.4f}")

# 第八步：对比
print("\n--- 模型对比 ---")
print(f"Logistic Regression Test AUC: {lr_auc:.4f}")
print(f"XGBoost Test AUC:             {xgb_auc:.4f}")
better = "XGBoost" if xgb_auc > lr_auc else "Logistic Regression"
print(f"更好的模型: {better}")

# 第九步：SHAP feature importance
import shap
import json

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# 计算每个特征的平均绝对SHAP值
mean_shap = pd.DataFrame({
    'feature': features,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\n--- SHAP Feature Importance ---")
print(f"{'Feature':25s} {'Importance':>12}")
print("-" * 40)
for _, row in mean_shap.iterrows():
    bar = '█' * int(row['importance'] * 500)
    print(f"{row['feature']:25s} {row['importance']:>12.4f}  {bar}")

# 第十步：保存结果到S3
import subprocess

# 保存SHAP结果到csv
mean_shap.to_csv('shap_importance.csv', index=False)

# 保存模型预测结果
results = pd.DataFrame({
    'actual': y_test.values,
    'lr_predicted_prob': lr_prob,
    'xgb_predicted_prob': xgb_prob
})
results.to_csv('model_predictions.csv', index=False)

# 保存模型性能摘要
summary = pd.DataFrame({
    'model': ['Logistic Regression', 'XGBoost'],
    'cv_auc': [lr_cv.mean(), xgb_cv.mean()],
    'test_auc': [lr_auc, xgb_auc]
})
summary.to_csv('model_summary.csv', index=False)

print("结果已保存到本地")

# 上传到S3
subprocess.run(['aws', 's3', 'cp', 'shap_importance.csv', 's3://putnam-ml-siyuan-2026/'])
subprocess.run(['aws', 's3', 'cp', 'model_predictions.csv', 's3://putnam-ml-siyuan-2026/'])
subprocess.run(['aws', 's3', 'cp', 'model_summary.csv', 's3://putnam-ml-siyuan-2026/'])
subprocess.run(['aws', 's3', 'cp', 'Putnam PHD.xlsx', 's3://putnam-ml-siyuan-2026/'])

print("所有文件已上传到S3")
