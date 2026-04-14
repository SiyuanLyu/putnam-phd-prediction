import pandas as pd
import numpy as np
from scipy import stats

# Read data
df = pd.read_excel('Putnam PHD.xlsx')

# 1. Basic attributes of data
print(f"Number of Obs: {len(df)}")
print(f"\nDistribution of PHD:")
print(df['PHD'].value_counts())
print(f"PHD=1 ratio: {df['PHD'].mean():.1%}")

# 2. Check missing data
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

print("\n--- Missing data ---")
for col in features_of_interest:
    missing = df[col].isna().sum()
    pct = missing / len(df) * 100
    print(f"{col:25s} Missing: {missing:3d} ({pct:.1f}%)")

# 3. t-test
numeric_features = [
    'Participation_Count', 'Top5_Count', 'Middle_Count',
    'Next9_Count', 'HM_Count', 'Honor_Level', 'School_Rank', 'Year'
]

print("\n--- t-test ---")
print(f"{'Feature':25s} {'t-stats':>10} {'p-value':>10} {'Significance':>10}")
print("-" * 60)

for col in numeric_features:
    group0 = df[df['PHD']==0][col]
    group1 = df[df['PHD']==1][col]
    t_stat, p_value = stats.ttest_ind(group0, group1)
    sig = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'n.s.'))
    print(f"{col:25s} {t_stat:>10.3f} {p_value:>10.4f} {sig:>10}")

# 4. Feature engineering
# College → binary variable
college_phd_rate = df.groupby('College')['PHD'].mean()
high_phd_schools = college_phd_rate[college_phd_rate >= 0.60].index.tolist()
df['top_phd_school'] = df['College'].isin(high_phd_schools).astype(int)

# Features
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

print("\n--- List of Features ---")
for f in features:
    print(f"  {f}")
print(f"\nObs: {len(X)}, Features: {len(features)}")

# 5. Partition of data
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
    stratify=y    # keep the ratio of PHD similar in train and test sets
)

print(f"Train set: {len(X_train)} Obs")
print(f"Test set: {len(X_test)} Obs")

# 6. Logistic Regression baseline
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

# 7. XGBoost
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

# 8. Comparison
print("\n--- Comparison ---")
print(f"Logistic Regression Test AUC: {lr_auc:.4f}")
print(f"XGBoost Test AUC:             {xgb_auc:.4f}")
better = "XGBoost" if xgb_auc > lr_auc else "Logistic Regression"
print(f"Better model: {better}")

# 9. SHAP feature importance
import shap
import json

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Calculate the absolute average SHAP value for each feature
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

# 10. Save to S3
import subprocess

# Save SHAP to csv
mean_shap.to_csv('shap_importance.csv', index=False)

# Save prediction results
results = pd.DataFrame({
    'actual': y_test.values,
    'lr_predicted_prob': lr_prob,
    'xgb_predicted_prob': xgb_prob
})
results.to_csv('model_predictions.csv', index=False)

# Save model summary
summary = pd.DataFrame({
    'model': ['Logistic Regression', 'XGBoost'],
    'cv_auc': [lr_cv.mean(), xgb_cv.mean()],
    'test_auc': [lr_auc, xgb_auc]
})
summary.to_csv('model_summary.csv', index=False)

print("Results saved")

subprocess.run(['aws', 's3', 'cp', 'shap_importance.csv', 's3://putnam-ml-siyuan-2026/'])
subprocess.run(['aws', 's3', 'cp', 'model_predictions.csv', 's3://putnam-ml-siyuan-2026/'])
subprocess.run(['aws', 's3', 'cp', 'model_summary.csv', 's3://putnam-ml-siyuan-2026/'])
subprocess.run(['aws', 's3', 'cp', 'Putnam PHD.xlsx', 's3://putnam-ml-siyuan-2026/'])

print("All documents uploaded to S3")
