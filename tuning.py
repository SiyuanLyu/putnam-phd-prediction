import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Read data
df = pd.read_excel('Putnam PHD.xlsx')

# Feature engineering
college_phd_rate = df.groupby('College')['PHD'].mean()
high_phd_schools = college_phd_rate[college_phd_rate >= 0.60].index.tolist()
df['top_phd_school'] = df['College'].isin(high_phd_schools).astype(int)

features = [
    'Participation_Count', 'Top5_Count', 'Middle_Count',
    'Next9_Count', 'HM_Count', 'Honor_Level', 'top_phd_school'
]

X = df[features]
y = df['PHD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# baseline
xgb_base = XGBClassifier(n_estimators=300, learning_rate=0.05,
                          max_depth=4, random_state=42)
xgb_base.fit(X_train, y_train)
base_auc = roc_auc_score(y_test, xgb_base.predict_proba(X_test)[:,1])
print(f"Baseline Test AUC: {base_auc:.4f}")

# Optuna tuning
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"\nOptimalCV AUC: {study.best_value:.4f}")
print(f"Optimal parameters: {study.best_params}")

best_model = XGBClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
best_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1])

print(f"\nBaseline Test AUC: {base_auc:.4f}")
print(f"Tuned Test AUC:    {best_auc:.4f}")
print(f"Improve:              {best_auc - base_auc:+.4f}")
